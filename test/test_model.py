import agentpy as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Person(ap.Agent):

    """ An agent with wealth """
    xv_tran_mat:list
    xv_con_prop:list
    ev_wealth:float
    ev_level:str

    def setup(self,init_level,init_wealth,tran_mat,con_prop):
        self.ev_wealth=init_wealth
        self.ev_level=init_level
        self.xv_tran_mat=tran_mat
        self.xv_con_prop=con_prop
        
    def wealth_transfer(self):
        if self.ev_wealth > 0:
            levels=list(self.model.ev_level_list.keys())
            for lv in levels:
                if len(self.model.ev_level_list[lv])>0:
                    person=self.model.ev_level_list[lv].random(1)
                    tran_wealth=np.round(self.ev_wealth*self.xv_con_prop[levels.index(self.ev_level)]*\
                                               self.xv_tran_mat[levels.index(self.ev_level)][levels.index(person.ev_level)])
                    person.ev_wealth+=tran_wealth
                    self.ev_wealth-=tran_wealth    
            pass

    def get_level(self):
        old_level=self.ev_level
        if self.ev_wealth<=5000: self.ev_level='c'
        elif self.ev_wealth<=50000: self.ev_level='b'
        else: self.ev_level='a'

        if old_level!=self.ev_level:
            self.model.ev_level_list[old_level].remove(self)
            self.model.ev_level_list[self.ev_level].append(self)


class Environment(ap.Model):

    """ A simple model of random wealth transfers """
    xv_ag_num:int
    xv_init_wealth_ratio:list
    ev_wealth_ratio:list
    ev_level_list:dict

    def setup(self):
        self.xv_ag_num=self.p['ag_num']
        self.xv_init_wealth_ratio=[self.p['init_a_ratio'], self.p['init_b_ratio'], self.p['init_c_ratio']]
        self.agents = ap.AgentList(self,[])
        self.ev_level_list={'a':ap.AgentList(self,[]), 'b':ap.AgentList(self,[]), 'c':ap.AgentList(self,[])}
        for i in range(self.xv_ag_num):
            wealth, level = 0, 0
            if i<sum(self.xv_init_wealth_ratio[:1])*self.xv_ag_num:
                wealth, level = self.p['init_wealth'][0], 'a'
            elif i<sum(self.xv_init_wealth_ratio[:2])*self.xv_ag_num:
                wealth, level = self.p['init_wealth'][1], 'b'
            else:
                wealth, level = self.p['init_wealth'][2], 'c'
            ag=Person(self, init_level=level, init_wealth=wealth, tran_mat=self.p['tran_mat'], con_prop=self.p['con_prop'])
            self.agents.append(ag)
            self.ev_level_list[level].append(ag)

    def step(self):
        self.agents.wealth_transfer()
        self.agents.get_level()

    def update(self):
        GINI=self.gini(self.agents.ev_wealth)
        [A_RATIO, B_RATIO, C_RATIO]=[len(self.ev_level_list[key])/self.xv_ag_num\
                                      for key in self.ev_level_list.keys()]
        self.record('a_ratio',A_RATIO)
        self.record('b_ratio',B_RATIO)
        self.record('c_ratio',C_RATIO)
        self.record('gini',GINI)

    def end(self):
        self.agents.record('ev_wealth')
        self.agents.record('ev_level')

    def gini(self,x):
        x = np.array(x)
        mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
        rmad = mad / np.mean(x)  # Relative mean absolute difference
        return 0.5 * rmad       
    
if __name__=='__main__':
    benchmark_par={
            'ag_num': 100,
            'init_wealth': [100000, 30000, 3000],
            'init_a_ratio': 0.1,
            'init_b_ratio': 0.2,
            'init_c_ratio': 0.7,
            'tran_mat': [[0.2,0.7,0.1], [0.3,0.5,0.2], [0.1,0.8,0.1]],
            'con_prop': [0.3, 0.2, 0.1],
            'steps': 100,
        }
    atts=['gini','a_ratio','b_ratio','c_ratio']
    df=pd.DataFrame([[0 for _ in range(len(atts))] for _ in range(101)], columns=atts, index=[i for i in range(101)])
    model=Environment(parameters=benchmark_par)

    for _ in range(10):
        model.sim_reset()
        result=model.run()
        model_data=result.variables[list(result.variables)[0]]
        for att in atts:
            df[att]+=model_data[att]
    
    for att in atts: 
        df[att]/=10
    print(df.values)
    df.to_csv('test\data\ModelBenchmark.csv')
    
    


    pass