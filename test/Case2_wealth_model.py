import agentpy as ap
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hav.HAV import HAV
from src.hav.methods import causal_analysis

class Person(ap.Agent): 
    """ An agent with wealth """
    xv_tran_mat: list
    xv_con_prop: list
    ev_wealth: float
    ev_level: str

    def setup(self,init_level,init_wealth,tran_mat,con_prop): 
        self.ev_wealth = init_wealth
        self.ev_level = init_level
        self.xv_tran_mat = tran_mat
        self.xv_con_prop = con_prop
        
    def wealth_transfer(self): 
        if self.ev_wealth > 0: 
            levels = list(self.model.ev_level_list.keys())
            for lv in levels: 
                if len(self.model.ev_level_list[lv]) > 0: 
                    person = self.model.ev_level_list[lv].random(1)
                    tran_wealth = np.round(self.ev_wealth * self.xv_con_prop[levels.index(self.ev_level)] * \
                                               self.xv_tran_mat[levels.index(self.ev_level)][levels.index(person.ev_level)])
                    person.ev_wealth += tran_wealth
                    self.ev_wealth -= tran_wealth    
                    self.model.ev_trans_dct[self.ev_level][lv] += tran_wealth
            pass

    def get_level(self): 
        old_level = self.ev_level
        if self.ev_wealth <= 5000: self.ev_level = 'c'
        elif self.ev_wealth <= 50000: self.ev_level = 'b'
        else: self.ev_level = 'a'

        if old_level != self.ev_level: 
            self.model.ev_level_list[old_level].remove(self)
            self.model.ev_level_list[self.ev_level].append(self)


class WealthModel(ap.Model): 
    """ A simple model of random wealth transfers """
    xv_ag_num: int
    xv_init_wealth_ratio: list
    ev_wealth_ratio: list
    ev_level_list: dict
    ev_trans_dct: dict
    
    def setup(self): 
        self.xv_ag_num = self.p['ag_num']
        self.xv_init_wealth_ratio = [self.p['init_a_ratio'], self.p['init_b_ratio'], self.p['init_c_ratio']]
        self.agents = ap.AgentList(self,[])
        self.ev_level_list = {'a': ap.AgentList(self,[]), 'b': ap.AgentList(self,[]), 'c': ap.AgentList(self,[])}
        self.ev_trans_dct = {lv1: {lv2: 0 for lv2 in ['a', 'b', 'c']} for lv1 in ['a', 'b', 'c']}

        for i in range(self.xv_ag_num): 
            wealth, level = 0, 0
            if i < sum(self.xv_init_wealth_ratio[: 1]) * self.xv_ag_num: 
                wealth, level = self.p['init_wealth'][0], 'a'
            elif i < sum(self.xv_init_wealth_ratio[: 2]) * self.xv_ag_num: 
                wealth, level = self.p['init_wealth'][1], 'b'
            else: 
                wealth, level = self.p['init_wealth'][2], 'c'
            ag = Person(self, init_level = level, init_wealth = wealth, tran_mat = self.p['tran_mat'], con_prop = self.p['con_prop'])
            self.agents.append(ag)
            self.ev_level_list[level].append(ag)

    def step(self): 
        self.agents.wealth_transfer()
        self.agents.get_level()

    def update(self): 
        GINI = self.gini(self.agents.ev_wealth)
        [A_RATIO, B_RATIO, C_RATIO] = [len(self.ev_level_list[key])/self.xv_ag_num\
                                      for key in self.ev_level_list.keys()]
        for lv1 in self.ev_level_list.keys():
            for lv2 in self.ev_level_list.keys():
                self.record(f'{lv1}2{lv2}', self.ev_trans_dct[lv1][lv2])
                self.ev_trans_dct[lv1][lv2] = 0
        self.record('a_ratio', A_RATIO)
        self.record('b_ratio', B_RATIO)
        self.record('c_ratio', C_RATIO)
        self.record('gini', GINI)

    def end(self): 
        for agent in self.agents:
            agent.record('final_wealth', agent.ev_wealth)
            agent.record('final_level', agent.ev_level)

    def gini(self,x): 
        x = np.array(x)
        mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
        rmad = mad / np.mean(x)  # Relative mean absolute difference
        return 0.5 * rmad       


class validate_wealth_model():
    '''This example encapsulates the provided parameters required in HAV, 
        and demonstrates a process of validating Wealth Model'''
    
    # Paramenters required in Traffic Model
    p = {    
        'ag_num': 100,
        'init_wealth': [100000, 30000, 3000],
        'init_a_ratio': 0.1,
        'init_b_ratio': 0.2,
        'init_c_ratio': 0.7,
        'tran_mat': [[0.2,0.7,0.1], [0.3,0.5,0.2], [0.1,0.8,0.1]],
        'con_prop': [0.3, 0.2, 0.1],
        'steps': 100
    }
    
    # Benchmark data used as a control reference in validation
    Benchmark = {
        'final_wealth': pd.read_csv('test/Case2_wealth_data/AgentBenchmark.csv')['final_wealth'].tolist(),
        'gini': pd.read_csv('test/Case2_wealth_data/ModelBenchmark.csv')['gini'].tolist(),
        'a_ratio': pd.read_csv('test/Case2_wealth_data/ModelBenchmark.csv')['a_ratio'].tolist(),
        'b_ratio': pd.read_csv('test/Case2_wealth_data/ModelBenchmark.csv')['b_ratio'].tolist(),
        'c_ratio': pd.read_csv('test/Case2_wealth_data/ModelBenchmark.csv')['c_ratio'].tolist(),
    }
    Benchmark.update({
        idx: pd.read_csv('test/Case2_wealth_data/ModelBenchmark.csv')[idx].tolist() 
        for idx in [f'{lv1}2{lv2}' for lv1 in ['a','b','c']  for lv2 in ['a','b','c']]
    })

    # Declare the parameter names required in Agent(A), Model(M) and Output(O) level, including Simulated data(S) and Benchmark data(B)
    A = {'S': ['final_wealth'], 'B': ['final_wealth']}
    M = {'S': [f'{lv1}2{lv2}' for lv1 in ['a','b','c']  for lv2 in ['a','b','c']], 
         'B': [f'{lv1}2{lv2}' for lv1 in ['a','b','c']  for lv2 in ['a','b','c']]}
    O = {'S': ['final_wealth', 'gini','a_ratio','b_ratio','c_ratio'], 
         'B': ['final_wealth', 'gini','a_ratio','b_ratio','c_ratio']}
    
    def __init__(self):        
        '''Execute HAV directly to visually demonstrate the process of validation'''
        np.random.seed(123)

        # 1 Create a HAV object
        my_hav = HAV(WealthModel, self.p, self.A, self.M, self.O)

        # 2 Transmit validation methods that required in each level
        my_hav.validate('Agent', self.test_WEALTH_dist)
        my_hav.validate('Model', self.test_CAUSALITY)
        my_hav.validate('Output', self.test_all_OUTPUT_data)
        
        # 3 Recored the results
        logger.info(my_hav.details)


    # The following are validation methods that defined by user
    # (For different models and users, the following content can be freely edited)

    def test_WEALTH_dist(self, hav:HAV):
        '''Validate the rationality of wealth distribution with MSE'''
        result = hav.run_model(self.p)
        sim_wealth = result.variables.Person['final_wealth'].T.tolist()
        bench_wealth = self.Benchmark['final_wealth']
        mse = np.mean([(sorted(sim_wealth)[i] - sorted(bench_wealth)[i]) ** 2 for i in range(len(sim_wealth))])
        nmse = mse / np.mean(bench_wealth) ** 2
        return nmse <= 0.2
    
    def test_CAUSALITY(self, hav:HAV):
        '''Validate structural similarity between Causal Graphs (DAGs) of Simulated data and Benchmark data'''
        result = hav.run_model(self.p)
        S = result.variables.WealthModel[self.M['S']]
        B = pd.DataFrame(np.array([v for k, v in self.Benchmark.items() if k in self.M['B']]).T)
        graph_similarity = causal_analysis(S, B)
        return graph_similarity >= 0.8
    
    def test_all_OUTPUT_data(self, hav:HAV):
        '''Validate the rationality of all output data with Kolmogorov-Smirnov Test'''
        result = hav.run_model(self.p)
        for indicator in self.O['B']:
            S = result.variables.WealthModel[indicator].tolist() if indicator in result.variables.WealthModel.keys() \
                else result.variables.Person[indicator].T.tolist()
            B = self.Benchmark[indicator]
            _, p_value = ks_2samp(S, B)
            if not (p_value > 0.05): 
                return False
            
        return True


if __name__ == '__main__':
    validate_wealth_model()
