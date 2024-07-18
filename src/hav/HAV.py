
import agentpy as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .methods import *
from .utils import *

class HAV:
    Results:dict
    ABM:ap.Model
    AgentLevel:dict
    ModelLevel:dict
    OutputLevel:dict
    BenchMark:dict

    def __init__(self, model:ap.Model, B:dict[str,float]):
        self.ABM=model
        self.Results={'Agent':None, 'Model':None, 'Output':None}
        self.BenchMark=B

        self.AgentLevel={'I':None, 'O':None, 'B':{'I':None, 'O':None}}
        self.ModelLevel={'I':None, 'O':None, 'B':{'I':None, 'O':None}}
        self.OutputLevel={'O':None, 'B':None}    
        pass
    
    def get_data(self, O:list[str], rep_t:int, level:str):
        data={}
        for _ in range(rep_t):
            abm_res=run_ABM(self.ABM)
            abm_vars=abm_res.variables[list(abm_res.variables)[int(level=='agent')]]
            if isinstance(abm_vars, pd.DataFrame):
                for att in O: 
                    item=abm_vars[att].values
                    if att not in data.keys(): data.update({att: item})
                    else: data[att]+=item
        for att in data.keys(): data[att]=flatten(data[att]/rep_t)
        return data

    def validate_agent_level(self, method:str, I:list[str], O:list[str], I_range:dict[str:list]=None):
        
        self.AgentLevel['I']={att:flatten(self.ABM.p[att]) for att in I}
        self.AgentLevel['O']=self.get_data(O, 10, 'agent')
        self.AgentLevel['B']['I']={att:flatten(self.BenchMark['A']['I'][att]) for att in I}
        self.AgentLevel['B']['O']={att:flatten(self.BenchMark['A']['O'][att]) for att in O}

        res, details = None, None
        if method=='MSE test':
            res, details = MSE(
                R=self.AgentLevel['I'],
                B=self.AgentLevel['B']['I']
            )

        elif method=='ks test':
            res, details = ks_test(
                R=self.AgentLevel['I'],
                B=self.AgentLevel['B']['I']
            )

        self.Results['Agent'] = Result(method, res, I, details)
        self.Results['Agent'].level='Agent'

    def calibration(self, I_range:dict[str:list[float]]):
        I=self.AgentLevel['I']
        B=self.AgentLevel['B']
        _, par_pool, _ = surrogate_analysis(self.ABM, I, B, I_range)
        return par_pool

    def validate_model_level(self, method:str, I:list[str], O:list[str], I_range:dict[str:list]=None):
        
        self.ModelLevel['O']=self.get_data(O, 10, 'model')
        self.ModelLevel['I']={att:flatten(self.ABM.p[att]) for att in I}
        self.ModelLevel['B']['I']={att:flatten(self.BenchMark['M']['I'][att]) for att in I}
        self.ModelLevel['B']['O']={att:flatten(self.BenchMark['M']['O'][att]) for att in O}

        res, details = None, None
        if method=='causal analysis':
            res, details = causal_analysis(
                R=self.ModelLevel['O'],
                B=self.ModelLevel['B']['O']
                )
            
        elif method=='bayesian':
            res, details = bayesian(
                ABM=self.ABM,
                R=self.ModelLevel['O'],
                B=self.ModelLevel['B']['O'],
                I_Range=I_range
                )
            
        elif method=='ks test':
            res, details = ks_test(
                R=self.ModelLevel['O'],
                B=self.ModelLevel['B']['O']
            )

        self.Results['Model'] = Result(method, res, O, details)
        self.Results['Model'].level='model'
    
    def validate_output_level(self, method:str, A:list[str], M:list[str], I_range:dict[str:list]=None):
        self.OutputLevel['O'], self.OutputLevel['B'] = {}, {}
        aData, mData = self.get_data(A, 10, 'agent'), self.get_data(M, 10, 'model')
        self.OutputLevel['O'].update(aData)
        self.OutputLevel['O'].update({att: mData[att][-1:] for att in M})
        self.OutputLevel['B'].update(self.BenchMark['A']['O'])
        self.OutputLevel['B'].update({att: self.BenchMark['M']['O'][att][-1:] for att in M})

        res, details = None, None
        if method=='ks test':
            res, details = ks_test(
                R=self.OutputLevel['O'],
                B=self.OutputLevel['B']
            )
            
        elif method=='variance':
            res, details = variance_test(
                R=self.OutputLevel['O'],
                B=self.OutputLevel['B']
            )
            
        elif method=='skewness':
            res, details = skewness_test(
                R=self.OutputLevel['O'],
                B=self.OutputLevel['B']
            )
            
        elif method=='kurtosis':
            res, details = kurtosis_test(
                R=self.OutputLevel['O'],
                B=self.OutputLevel['B']
            )
            
        elif method=='sensitivity analysis':
            res, details = sentivity_analysis(
                ABM=self.ABM,
                B=self.OutputLevel['B'],
                I_Range=I_range
            )

        self.Results['Output'] = Result(method, res, A+M, details)
        self.Results['Output'].level='output'



class Result:
    method:str
    status:bool
    details:dict[str,str]
    time:datetime
    level:str

    def __init__(self, method, res, atts, details) -> None:
        self.method=method
        self.details={atts[i]: details[i] for i in range(len(atts))}
        self.status=res
        self.time=datetime.now()
        self.level=None
        pass

    def print_result(self):
        print(f"\n{self.level} level {'passed' if self.status else 'failed'} {self.method} validation ({self.time})")
        print(' '*3+'details:')
        for att in self.details.keys(): print(' '*3+self.details[att])
        print('-'*100)