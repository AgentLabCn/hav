from test_model import Environment
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hav.HAV import HAV, Result

class TestHAV():
    def __init__(self):

        # User provides a list of model parameters to be validated   
        self.att_agent = {'I':['tran_mat','init_wealth','con_prop'], 'O':['ev_wealth']}
        self.att_model = {'I':['init_a_ratio','init_b_ratio','init_c_ratio'], 'O':['gini','a_ratio','b_ratio','c_ratio']}
        self.att_output = {'A':self.att_agent['O'], 'M':self.att_model['O']}
        
        # Benchmark for input data
        benchmark_par = {
            'ag_num': 100,
            'init_wealth': [100000, 30000, 3000],
            'init_a_ratio': 0.1,
            'init_b_ratio': 0.2,
            'init_c_ratio': 0.7,
            'tran_mat': [[0.2,0.7,0.1], [0.3,0.5,0.2], [0.1,0.8,0.1]],
            'con_prop': [0.3, 0.2, 0.1],
            'steps': 100,
        }

        # Set up Agent-Based Model with parameter tables
        self.model = Environment(parameters = benchmark_par)

        # initiate benchmark data
        B_agent = pd.read_csv('test/data/AgentBenchmark.csv').to_dict()
        B_model = pd.read_csv('test/data/ModelBenchmark.csv').to_dict()
        self.benchmark = {
            'A':{
                'I':{att: benchmark_par[att] for att in self.att_agent['I']},
                'O':{att: list(B_agent[att].values()) for att in self.att_agent['O']}
                },
            'M':{
                'I':{att: benchmark_par[att] for att in self.att_model['I']},
                'O':{att: list(B_model[att].values()) for att in self.att_model['O']}
                },
        }

        # Set up a HAV object 
        self.HAV = HAV(self.model, self.benchmark)

        pass

    def test_agent_level(self):
        self.HAV.validate_agent_level('MSE test', self.att_agent['I'], self.att_agent['O'])

    def test_model_level(self):
        self.HAV.validate_model_level('causal analysis', self.att_model['I'], self.att_model['O'])

    def test_output_level(self):
        self.HAV.validate_output_level('ks test', self.att_output['A'], self.att_output['M'])

if __name__ == '__main__':
    test = TestHAV()
    test.test_agent_level()
    test.test_model_level()
    test.test_output_level()
    for _, res in test.HAV.Results.items():
        if isinstance(res, Result):
            res.print_result()
    pass