import agentpy as ap
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hav.HAV import HAV

class FNNR_data(ap.Model):
    '''
    We have run the model and stored the results in CSV files. 
    In order to efficiently demonstrate the validation process, we design this class as a formal model that only outputs the results of the run.
    If you need to view the detailed running process, you can do so at test/Case3_FNNR_data/FNNR/ABM Create a virtual environment to execute validation.py (please ensure that Mesa==0.8.3 and Tornado==4.5.2 in the virtual environment)
    '''
    def get_data(self, level:str):
        if level == 'agent':
            return pd.read_csv('test/Case3_FNNR_data/samples_A.csv')
        elif level == 'model':
            return pd.read_csv('test/Case3_FNNR_data/samples_M.csv')
        elif level == 'output':
            return pd.read_csv('test/Case3_FNNR_data/samples_O.csv')

class validate_FNNR_model():
    '''This example encapsulates the provided parameters required in HAV, 
        and demonstrates a process of validating FNNR Model'''

    # Benchmark data used as a control reference in validation
    Benchmark = {
        'expected_p': 0.05,
        'expected_sign_A': 1,
        'expected_sign_M': -1,
        'expected_sign_O': 1,
    }

    # Declare the parameter names required in Agent(A), Model(M) and Output(O) level, including Simulated data(S) and Benchmark data(B)
    A = {'S': ['flat_comp', 'gtgp_participation'], 'B': ['expected_p', 'expected_sign_A']}
    M = {'S': ['fertility_rate', 'final_monkey_num'], 'B': ['expected_p', 'expected_sign_M']}
    O = {'S': ['avg_forest', 'avg_monkey'], 'B': ['expected_p', 'expected_sign_O']}

    def __init__(self):
        '''Execute HAV directly to visually demonstrate the process of validation'''    

        # 1 Create a HAV object
        my_hav = HAV(FNNR_data, None, self.A, self.M, self.O)

        # 2 Transmit validation methods that required in each level
        my_hav.validate('Agent', self.REGRESSION_ANALYSIS_agent_lv)
        my_hav.validate('Model', self.REGRESSION_ANALYSIS_model_lv)
        my_hav.validate('Output', self.PEARSON_CORRELATION_ANALYSIS)

        # 3 Recored the results
        logger.info(my_hav.details)

        pass

    def REGRESSION_ANALYSIS_agent_lv(self, hav: HAV):
        '''Validate the influence of GTGP Compensation Amount on Participation Rate'''
        A_samples = hav.ABM.get_data('agent')
        x = A_samples[self.A['S'][0]]
        y = A_samples[self.A['S'][1]]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        condition1 = model.pvalues[self.A['S'][0]] < self.Benchmark[self.A['B'][0]]
        condition2 = model.params[self.A['S'][0]] * self.Benchmark[self.A['B'][1]] > 0
        return condition1 and condition2
    
    def REGRESSION_ANALYSIS_model_lv(self, hav: HAV):
        '''Validate the influence of Human Growth Rate on the Number of Monkeys'''
        M_samples = hav.ABM.get_data('model')
        x = M_samples[self.M['S'][0]]
        y = M_samples[self.M['S'][1]]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        condition1 = model.pvalues[self.M['S'][0]] < self.Benchmark[self.M['B'][0]]
        condition2 = model.params[self.M['S'][0]] * self.Benchmark[self.M['B'][1]] > 0
        return condition1 and condition2

    def PEARSON_CORRELATION_ANALYSIS(self, hav: HAV):
        '''Validate the correlation between Vegetation Area and Monkey Population'''
        O_samples = hav.ABM.get_data('output')
        x = O_samples[self.O['S'][0]]
        y = O_samples[self.O['S'][1]]
        corr_coef, p_value = pearsonr(x, y)
        condition1 = p_value < self.Benchmark[self.O['B'][0]]
        condition2 = corr_coef * self.Benchmark[self.O['B'][1]] > 0
        return condition1 and condition2

if __name__ == '__main__':
    validate_FNNR_model()
