
import agentpy as ap

class HAV: 
    '''A tool to guide and assist in the process of Hierarchical ABM Validation'''
    Results: dict   # to store the validation results of each level (Agent, Model, Output)
    ABM: ap.Model   # target model

    def __init__(self, model:ap.Model, p:dict, A:dict[str, list[str]], M:dict[str, list[str]], O:dict[str, list[str]]):
        '''Users need to declare the target model and indicators that to validated in each level'''
        self.Results = {'Agent': {'result': None, '* Simulation Data': A['S'], '* Benchmark Data': A['B'], '* Validation Method': None}, 
                        'Model': {'result': None, '* Simulation Data': M['S'], '* Benchmark Data': M['B'], '* Validation Method': None}, 
                        'Output': {'result': None, '* Simulation Data': O['S'], '* Benchmark Data': O['B'], '* Validation Method': None}}
        self.ABM = model(p)
        pass
    
    def run_model(self, p:dict):
        '''Run ABM model with given parameters (p)'''
        self.ABM.sim_reset()
        self.ABM.set_parameters(p)
        return self.ABM.run(display=False)
    
    def validate(self, level:str, method):
        '''Do validation with the method user defined'''
        result = method(self)
        self.Results[level]['result'] = result
        self.Results[level]['* Validation Method'] = method.__doc__ if method.__doc__ else "not mentioned"
        pass

    @property
    def details(self):
        '''Return the details of validation results'''
        detail = 'Validation Results:\n'
        for level, items in self.Results.items():
            for item, info in items.items():
                if item == 'result':
                    detail += f"\n{level} level: {info}\n"
                else:
                    detail += f"{item}: {info}\n"
        return detail