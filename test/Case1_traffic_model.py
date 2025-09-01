import agentpy as ap
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.tsa.ar_model import AutoReg

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hav.HAV import HAV

class Vehicle(ap.Agent):
    '''An agent of vehicle'''
    def setup(self):
        self.length = np.random.choice(self.p.vehicle_lengths, p=self.p.vehicle_length_probs)
        self.speed = 0
        self.position = np.random.randint(0, self.p.road_length)

    def update_speed(self, leading_vehicle):
        # Acceleration
        if self.speed < self.p.speed_limit:
            self.speed = min(self.speed + self.p.acceleration, self.p.speed_limit)

        # Prevent collision
        if leading_vehicle:
            gap = leading_vehicle.position - self.position - self.length
            if gap < 0:
                gap += self.p.road_length
            if self.speed >= gap:
                self.speed = max(gap - 1, 0)

        # Deceleration randomly
        if np.random.random() < self.p.slow_down_prob and self.speed > 0:
            self.speed = max(self.speed - self.p.deceleration, 0)

    def move(self):
        self.position = (self.position + self.speed) % self.p.road_length


class TrafficModel(ap.Model):
    '''A basic traffic simulation model'''
    def setup(self):
        self.vehicles = ap.AgentList(self, self.p.num_vehicles, Vehicle)
        self.vehicles = ap.AgentList(self, sorted(self.vehicles, key=lambda x: x.position))

    def step(self):
        self.vehicles = ap.AgentList(self, sorted(self.vehicles, key=lambda x: x.position))
        for i, vehicle in enumerate(self.vehicles):
            leading_vehicle = self.vehicles[(i + 1) % len(self.vehicles)]
            vehicle.update_speed(leading_vehicle)

        for vehicle in self.vehicles:
            vehicle.move()

    def update(self):
        max_speed = np.max([v.speed for v in self.vehicles])
        self.record('Max_Speed', max_speed)

        min_speed = np.min([v.speed for v in self.vehicles])
        self.record('Min_Speed', min_speed)

        observer_speed = [v.speed for v in self.vehicles if v.id == self.p['observer_id']][0]
        self.record('Observer_Speed', observer_speed)

        congestion = len([v for v in self.vehicles if v.speed == 0])
        self.record('Congestion', congestion)

    def end(self):
        for v in self.vehicles:
            v.record('Speed', v.speed)

class validate_traffic_model():
    '''This example encapsulates the provided parameters required in HAV, 
        and demonstrates a process of validating Traffic Model'''
    
    # Paramenters required in Traffic Model
    p = {    
        'road_length': 50000,  
        'speed_limit': 120 * 1000 / 60, 
        'num_vehicles': 50,  
        'slow_down_prob': 0.15,  
        'acceleration': 2.5 * 1000 / 60,  
        'deceleration': 3.5 * 1000 / 60, 
        'vehicle_lengths': [4, 5, 6, 12],  
        'vehicle_length_probs': [0.7, 0.15, 0.1, 0.05], 
        'steps': 100,
        'observer_id': 10
    }

    # Benchmark data used as a control reference in validation
    Benchmark = {
        'Speed_range': (580, 300),
        'SIR_beta_range': (0.2, 1.0),
        'Pearson_corr_range': (0.8, 1.0)
    }

    # Declare the parameter names required in Agent(A), Model(M) and Output(O) level, including Simulated data(S) and Benchmark data(B)
    A = {'S': ['Speed'], 'B': ['Speed_range']}
    M = {'S': ['Congestion'], 'B': ['SIR_beta_range']}
    O = {'S': ['num_vehicles', 'Congestion'], 'B': ['Pearson_corr_range']}

    def __init__(self):
        '''Execute HAV directly to visually demonstrate the process of validation'''
        np.random.seed(123)
        
        # 1 Create a HAV object
        my_hav = HAV(TrafficModel, self.p, self.A, self.M, self.O)

        # 2 Transmit validation methods that required in each level
        my_hav.validate('Agent', self.test_SPEED_dist)
        my_hav.validate('Model', self.test_CONGESTION_trans)
        my_hav.validate('Output', self.test_CORR_FLOW_CONGESTION)

        # 3 Recored the results
        logger.info(my_hav.details)


    # The following are validation methods that defined by user
    # (For different models and users, the following content can be freely edited)

    def test_SPEED_dist(self, hav:HAV):
        '''Validate the rationality of speed distribution using Kolmogorov-Smirnov Test '''
        result = hav.run_model(self.p)
        speed = result.variables.Vehicle['Speed'].T.tolist()
        speed_bench = np.random.normal(self.Benchmark["Speed_range"][0], self.Benchmark["Speed_range"][1], len(speed))
        _, p_value = ks_2samp(speed, speed_bench)
        return p_value > 0.05

    def test_CONGESTION_trans(self, hav:HAV):
        '''Validate the rationality of congestion spread mechanism (Based on SIR Model)'''
        result = hav.run_model(self.p)
        congestion_ts = result.variables.TrafficModel['Congestion'] > 1
        model_sir = AutoReg(congestion_ts, lags=1).fit()
        beta = model_sir.params[1]  
        return self.Benchmark['SIR_beta_range'][0] < beta < self.Benchmark['SIR_beta_range'][1]

    def test_CORR_FLOW_CONGESTION(self, hav:HAV):
        '''Validate the the correlation between traffic flow and congestion rate with Pearson Correlation Analysis'''
        flow_ts, congestion_ts = [], []
        for flow in range(10, 101, 10):
            p_copy = self.p.copy()
            p_copy['num_vehicles'] = flow
            result = hav.run_model(p_copy)
            flow_ts.append(flow)
            congestion_ts.append(np.mean(result.variables.TrafficModel['Congestion']))
        corr = np.corrcoef(flow_ts, congestion_ts)[0, 1]
        return self.Benchmark['Pearson_corr_range'][0] < corr < self.Benchmark['Pearson_corr_range'][1]

if __name__ == '__main__':
    validate_traffic_model()
