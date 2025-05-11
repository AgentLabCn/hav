import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    parameters = {    
        'road_length': 50000,  
        'speed_limit': 120 * 1000 / 60,  # m/min
        'num_vehicles': 50,  
        'slow_down_prob': 0.15,  
        'acceleration': 2.5 * 1000 / 60,  # m/min^2
        'deceleration': 3.5 * 1000 / 60,  # m/min^2
        'vehicle_lengths': [4, 5, 6, 12],  
        'vehicle_length_probs': [0.7, 0.15, 0.1, 0.05], 
        'steps': 100, 
        'observer_id': 10
    }

    model = TrafficModel(parameters)
    results = model.run()

    data = results.variables['TrafficModel'][['Max_Speed', 'Min_Speed', 'Observer_Speed']]
    plt.plot(data)
    plt.show()
