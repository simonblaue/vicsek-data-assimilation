import numpy as np
from config import SimulationConfig


class Simulation:

    def __init__(self):
        # Init Config
        self.config = SimulationConfig()
        # Array with posx, posy, orientation as rad for n walkers
        self.walkers = np.random.rand(self.config.n_particles, 3)
        self.time = 0
        
    # TODO
    def step(self):
        
        self.time += self.config.timestepsize
        return 
    
    # TODO
    def plot(self):
        return
    
