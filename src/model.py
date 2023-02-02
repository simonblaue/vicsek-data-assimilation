import numpy as np
from config import SimulationConfig

from itertools import permutations



class Simulation:

    def __init__(self):
        # Init Config
        self.config = SimulationConfig()
        # Array with posx, posy, orientation as rad for n walkers
        self.walkers = np.random.rand(self.config.n_particles, 3)
        self.time = 0
        

    def distances(self):
        distances = np.zeros((self.config.n_particles, self.config.n_particles)) 
        walker_pos = self.walkers[:,0:2]

        for i,walker in enumerate(walker_pos):
            distances[i,:] = np.linalg.norm(walker_pos - walker, axis=1)

        return distances
        
    
    def step(self):
        # Calculate all distances to every other walker

        # In 
        
        self.time += self.config.timestepsize
        return 
    
    # TODO
    def plot(self):
        return
    
