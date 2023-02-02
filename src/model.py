import numpy as np



class Simulation:

    def __init__(self):
        # Init Config
        self.config = config.SimulationConfig()
        # Array with posx, posy, orientation as rad for n walkers
        self.walkers = np.random.rand(self.config.n_particles, 3)


    # TODO
    def step(self):
        return 
    
    # TODO
    def plot(self):
        return