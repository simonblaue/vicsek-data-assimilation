import numpy as np
from config import SimulationConfig
from plotting import Animation

class Simulation:

    def __init__(self):
        # Init Config
        self.config = SimulationConfig()
        # Array with posx, posy, orientation as rad for n walkers
        self.walkers = np.random.rand(self.config.n_particles, 3)
        self.walkers[:,0] *= self.config.x_axis
        self.walkers[:,1] *= self.config.y_axis
        self.walkers[:,2] *= 2*np.pi
        self.time = 0
        


    def abs_distances(self):
        """
        Calculates the absolute distance from every walker to every other walker. 
        Distance to itself is set to Inf for easier use, TODO maybe change this?

        Returns:
            numpy Array(n,n): distances from every walker to every walker
        """
        distances = np.zeros((self.config.n_particles, self.config.n_particles)) 
        walker_pos = self.walkers[:,0:2]

        for i,walker in enumerate(walker_pos):
 
            distances[i,:] = np.linalg.norm(walker_pos - walker, axis=1)
            
            distances[i,i] = np.Inf

        return distances
        
    
    
    def step(self):
        """
        Does one timestep in the visceck model

        Returns:
            walkers after step
        """
        
        # get which are neighbors 
        aligner = self.abs_distances() < self.config.alignment_radius
        
        # calculate mean angles
        all_phi = self.walkers[:,2]
        av_phi_per_walker = np.zeros(self.config.n_particles)
        for i in range(self.config.n_particles):
            if np.all(aligner[i] == False):
                av_phi_per_walker[i] = 0
                continue
            av_phi_per_walker[i] = np.mean(all_phi[aligner[i]])

        # noise for new angle
        noise = np.random.randn(self.config.n_particles)
        self.walkers[:,2] = av_phi_per_walker + noise 
        
        # Calculate and set new positions
        new_directions = np.array([np.cos(self.walkers[:,2]), np.sin(self.walkers[:,2])]).transpose()
        self.walkers[:,0:2] = self.walkers[:,0:2] + self.config.velocity * self.config.timestepsize * new_directions 
    
        self.time += self.config.timestepsize
        return self.walkers
    
    # TODO
    def run(self, write=False):
        tend = self.config.timestepsize * self.config.runtimesteps
        if write:
            # TODO write initial pos to file 
            pass
        while self.time < tend:
            self.step()
            if write:
            # TODO write pos to file 
                pass
            
        print(f"Reached time t: {self.time} in simulation. Run for {self.time/self.config.timestepsize} steps in total.")

