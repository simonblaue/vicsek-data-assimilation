from dataclasses import dataclass

import numpy as np


class ViszecSimulation:

    def __init__(self, config):
        # Init Config
        self.config = config
        # Array with posx, posy, orientation as rad for n walkers
        self.walkers = np.random.rand(self.config.n_particles, 3)
        self.walkers[:,0] *= self.config.x_axis
        self.walkers[:,1] *= self.config.y_axis
        self.walkers[:,2] *= 2*np.pi
        self.time = 0
        

    def distances(self) -> np.ndarray:
        """
        Calculates the  distance vector from every walker to every other walker. 

        Returns:
            numpy Array(n,n): distances from every walker to every walker
        """
        distances = np.zeros((self.config.n_particles, self.config.n_particles, 2)) 
        walker_pos = self.walkers[:,0:2]

        for i,walker in enumerate(walker_pos):
 
            distances[i,:,:] = walker_pos - walker

        # Enforce boundaries
        distances[:,:,0] = np.where(distances[:,:,0]>self.config.x_axis/2,distances[:,:,0]-self.config.x_axis,distances[:,:,0])
        distances[:,:,0] = np.where(distances[:,:,0]<-self.config.x_axis/2,distances[:,:,0]+self.config.x_axis,distances[:,:,0])
        
        distances[:,:,1] = np.where(distances[:,:,1]>self.config.y_axis/2,distances[:,:,1]-self.config.y_axis,distances[:,:,1])
        distances[:,:,1] = np.where(distances[:,:,1]<-self.config.y_axis/2,distances[:,:,1]+self.config.y_axis,distances[:,:,1])
        return distances
        
        
    def av_directions(self) -> np.ndarray:
         # get which are neighbors 
        dists = self.distances()
        d =  np.linalg.norm(dists, axis=2)
        
        aligner = d < self.config.alignment_radius
        
        # calculate mean angles
        all_phi = self.walkers[:,2]
        av_phi_per_walker = np.zeros(self.config.n_particles)
        for i in range(self.config.n_particles):
            av_phi_per_walker[i] = np.mean(all_phi[aligner[i]])
            
        return av_phi_per_walker
    
    
    def update(self) -> np.ndarray:
        """
        Does one timestep in the visceck model
        """
        self.walkers = self._step(self.walkers)
        return self.walkers
        
        
    def _step(self, state: np.ndarray) -> np.ndarray:
        walkers = state.copy()
        av_phi_per_walker = self.av_directions()
        
        # noise for new angle
        noise = np.random.normal(0,self.config.noisestrength, self.config.n_particles)
        
        # set the new direction 
        walkers[:,2] = self.config.xi*(walkers[:,2])+(1-self.config.xi)*av_phi_per_walker + noise 
        
        # Calculate and set new positions
        new_directions = np.array([np.cos(self.walkers[:,2]), np.sin(self.walkers[:,2])]).transpose()
        walkers[:,0:2] +=  self.config.velocity * self.config.timestepsize * new_directions 

        # Apply boundaries
        walkers[:,0] = np.mod(self.walkers[:,0], self.config.x_axis)
        walkers[:,1] = np.mod(self.walkers[:,1], self.config.y_axis)
    
        return walkers
    

@dataclass
class BaseSimulationConfig:

    exec_ref = ViszecSimulation

    # particles
    n_particles: int = 300
    # repulsion_radius: float = 0.5
    alignment_radius: float = 1.0

    # field
    x_axis = 25
    y_axis = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 2.0
    xi = 0

    endtime: float = 200
    timestepsize: float = 1.0


# Das sind base parameter für random movement:

@dataclass
class RandomSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 10
    y_axis = 10
    xi = 0.5
    # simulation
    noisestrength: float = 0.5

######

# Das sind parameter für orderd movement:

@dataclass
class GroupingSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 25
    y_axis = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 0.1
    
    
@dataclass
class OrderedSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 7
    y_axis = 7

    # simulation
    noisestrength: float = 2.0
    
    
    

