from dataclasses import dataclass
import numpy as np

"""
This script contains the Viscec Model
"""

class ViszecSimulation:

    def __init__(self, config):
        """
        Receives a Config file with simulation parameters.
        Sets up the initial coditions (positions and angles of all walkers in the box) 
        
        """
        # Init Config
        self.config = config
        # Array with posx, posy, orientation as rad for n walkers (chosen randomly)
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

        # Enforce boundaries, use nearest image convention 
        distances[:,:,0] = np.where(distances[:,:,0]>self.config.x_axis/2,distances[:,:,0]-self.config.x_axis,distances[:,:,0])
        distances[:,:,0] = np.where(distances[:,:,0]<-self.config.x_axis/2,distances[:,:,0]+self.config.x_axis,distances[:,:,0])
        
        distances[:,:,1] = np.where(distances[:,:,1]>self.config.y_axis/2,distances[:,:,1]-self.config.y_axis,distances[:,:,1])
        distances[:,:,1] = np.where(distances[:,:,1]<-self.config.y_axis/2,distances[:,:,1]+self.config.y_axis,distances[:,:,1])
        return distances
        
        
    def av_directions(self) -> np.ndarray:
        """
        Calculates the new angle for each particle, calculated as the average angle over neighbouring particles.
        Returns: 
            numpy Array(n): averaged angle from neighbours for each particle 
        """
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
        # return self.walkers
        
        
    def _step(self, state: np.ndarray) -> np.ndarray:
        walkers = state.copy()
        av_phi_per_walker = self.av_directions()
        
        # noise for new angle
        noise = np.random.normal(0,self.config.noisestrength, self.config.n_particles)
        
        # set the new direction 
        walkers[:,2] = self.config.xi*(walkers[:,2])+(1-self.config.xi)*av_phi_per_walker + noise 
        
        # Calculate and set new positions
        new_directions = np.array([np.cos(walkers[:,2]), np.sin(walkers[:,2])]).transpose()
        walkers[:,0:2] +=  self.config.velocity * self.config.timestepsize * new_directions 
        
        # Apply boundaries
        walkers[:,0] = np.mod(walkers[:,0], self.config.x_axis)
        walkers[:,1] = np.mod(walkers[:,1], self.config.y_axis)
    
        return walkers
    

