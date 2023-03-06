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
        # Array with posx, posy, velocity_x, velocity_y, orientations  n agents
        self.agents = np.random.rand(self.config["n_particles"], 4)
        self.agents[:,0] *= self.config["x_axis"]
        self.agents[:,1] *= self.config["y_axis"]
        self.agents[:,2] = self.config['velocity'] #* (np.random.rand(self.config["n_particles"]))
        self.agents[:,3] *= 2*np.pi

       
        self.time = 0


    def distances(self) -> np.ndarray:
        """
        Calculates the  distance vector from every walker to every other walker. 

        Returns:
            numpy Array(n,n): distances from every walker to every walker
        """
        distances = np.zeros((self.config["n_particles"], self.config["n_particles"], 2)) 
        walker_pos = self.agents[:,0:2]

        for i,walker in enumerate(walker_pos):
 
            distances[i,:,:] = walker_pos - walker

        # Enforce boundaries, use nearest image convention 
        distances[:,:,0] = np.where(distances[:,:,0]>self.config["x_axis"]/2,distances[:,:,0]-self.config["x_axis"],distances[:,:,0])
        distances[:,:,0] = np.where(distances[:,:,0]<-self.config["x_axis"]/2,distances[:,:,0]+self.config["x_axis"],distances[:,:,0])
        
        distances[:,:,1] = np.where(distances[:,:,1]>self.config["y_axis"]/2,distances[:,:,1]-self.config["y_axis"],distances[:,:,1])
        distances[:,:,1] = np.where(distances[:,:,1]<-self.config["y_axis"]/2,distances[:,:,1]+self.config["y_axis"],distances[:,:,1])
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
        
        # Has to be commented out because in the complex exponential the particle itself needs to be subtracted.
        #d[d == 0] = np.inf
        
        aligner = d < self.config["alignment_radius"]
        
        # calculate mean angles
        all_phi = self.agents[:,3]
        av_phi_per_walker = np.zeros(self.config["n_particles"])
        for i in range(self.config["n_particles"]):
            av_phi_per_walker[i] = np.angle(1/self.config["n_particles"]*np.sum(np.exp(1j*(all_phi[aligner[i]]-all_phi[i]))) )
            
        return av_phi_per_walker
        
        
    def av_directions_gauss(self) -> np.ndarray:
        """
        Difference to the function above: Here the average angle is weighted by a Gaussian factor instead of a Step function 
        """
        
        dists = self.distances()
        d =  np.linalg.norm(dists, axis=2)
        weight_factor = np.exp(-d**2/2/self.config["sigma"]**2)
            
        all_phi = self.agents[:,3]
        av_phi_per_walker = np.zeros(self.config["n_particles"])
        
        #for i in range(self.config["n_particles"]):
        av_phi_per_walker = weight_factor @ all_phi 
        
        return av_phi_per_walker     
    
    def update(self) -> np.ndarray:
        """
        Does one timestep in the visceck model
        """
        self.agents = self._step(self.agents)

        # return self.agents
        
        
        
    def _step(self, state: np.ndarray) -> np.ndarray:
        agents = state.copy()
        av_phi_per_walker = self.av_directions()
                
        
        agents[:,3] += self.config["timestepsize"]*av_phi_per_walker*self.config["alignment_strength"] \
            + np.random.normal(0,self.config["noisestrength"],self.config["n_particles"])*0.5*np.sqrt(self.config["timestepsize"])
        
        agents[:,0] += np.cos(agents[:,3])*agents[:,2]
        agents[:,1] += np.sin(agents[:,3])*agents[:,2]
        
        # Apply boundaries
        agents[:,0] = np.mod(agents[:,0], self.config["x_axis"])
        agents[:,1] = np.mod(agents[:,1], self.config["y_axis"])
            
        return agents
    

