import numpy as np
from misc import absolute_distances_with_periodic_box_size

"""
This script contains the Viscec Model
"""

class ViszecSimulation:

    def __init__(self, config):
        """
        Receives a Config file with simulation parameters.
        Sets up the initial coditions (positions and angles of all walkers in the box) 
        
        """

        # Array with posx, posy, velocity_x, velocity_y, orientations  n agents
        self.agents = np.random.rand(config["n_particles"], 4)
        self.agents[:,0:2] *= config["box_size"]
        self.agents[:,2] = config['velocity'] #* (np.random.rand(self.config["n_particles"]))
        self.agents[:,3] *= 2*np.pi
        
        self.n_particles = config["n_particles"]
        self.alignment_radius = config["alignment_radius"]
        self.noisestrength = config["noisestrength"]

        self.box_size = config["box_size"]

        self.dt = config["timestep"]
        self.time = 0
        
        
    def av_directions(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the new angle for each particle, calculated as the average angle over neighbouring particles.
        Returns: 
            numpy Array(n): averaged angle from neighbours for each particle 
        """
        agents = state.copy()

         # get which are neighbors 
        # dists = self.distances(agents[:,0:2])
        # d =  np.linalg.norm(dists, axis=2)
        d = absolute_distances_with_periodic_box_size(agents[:,0:2],agents[:,0:2], self.box_size)
        
        # Has to be commented out because in the complex exponential the particle itself needs to be subtracted.
        #d[d == 0] = np.inf
        
        aligner = d < selfalignment_radius
        
        # calculate mean angles
        all_phi = agents[:,3]
        av_phi_per_walker = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            av_phi_per_walker[i] = np.angle(np.sum(np.exp(1j*(all_phi[aligner[i]]-all_phi[i]))) )
            
        return av_phi_per_walker
        
    
    def update(self) -> np.ndarray:
        """
        Does one timestep in the visceck model
        """
        self.agents = self._step(self.agents)
        self.time += self.dt
        # return self.agents
        
        
        
    def _step(self, state: np.ndarray) -> np.ndarray:
        agents = state.copy()
         
        av_phi_per_walker = self.av_directions(agents)
                
        noise = np.random.normal(0,self.noisestrength,self.n_particles) * 0.5 * np.sqrt(self.dt)
        
        agents[:,3] += self.dt * av_phi_per_walker * self.alignment_strength + noise
        
        agents[:,0] += np.cos(agents[:,3]) * agents[:,2] * self.dt
        agents[:,1] += np.sin(agents[:,3]) * agents[:,2] * self.dt
        
        # Apply boundaries
        agents[:,0:2] = np.mod(agents[:,0:2], self.box_size)

            
        return agents
    

