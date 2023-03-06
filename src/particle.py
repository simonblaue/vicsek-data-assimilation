import numpy as np
import time
import scipy

"""
This file contains the Kalman filterclass
"""

class ParticleFilter():
    def __init__(self, config):
        """
        Receives parameters for the Particle filter (e.g. Ensemble size) via the config
        Receives model_forecast which is the Vicsek step 
        
        """
        self.config = config
        self.agents = self.config.agents
        self.weights = self.config.weights # shape: n_ensembles # this does not exist in Kalman filter 
        # print(self.agents.shape)
        self.model_forecast = self.config.model_forecast

    def update(self, measurement: np.ndarray, ) -> np.ndarray:
            t = time.time()
            
            #generating forecast ensamples
            forecast_ensemble = np.array([
                self.model_forecast(self.agents) for _ in range(self.config.n_ensembles)
            ])
            # Shape: (n_ensembles, n_particles, 3) 
             

            # Virtual observation = Measurement (without noise) 
            virtual_observations = (np.tile(measurement, (self.config.n_ensembles, 1, 1)))[:,:,self.config.observable_axis]
            
            errors = forecast_ensemble[:,:,self.config.observable_axis] - virtual_observations
            
            sigma_weights = 0.5
            gaussfactors = []
            for e,w in zip(errors,self.weights):     
                gaussfactors.append( 1/np.sqrt(2*np.pi*sigma_weights**2) * np.exp(-np.mean(e)**2/2/sigma_weights**2) )
            
            self.weights *= np.array(gaussfactors)
            
            self.weights *= 1/np.sum(self.weights)
            
            Neff = 1/np.sum( self.weights**2 )
            
            if Neff < self.config.Ncrit:
                
                # do resampling: Particles will be set to new positions according to their weight  
                #self.agents = ...
                self.weights = 1/self.config.n_particles
          
            # Updated state is mean over the updated ensemble members 
            
            self.agents[:,0] = np.mod(self.agents[:,0], self.config.x_axis)
            self.agents[:,1] = np.mod(self.agents[:,1], self.config.y_axis)
            
            # print(f'Update time:\t{time.time()-t}')
        
            return self.agents
            
            
            
    #def resampling(self):
     #   return ...


