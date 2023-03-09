import numpy as np
import time
import scipy
from misc import assign_fn, mean_over_ensemble, foldback_dist_states, foldback_dist_ensemble, periodic_distant_vectors

"""
This file contains the Kalman filterclass
"""


def create_H(particle_number, oax):
    z_dim = particle_number * sum(oax)
    x_dim = particle_number * len(oax)
    oax_ints =  [int(b) for b in oax]
    H = np.zeros((z_dim, x_dim))
    i = 0
    j = 0
    while i < x_dim:
        for value in oax_ints:
            # print(i)
            if value == 1:
                H[j,i] = value
                j += 1
            i+=1
            # print(H)
    return H

class EnsembleKalman():
    def __init__(self, config, forecast_func, init_state):
        """
        Receives parameters for the Kalman filter (e.g. Ensemble size) via the config
        Receives model_forecast which is the Vicsek step 
        
        """
        self.config = config
        
        self.number_particles = config['n_particles']
        
        self.number_ensembles = config['n_ensembles']

        self.model_forecast = forecast_func
        
        
        self.agents = np.random.rand(self.number_particles, 4)
        
        self.agents[:,0] *= self.config["x_axis"]
        self.agents[:,1] *= self.config["y_axis"]
        self.agents[:,2] = self.config['velocity']
        self.agents[:,3] *= 2*np.pi
        
        self.H = create_H(config["n_particles"], config["observable_axis"] )
        
        self.number_obs = sum(config["observable_axis"])
        self.number_dims = len(config["observable_axis"])
        

        self._idxs = np.arange(0, config['n_particles'], dtype=np.uint8)
        
        
    def suffle_and_reassign(self, measurement_pos, agents_pos):
        shuffled_idxs = self._idxs.copy()
        np.random.shuffle(shuffled_idxs)
        
        measurement_shuffled = measurement_pos[shuffled_idxs].squeeze()
        assign_idxs = assign_fn(measurement_shuffled, agents_pos, boundary=self.config["x_axis"])
        # new measurement, assignment

        return shuffled_idxs[assign_idxs]
    
    
    def create_forecast_ensemble(self, agents):
        
        state = agents.copy()
        
        
        ensemble = np.tile(state, (self.number_ensembles, 1, 1)) 
        ensemble[:,:,0:2] += np.random.normal(size=(self.number_ensembles, self.number_particles, 2), scale=0.15)
        
        for _ in range(self.config["sampling_rate"]):
            for j in range(self.number_ensembles):
                ensemble[j] = self.model_forecast(ensemble[j]) 
                
        return ensemble
    
        
    
    # TODO for parameter estimation 4 hardcoded = bad 
    def create_virtual_observations_ensemble(self, measured_state):
        virtual_observations = (
                np.tile(measured_state, (self.number_ensembles, 1, 1)) + 
                np.random.normal(size=(self.number_ensembles,self.number_particles, 4), scale=self.config["observation_noise"])
            )
        # print(virtual_observations)
        return virtual_observations
    
    
    def create_ensemble_covariance(self, forecast_ensemble):
        
        # forecast_ensemble = ensemble.reshape(self.number_ensembles,self.number_particles, self.number_dims)
        
        
        mean_forecast = mean_over_ensemble(forecast_ensemble, self.config['x_axis'], self.config['y_axis'])
           
        # Errors within the ensemble = distance between ensemble members and the mean ensemble 
        X_tilde = foldback_dist_ensemble(forecast_ensemble, np.tile(mean_forecast, (self.number_ensembles, 1, 1)), self.config['x_axis'], self.config['y_axis'])
                     
        errors =  np.array([np.outer(e.flatten(), e.flatten()) for e in X_tilde])
        
        # c = 1.5
        # new_forecast_ensemble =  np.tile(mean_forecast, (self.number_ensembles, 1, 1)) + c * X_tilde    
        
        # new_forecast_ensemble[:,:,0] = np.mod(new_forecast_ensemble[:,:,0], self.config['x_axis'])
        # new_forecast_ensemble[:,:,1] = np.mod(new_forecast_ensemble[:,:,1], self.config['y_axis'])
        
        # new_forecast_ensemble[:,:,3] = np.mod(new_forecast_ensemble[:,:,1], self.config['y_axis'])
        
        # # if theata_axis:
        # new_forecast_ensemble[:,:,3] = np.mod(np.tile(mean_forecast, (self.number_ensembles, 1, 1))[:,:,3] + c * X_tilde[:,:,3] + np.pi, 2*np.pi) - np.pi
        
        
        
        pf = 1/(self.number_ensembles-1) * np.sum(errors,axis = 0)

        return pf, forecast_ensemble
        

    def update(self, _measurement: np.ndarray, ) -> np.ndarray:
        
            predicted_idxs = np.arange(self.config['n_particles'])
            if self.config['shuffle_measurements']:
                predicted_idxs = self.suffle_and_reassign(_measurement[:,0:2], self.agents[:,0:2])
            measurement = _measurement[predicted_idxs]

                
            #Generating forecast ensamples
            forecast_ensemble = self.create_forecast_ensemble(self.agents)
            
            # Ensemble Covariance
            PF, forecast_ensemble = self.create_ensemble_covariance(forecast_ensemble)

            # Virtual observation = Measurement + Noise 
            virtual_observations = self.create_virtual_observations_ensemble(measurement)
            virtual_observations = virtual_observations[:,:,self.config["observable_axis"]]
            
            # Virtual observation covariance
            R = np.eye(self.number_particles*self.number_obs, self.number_particles*self.number_obs) * self.config["observation_noise"]

            # Kalman Gain is calculated using the pseudo inverse 
            K = PF @ self.H.T @ scipy.linalg.pinv(self.H @ PF @ self.H.T + R)
            
            
            ensemble_update = np.array([ 
                x + (K @ foldback_dist_states(z, \
                    (self.H @ x.flatten()).reshape(self.number_particles, self.number_obs),\
                        self.config["x_axis"], \
                        self.config["y_axis"], \
                        theat_axis=self.config['theta_observerd']).flatten()).reshape(self.number_particles, self.number_dims)
                for x, z in zip(forecast_ensemble, virtual_observations)
            ])
            
            
            # Updated state is mean over the updated ensemble members 
            agents = mean_over_ensemble(np.array(ensemble_update),self.config["x_axis"], self.config["y_axis"] )
            
            agents[:,0] = np.mod(agents[:,0], self.config["x_axis"])
            agents[:,1] = np.mod(agents[:,1], self.config["y_axis"])
            

            return agents, predicted_idxs


