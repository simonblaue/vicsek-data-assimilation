import numpy as np
import time
import scipy
from misc import assign_fn, mean_over_ensemble, foldback_dist_states, foldback_dist_ensemble, periodic_distant_vectors

"""
This file contains the Kalman filterclass
"""

class EnsembleKalman():
    def __init__(self, config, forecast_func, init_state):
        """
        Receives parameters for the Kalman filter (e.g. Ensemble size) via the config
        Receives model_forecast which is the Vicsek step 
        
        """
        self.config = config
        self.agents = np.random.rand(self.config["n_particles"], 4)
        self.agents[:,0] *= self.config["x_axis"]
        self.agents[:,1] *= self.config["y_axis"]
        self.agents[:,2] = self.config['velocity']
        self.agents[:,3] *= 2*np.pi

        #self.agents = init_state
        # print(self.agents.shape)
        self.model_forecast = forecast_func
        
        self._idxs = np.arange(0, config['n_particles'], dtype=np.uint8)
        
    def suffle_and_reassign(self, measurement_pos, agents_pos):
        shuffled_idxs = self._idxs.copy()
        np.random.shuffle(shuffled_idxs)
        
        measurement_shuffled = measurement_pos[shuffled_idxs].squeeze()
        assign_idxs = assign_fn(measurement_shuffled, agents_pos, boundary=self.config["x_axis"])
        # new measurement, assignment

        return shuffled_idxs[assign_idxs]
    
    
    def create_forecast_ensemble(self, state):
        ensemble = np.tile(state, (self.config["n_ensembles"], 1, 1))  
        
        for _ in range(self.config["sampling_rate"]):
            for j in range(self.config["n_ensembles"]):
                ensemble[j] = self.model_forecast(ensemble[j])
                
            
        return ensemble
    
    # TODO for parameter estimation 4 hardcoded = bad 
    def create_virtual_observations_ensemble(self, measured_state):
        virtual_observations = (
                np.tile(measured_state, (self.config["n_ensembles"], 1, 1)) + 
                np.random.normal(size=(self.config["n_ensembles"],self.config["n_particles"], 4), scale=self.config["observation_noise"])
            )
        # print(virtual_observations)
        return virtual_observations
        

    def update(self, _measurement: np.ndarray, ) -> np.ndarray:
        
            predicted_idxs = np.arange(self.config['n_particles'])
            if self.config['shuffle_measurements']:
                predicted_idxs = self.suffle_and_reassign(_measurement[:,0:2], self.agents[:,0:2])
            measurement = _measurement[predicted_idxs]

                
            #generating forecast ensamples
            forecast_ensemble = self.create_forecast_ensemble(self.agents)
            
            #step the forecast ensemble

            # Virtual observation = Measurement + Noise 
            virtual_observations = self.create_virtual_observations_ensemble(measurement)
            
            # Set velocity onto this one value if we dont want to approximate it
            if not self.config['find_velocities']:
                virtual_observations[:,:,2] = forecast_ensemble[:,:,2]
                
            virtual_observations = virtual_observations[:,:,self.config["observable_axis"]]
            # print(np.var(virtual_observations, axis=0)[0][0])

            # Mean forecast over ensembles 
            mean_forecast = mean_over_ensemble(forecast_ensemble, self.config['x_axis'], self.config['y_axis'])
            # print(mean_forecast[:,3])
            # Errors within the ensemble = distance between ensemble members and the mean ensemble 
            errors = foldback_dist_ensemble(forecast_ensemble, np.tile(mean_forecast, (self.config["n_ensembles"], 1, 1)), self.config['x_axis'], self.config['y_axis'])
                    
            # Forecast ensemble covariance
            pf = 1/(self.config["n_ensembles"]-1) * np.sum(
                [np.matmul(e, e.T) for e in errors],
                axis = 0
            )
            
            # Virtual observation covariance
            R = np.diag(np.ones(self.config["n_particles"])) * self.config["observation_noise"]

            # Kalman Gain is calculated using the pseudo inverse 
            K = np.matmul(pf, scipy.linalg.pinv(pf+R))
            
            # K = np.zeros((self.config["n_particles"], self.config["n_particles"]))

            # Update the forecasts            
            where_list = [i for i,m in enumerate(self.config["observable_axis"]) if not m]
            
            for i,m in enumerate(where_list):
                if m > virtual_observations.shape[2]:
                    where_list[i] = virtual_observations.shape[2]
            
            ## Foldback 
            ensemble_update = [
                x + np.insert(
                    K @ foldback_dist_states(z,x[:,self.config["observable_axis"]],self.config["x_axis"], self.config["y_axis"], theat_axis=self.config['theta_observerd'] ), 
                    where_list,
                    np.zeros((self.config["n_particles"],1)),
                    axis = 1) 
                for x, z in zip(forecast_ensemble, virtual_observations)
            ]
            
                        
            # Updated state is mean over the updated ensemble members 
            agents = mean_over_ensemble(np.array(ensemble_update),self.config["x_axis"], self.config["y_axis"] )
            
            agents[:,0] = np.mod(agents[:,0], self.config["x_axis"])
            agents[:,1] = np.mod(agents[:,1], self.config["y_axis"])
            
            # print(f'Update time:\t{time.time()-t}')

            return agents, predicted_idxs


