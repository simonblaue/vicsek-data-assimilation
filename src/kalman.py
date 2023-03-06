import numpy as np
import time
import scipy
from misc import assign_fn

"""
This file contains the Kalman filterclass
"""

class EnsembleKalman():
    def __init__(self, config, forecast_func):
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
        

    def update(self, _measurement: np.ndarray, ) -> np.ndarray:
        
            predicted_idxs = np.arange(self.config['n_particles'])
            if self.config['shuffle_measurements']:
                predicted_idxs = self.suffle_and_reassign(_measurement[:,0:2], self.agents[:,0:2])
            measurement = _measurement[predicted_idxs]

                
            #generating forecast ensamples
            forecast_ensemble = np.array([
                self.model_forecast(self.agents) for _ in range(self.config["n_ensembles"])
            ])

            # Virtual observation = Measurement + Noise 
            virtual_observations = (
                np.tile(measurement, (self.config["n_ensembles"], 1, 1)) + 
                np.random.normal(size=(self.config["n_ensembles"],self.config["n_particles"], 4), scale=self.config["observation_noise"])
            )[:,:,self.config["observable_axis"]]
            
            # Set velocity onto this one value if we dont want to approximate it
            if not self.config['find_velocities']:
                virtual_observations[:,:,2] = forecast_ensemble[:,:,2]
                
            # print(np.var(virtual_observations, axis=0)[0][0])

            # Mean forecast over ensembles 
            mean_forecast = np.mean(forecast_ensemble[:,:,self.config["observable_axis"]], axis = 0)
            
            # Errors within the ensemble = distance between ensemble members and the mean ensemble 
            errors = forecast_ensemble[:,:,self.config["observable_axis"]] - np.tile(mean_forecast, (self.config["n_ensembles"], 1, 1))

            # #boundaries
            errors[:,:,0] = np.where(errors[:,:,0]>self.config["x_axis"]/2,errors[:,:,0]-self.config["x_axis"],errors[:,:,0])
            errors[:,:,0] = np.where(errors[:,:,0]<-self.config["x_axis"]/2,errors[:,:,0]+self.config["x_axis"],errors[:,:,0])
            
            errors[:,:,1] = np.where(errors[:,:,1]>self.config["y_axis"]/2,errors[:,:,1]-self.config["y_axis"],errors[:,:,1])
            errors[:,:,1] = np.where(errors[:,:,1]<-self.config["y_axis"]/2,errors[:,:,1]+self.config["y_axis"],errors[:,:,1])
            
            # Forecast ensemble covariance
            pf = 1/(self.config["n_ensembles"]-1) * np.sum(
                [np.matmul(e, e.T) for e in errors],
                axis = 0
            )
            
            # Virtual observation covariance
            R = np.diag(np.ones(self.config["n_particles"])) * self.config["observation_noise"]

            # Kalman Gain is calculated using the pseudo inverse 
            K = np.matmul(pf, scipy.linalg.pinv(pf+R))

            # Update the forecasts
            # xobs =  np.zeros(
            #     (
            #         self.config["n_particles"],
            #         np.size(self.config["observable_axis"])-np.count_nonzero(self.config["observable_axis"])
            #     )
            # )
            
            where_list = [i for i,m in enumerate(self.config["observable_axis"]) if not m]
            
            ensemble_update = [
                x + np.insert(
                    K @ (z-x[:,self.config["observable_axis"]]), 
                    where_list,
                    np.zeros((self.config["n_particles"],1)),
                    axis = 1) 
                for x, z in zip(forecast_ensemble, virtual_observations)
            ]
            
            
            
            # Updated state is mean over the updated ensemble members 
            self.agents = np.mean(ensemble_update, axis = 0)
            
            self.agents[:,0] = np.mod(self.agents[:,0], self.config["x_axis"])
            self.agents[:,1] = np.mod(self.agents[:,1], self.config["y_axis"])
            
            # print(f'Update time:\t{time.time()-t}')

            return self.agents, predicted_idxs


