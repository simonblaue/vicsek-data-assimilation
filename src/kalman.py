import numpy as np
import time
import scipy

"""
This file contains the Kalman filterclass
"""

class EnsembleKalman():
    def __init__(self, config):
        """
        Receives parameters for the Kalman filter (e.g. Ensemble size) via the config
        Receives model_forecast which is the Vicsek step 
        
        """
            self.config = config
            self.agents = self.config.agents
            # print(self.agents.shape)
            self.model_forecast = self.config.model_forecast

    def update(self, measurement: np.ndarray, ) -> np.ndarray:
            t = time.time()
            
            #generating forecast ensamples
            forecast_ensemble = np.array([
                self.model_forecast(self.agents) for _ in range(self.config.n_ensembles)
            ])

            # Virtual observation = Measurement + Noise 
            virtual_observations = (
                np.tile(measurement, (self.config.n_ensembles, 1, 1)) + 
                np.random.normal(size=(self.config.n_particles, 3), scale=self.config.noise_ratio)
            )[:,:,self.config.observable_axis]
            

            # Mean forecast over ensembles 
            mean_forecast = np.mean(forecast_ensemble[:,:,self.config.observable_axis], axis = 0)
            
            # Errors within the ensemble = distance between ensemble members and the mean ensemble 
            errors = forecast_ensemble[:,:,self.config.observable_axis] - np.tile(mean_forecast, (self.config.n_ensembles, 1, 1))

            # #boundaries
            errors[:,:,0] = np.where(errors[:,:,0]>self.config.x_axis/2,errors[:,:,0]-self.config.x_axis,errors[:,:,0])
            errors[:,:,0] = np.where(errors[:,:,0]<-self.config.x_axis/2,errors[:,:,0]+self.config.x_axis,errors[:,:,0])
            
            errors[:,:,1] = np.where(errors[:,:,1]>self.config.y_axis/2,errors[:,:,1]-self.config.y_axis,errors[:,:,1])
            errors[:,:,1] = np.where(errors[:,:,1]<-self.config.y_axis/2,errors[:,:,1]+self.config.y_axis,errors[:,:,1])
            
            # Forecast ensemble covariance
            pf = 1/(self.config.n_ensembles-1) * np.sum(
                [np.matmul(e, e.T) for e in errors],
                axis = 0
            )
            
            # Virtual observation covariance
            R = np.diag(np.ones(self.config.n_particles)) * self.config.noise_ratio

            # Kalman Gain is calculated using the pseudo inverse 
            K = np.matmul(pf, scipy.linalg.pinv(pf+R))

            # Update the forecasts
            ensemble_update = [
                x + np.hstack(((K @ (z-x[:,self.config.observable_axis])),np.zeros((self.config.n_particles, np.size(self.config.observable_axis)-np.count_nonzero(self.config.observable_axis))))) for x, z in zip(forecast_ensemble, virtual_observations)
            ]
            
            
            # Updated state is mean over the updated ensemble members 
            self.state = np.mean(ensemble_update, axis = 0)
            
            self.agents[:,0] = np.mod(self.agents[:,0], self.config.x_axis)
            self.agents[:,1] = np.mod(self.agents[:,1], self.config.y_axis)
            
            # print(f'Update time:\t{time.time()-t}')
        
            return self.agents


