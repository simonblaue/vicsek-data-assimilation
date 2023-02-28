import numpy as np
import time
import scipy

class EnsembleKalman():
    def __init__(self, config,):
            self.config = config
            self.state = self.config.state
            self.model_forecast = self.config.model_forecast

            
    # def update(self, measurement: np.ndarray) -> np.ndarray:
    #     t = time.time()
    #     #generating ensamples
    #     forecast_ensemble = [
    #         self.model_forecast(self.state) for _ in range(self.config.n_ensembles)
    #     ]

    #     virtual_observations = (
    #         np.tile(measurement, (self.config.n_ensembles, 1, 1)) + 
    #         np.random.normal(size=(self.config.n_particles, 3), scale=self.config.noise_ratio)
    #     )
        
    #     # forecast matrix
    #     mean_forecast = np.mean(forecast_ensemble, axis = 0)
        
    #     errors = forecast_ensemble - np.tile(mean_forecast, (self.config.n_ensembles, 1, 1))

    #     # #boundaries 
    #     errors[:,:,0] = np.where(errors[:,:,0]>self.config.x_axis/2,errors[:,:,0]-self.config.x_axis,errors[:,:,0])
    #     errors[:,:,0] = np.where(errors[:,:,0]<-self.config.x_axis/2,errors[:,:,0]+self.config.x_axis,errors[:,:,0])
        
    #     errors[:,:,1] = np.where(errors[:,:,1]>self.config.y_axis/2,errors[:,:,1]-self.config.y_axis,errors[:,:,1])
    #     errors[:,:,1] = np.where(errors[:,:,1]<-self.config.y_axis/2,errors[:,:,1]+self.config.y_axis,errors[:,:,1])
        
    #     # Forecast ensemble covariance
    #     pf = 1/(self.config.n_ensembles-1) * np.sum(
    #         [np.matmul(e, e.T) for e in errors],
    #         axis = 0
    #     )
        
    #     # Virtual observation covariance
    #     R = np.diag(np.ones(self.config.n_particles)) * self.config.noise_ratio

    #     K = np.matmul(pf, scipy.linalg.pinv(pf+R))

    #     # update
    #     ensemble_update = [
    #          x + (K @ (z-x)) for x, z in zip(forecast_ensemble, virtual_observations)
    #     ]
        
    #     self.state = np.mean(ensemble_update, axis = 0)
        
    #     self.state[:,0] = np.mod(self.state[:,0], self.config.x_axis)
    #     self.state[:,1] = np.mod(self.state[:,1], self.config.y_axis)
        
    #     # print(f'Update time:\t{time.time()-t}')
        
    #     return self.state



    def update(self, measurement: np.ndarray, H: np.ndarray) -> np.ndarray:
            t = time.time()
            #generating ensamples
            forecast_ensemble = [
                self.model_forecast(self.state) for _ in range(self.config.n_ensembles)
            ]

            virtual_observations = (
                np.tile(measurement, (self.config.n_ensembles, 1, 1)) + 
                np.random.normal(size=(self.config.n_particles, 3), scale=self.config.noise_ratio)
            )
            
            # forecast matrix
            mean_forecast = np.mean(forecast_ensemble, axis = 0)
            
            errors = forecast_ensemble - np.tile(mean_forecast, (self.config.n_ensembles, 1, 1))

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

            K = pf @ H.T @ scipy.linalg.pinv(H @ pf @ H + R )

            # update
            ensemble_update = [
                x + (K @ (z- H @ x)) for x, z in zip(forecast_ensemble, virtual_observations)
            ]
            
            self.state = np.mean(ensemble_update, axis = 0)
            
            self.state[:,0] = np.mod(self.state[:,0], self.config.x_axis)
            self.state[:,1] = np.mod(self.state[:,1], self.config.y_axis)
            
            # print(f'Update time:\t{time.time()-t}')
            
            return self.state


