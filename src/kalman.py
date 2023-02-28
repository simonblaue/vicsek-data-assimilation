import numpy as np
from vicsek import RandomSimulationConfig, ViszecSimulation
from dataclasses import dataclass
import time

class EnsembleKalman():
    def __init__(self, config,):
            self.config = config
            self.state = self.config.state
            self.model_forecast = self.config.model_forecast
            # print(self.config.x_axis)
            
    def update(self, measurement: np.ndarray) -> np.ndarray:
        t = time.time()
        #generating ensamples
        forecast_ensemble = [
            self.model_forecast(self.state) for _ in range(self.config.n_ensembles)
        ]
        # measurement_ensemble = np.tile(measurement, (self.config.n_ensembles, 1, 1))
        # noise = np.random.normal(size=(self.config.n_particles, 3), scale=self.config.noise_ratio)
        # virtual_observations = measurement_ensemble+noise
        # speed advantage??
        virtual_observations = (
            np.tile(measurement, (self.config.n_ensembles, 1, 1)) + 
            np.random.normal(size=(self.config.n_particles, 3), scale=self.config.noise_ratio)
        )
        
        # forecast matrix
        mean_forecast = np.mean(forecast_ensemble, axis = 0)
        
        # errors = np.array([f-mean_forecast for f in forecast_ensemble])
        # speed advantage?
        errors = forecast_ensemble - np.tile(mean_forecast, (self.config.n_ensembles, 1, 1))

        #boundaries 
        errors[:,:,0] = np.where(errors[:,:,0]>self.config.x_axis/2,
                                 errors[:,:,0]-self.config.x_axis,
                                 errors[:,:,0])
        errors[:,:,0] = np.where(errors[:,:,0]<-self.config.x_axis/2,errors[:,:,0]+self.config.x_axis,errors[:,:,0])
        
        errors[:,:,1] = np.where(errors[:,:,1]>self.config.y_axis/2,errors[:,:,1]-self.config.y_axis,errors[:,:,1])
        errors[:,:,1] = np.where(errors[:,:,1]<-self.config.y_axis/2,errors[:,:,1]+self.config.y_axis,errors[:,:,1])
        
        # print(errors)
        # python map
        pf = 1/(self.config.n_ensembles-1) * np.sum(
            [np.matmul(e, e.T) for e in errors],
            axis = 0
        )
        
        R = np.diag(np.ones(self.config.n_particles)) * self.config.noise_ratio
        K = np.matmul(pf, np.linalg.pinv(pf+R))

        # update
        ensemble_update = [
            x + np.matmul(K, (z-x)) for x, z in zip(forecast_ensemble, virtual_observations)
        ]
        
        self.state = np.mean(ensemble_update, axis = 0)
        
        self.state[:,0] = np.mod(self.state[:,0], 10)
        self.state[:,1] = np.mod(self.state[:,1], 10)
        
        # print(f'Update time:\t{time.time()-t}')
        
        return self.state


@dataclass
class EnsembleKalmanConfig:
    
    exec_ref = EnsembleKalman
    
    n_ensembles: int = 20
    
    noise_ratio: float = 0.001
    
    n_ensembles: int = 5
    n_particles: int = 100
    x_axis: int = 25
    y_axis: int = 25
    
    state: np.ndarray = np.random.rand(n_particles, 3)
    
    model_forecast: callable = None
    epsilon: np.ndarray = np.ones((n_particles, n_particles))*1e-11
