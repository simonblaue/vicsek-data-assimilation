import numpy as np
from vicsek import RandomSimulationConfig, ViszecSimulation
from dataclasses import dataclass

class EnsembleKalman():
    def __init__(self, config, model_forecast):

            self.config = config

            self.state = np.random.rand(100, 3)
            self.state[:,0] *= 10
            self.state[:,1] *= 10
            self.state[:,2] *= 2*np.pi
            # ensemble size
            self.N = self.config.n_ensembles
            # noise parameter
            self.r = self.config.r
            
            self.model_forecast = model_forecast
            self.epsilon = np.ones((100, 100))*1e-11
            
    def update(self, measurement: np.ndarray) -> np.ndarray:
        #generating ensamples
        forecast_ensemble = [
            self.model_forecast(self.state) for _ in range(self.N)
        ]
        measurement_ensemble = np.array([measurement for _ in range(self.N)])
        noise = np.random.normal(size=(100, 3), scale=self.r)
        virtual_observations = measurement_ensemble+noise
        
        # forecast matrix
        mean_forecast = np.mean(forecast_ensemble, axis = 0)
        errors = [f-mean_forecast for f in forecast_ensemble]
        pf = 1/(self.N-1) * np.sum(
            [e @ e.T for e in errors],
            axis = 0
        )
        
        R = np.diag(np.ones(RandomSimulationConfig.n_particles))*self.r
        
        K = pf*np.linalg.inv(pf+R+self.epsilon)
        # update
        ensemble_update = [
            x + K @ (z-x) for x, z in zip(forecast_ensemble, virtual_observations)
        ]
        
        self.state = np.mean(ensemble_update, axis = 0)
        
        return self.state


@dataclass
class EnsembleKalmanConfig:
    
    exec_ref = EnsembleKalman
    
    n_ensembles: int = 5
    
    r: float = 0.01
 