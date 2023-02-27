import numpy as np
from vicsek import RandomSimulationConfig, ViszecSimulation

model_forecast = ViszecSimulation._step


class EnsembleKalman():
    def __init__(self, config):

            self.config = config

            self.state = 
            # ensemble size
            self.N = 5
            # noise parameter
            self.r = 0.01
            
    def update(self, measurement):
        #generating ensamples
        forecast_ensemble = [model_forecast(self.state) for _ in range(self.N)]
        measurement_ensemble = np.array([measurement for _ in range(self.N)])
        noise = np.random.normal(size=(300, 3), scale=self.r)
        virtual_observations = measurement_ensemble+noise
        
        mean_forecast = np.mean(forecast_ensemble, axis = 0)
        errors = [f-mean_forecast for f in forecast_ensemble]
        pf = 1/(self.N-1) * np.sum(
            [e @ e.T for e in errors],
            axis = 0
        )
        
        R = np.diag(np.ones(RandomSimulationConfig.n_particles))*self.r
        
        K = pf*np.linalg.inv(pf+R)
        
        ensemble_update = [
            x + K @ (z-x) for x, z in zip(forecast_ensemble, virtual_observations)
        ]
        
        self.state = np.mean(ensemble_update, axis = 0)
        
        return self.state
