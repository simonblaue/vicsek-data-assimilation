import numpy as np
from vicsek import RandomSimulationConfig, ViszecSimulation
from dataclasses import dataclass

class EnsembleKalman():
    def __init__(
        self,
        config,
        model_forecast,
        x_axis,
        y_axis,
        n_particles,
    ):

            self.config = config
            self.n_particles = n_particles

            self.state = np.random.rand(self.n_particles, 3)
            self.state[:,0] *= x_axis
            self.state[:,1] *= y_axis
            self.state[:,2] *= 2*np.pi
            # ensemble size
            self.N = self.config.n_ensembles
            # noise parameter
            self.r = self.config.r
            
            self.model_forecast = model_forecast
            self.epsilon = np.ones((self.n_particles, self.n_particles))*1e-11
            
    def update(self, measurement: np.ndarray) -> np.ndarray:
        #generating ensamples
        forecast_ensemble = [
            self.model_forecast(self.state) for _ in range(self.N)
        ]
        measurement_ensemble = np.array([measurement for _ in range(self.N)])
        noise = np.random.normal(size=(self.n_particles, 3), scale=self.r)
        virtual_observations = measurement_ensemble+noise
        
        # forecast matrix
        mean_forecast = np.mean(forecast_ensemble, axis = 0)
        errors = np.array([f-mean_forecast for f in forecast_ensemble])
        #boundaries 
        errors[:,:,:,0] = np.where(errors[:,:,:,0]>self.config.x_axis/2,errors[:,:,:,0]-self.config.x_axis,errors[:,:,:,0])
        errors[:,:,:,0] = np.where(errors[:,:,:,0]<-self.config.x_axis/2,errors[:,:,:,0]+self.config.x_axis,errors[:,:,:,0])
        
        errors[:,:,:,1] = np.where(errors[:,:,:,1]>self.config.y_axis/2,errors[:,:,:,1]-self.config.y_axis,errors[:,:,:,1])
        errors[:,:,:,1] = np.where(errors[:,:,:,1]<-self.config.y_axis/2,errors[:,:,:,1]+self.config.y_axis,errors[:,:,:,1])
        
        
        pf = 1/(self.N-1) * np.sum(
            [e @ e.T for e in errors],
            axis = 0
        )
        
        R = np.diag(np.ones(self.n_particles))*self.r
        
        K = pf*np.linalg.pinv(pf+R+self.epsilon)
        # update
        ensemble_update = [
            x + K @ (z-x) for x, z in zip(forecast_ensemble, virtual_observations)
        ]
        
        self.state = np.mean(ensemble_update, axis = 0)
        
        self.state[:,0] = np.mod(self.state[:,0], 10)
        self.state[:,1] = np.mod(self.state[:,1], 10)
        
        return self.state


@dataclass
class EnsembleKalmanConfig:
    
    exec_ref = EnsembleKalman
    
    n_ensembles: int = 5
    
    r: float = 0.00001
 