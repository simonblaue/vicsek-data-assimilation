import numpy as np

x = [1,2,3,4]

class EnsembleKalman():
    def __init__(self, config):

            self.config = config

            self.H = np.array(
                [
                  1, 1
                ]
            )
            
            self.ensemble_size = 
            self.state = 
            self.model_prediction = 
            
    def state_ensemble(self):
        return np.array(
            [
                self.model_prediction(self.state) for n in range(self.ensemble_size)
            ]
        )
        
    def virtual_observations(self):
        pass
    
    def posteriori_estimate(self):
        pass
        ensemble_mean = 
        errors = 
        pf = 
        return pf

    def gain(self, Pf, H, R):
        return
    
    def update(self):
        pass
        update_ensemble = 