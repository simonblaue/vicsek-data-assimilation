import numpy as np
import kalman
import vicsek

DATASET_PATH = "../saves/dataset/ds_model.npy"

parameters = {
        'name': 'Dataset',
        'seeds': None,
        'steps': 1001,
        'timestepsize': 1,
        'n_particles': 361,
        'n_ensembles': 100,
        'observation_noise': 0.001,
        'noisestrength':0.5,
        'velocity': 0.003,
        'sampling_rate': 1,
        'alignment_radius': 1.,
        'observable_axis': (True,True,False,True),
        'x_axis': 50,
        'y_axis': 50,
        'find_velocities': True,
        }

def run_filters(cfg=parameters):
    
    model_states = np.load(DATASET_PATH)
    
    vicsek_model = vicsek.ViszecSimulation(cfg)
    
    kalman.EnsembleKalman(cfg, )
    
    
    