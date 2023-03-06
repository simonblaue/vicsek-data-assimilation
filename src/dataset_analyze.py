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
        'shuffle_measurements' : True,
        }

def run_filters(cfg=parameters):
    
    modelstates = np.load(DATASET_PATH)
    vicsekmodel = vicsek.ViszecSimulation(cfg)    
    filtermodel = kalman.EnsembleKalman(cfg, vicsekmodel._step)
    
    n_steps = modelstates.shape[0]

    filterstates = []
    assignments = []
    
    for t in range(100):

        filtermodel.agents, predicted_idxs = filtermodel.update(modelstates)
        filterstates.append(filtermodel.agents)
        assignments.append(predicted_idxs)
    
    np.save(DATASET_PATH.replace('model', 'filter'))
    
    
if __name__ == "__main__":
    run_filters()