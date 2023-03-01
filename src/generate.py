import time
from datetime import datetime
import numpy as np
from pathlib import Path
import json
import os

from vicsek import ViszecSimulation
from kalman import EnsembleKalman
from typing import Dict, Tuple, List

"""
This is a script to run generate experiment data without visualization
The simulation parameters and results will be saved under a given directory
"""

def simulate(parameters: Dict) -> Tuple[List, List, Dict]:    
    viscecmodel = ViszecSimulation(parameters)
    filtermodel = EnsembleKalman(parameters, viscecmodel.agents, viscecmodel._step)
    viscecstates = []
    filterstates = []
   
    for t in range(parameters['steps']):
        viscecmodel.update()
        viscecstates.append(viscecmodel.agents)
        if t % parameters['sampling_rate'] == 0:
            filtermodel.update(viscecmodel.agents)
            filterstates.append(filtermodel.agents)

    
    return viscecstates, filterstates

def execute_experiment():
    parameters = {
        'name': 'Baseline',
        'seeds': [1, 2],
        'steps': 100,
        'timestepsize': 1,
        'n_particles': 100,
        'n_ensembles': 50,
        'observation_noise': 0.0001,
        'viscec_noise': 0.5,
        'xi' : 0.8,
        'noisestrength':0.5,
        'velocity': 0.03,
        'sampling_rate': 2,
        'alignment_radius': 1.,
        'observable_axis': (True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        'total_runtime': 0.0
    }
    t0 = time.time()
    for seed in parameters['seeds']:
        experimentname = parameters['name']
        print(f'Running experiment {experimentname} with seed {seed}')

        experiment_path = f'saves/{experimentname}/'
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        
        np.random.seed(int(seed))

        t = time.time()
        
        viscecstates, filterstates = simulate(parameters)
        
        runtime = time.time()-t
        print(f'Runtime: \t {runtime}, \t Saving to {experiment_path}')
        
        np.save(experiment_path+f'{seed}_model.npy', viscecstates)
        np.save(experiment_path+f'{seed}_filter.npy', filterstates)

    parameters['total_runtime'] = time.time() - t0
        
    # saving parameters
    with open(f'{experiment_path}params.json', 'w') as fp:
        json.dump(parameters, fp, indent=4)
    

if __name__ =="__main__":
    execute_experiment()