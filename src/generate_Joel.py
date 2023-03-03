import time
from datetime import datetime
import numpy as np
from pathlib import Path
import json
import os
from alive_progress import alive_bar
from tqdm.auto import tqdm

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
    assignments = []
   
    # with alive_bar(parameters['steps']) as bar:
    for t in tqdm(range(parameters['steps']), position=5, leave=False):
        viscecmodel.update()
        viscecstates.append(viscecmodel.agents)

        if t % parameters['sampling_rate'] == 0:
            filtermodel.agents, predicted_idxs = filtermodel.update(viscecmodel.agents)
            filterstates.append(filtermodel.agents)
            assignments.append(predicted_idxs)

    
    return viscecstates, filterstates, assignments

def execute_experiment(
    parameters = {
        'name': 'Baseline',
        'seeds': [1],
        'steps': 50,
        'timestepsize': 0.2,
        'n_particles': 50,
        'n_ensembles': 2,
        'observation_noise': 0.0001,
        'xi' : 0.05,
        'noisestrength': 0.3,
        'velocity': 0.03,
        'sampling_rate': 1,
        'alignment_radius': 1,
        'observable_axis': (True,True,True,True,True),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': True,
        'shuffle_measurements': False
        }):
    t0 = time.time()
    for seed in parameters['seeds']:
        experimentname = parameters['name']
        # print(f'Running experiment {experimentname} with seed {seed}')

        experiment_path = f'../saves/{experimentname}/'
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
            
        # if os.path.exists(experiment_path+f'{seed}_model.npy'):
        #     continue
        
        np.random.seed(int(seed))
        
        viscecstates, filterstates, assignments = simulate(parameters)
        # print(f'Runtime: \t {runtime}, \t Saving to {experiment_path}')
        
        np.save(experiment_path+f'{seed}_model.npy', viscecstates)
        np.save(experiment_path+f'{seed}_filter.npy', filterstates)
        np.save(experiment_path+f'{seed}_assignments.npy', assignments)

    parameters['total_runtime'] = time.time() - t0
 
    # saving parameters
    with open(f'{experiment_path}params.json', 'w') as fp:
        json.dump(parameters, fp, indent=4)
    

if __name__ =="__main__":
    execute_experiment()