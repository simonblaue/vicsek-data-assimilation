import time
import numpy as np
from typing import Dict, Tuple, List
import json
import os
from tqdm.auto import tqdm

from vicsek import ViszecSimulation
from kalman import EnsembleKalman

"""
This is a script to run generate experiment data without visualization
The simulation parameters and results will be saved under a given directory
"""

def simulate(parameters: Dict) -> Tuple[List, List, Dict]:    
    viscecmodel = ViszecSimulation(parameters)
    filtermodel = EnsembleKalman(parameters, viscecmodel._step, viscecmodel.agents)
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

def execute_experiment(parameters):
    
    
    parameters['theta_observerd'] = parameters['observable_axis'][-1] 
    
    t0 = time.time()
    for seed in parameters['seeds']:
        experimentname = parameters['name']
        # print(f'Running experiment {experimentname} with seed {seed}')

        # experiment_path = f'/saves/{experimentname}/'
        experiment_path = f'saves/{experimentname}/'
        if not os.path.exists(experiment_path):
            # print("Created new Directory!")
            os.makedirs(experiment_path)
            
        if os.path.exists(experiment_path+f'{seed}_model.npy'):
            continue
        
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
    



    
