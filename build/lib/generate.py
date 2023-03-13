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

def execute_experiment(
    parameters = {
        'name': 'Baseline',
        'seeds': [np.random.randint(1,1000)],
        'steps': 300,
        'timestepsize': 1,
        'n_particles': 10,
        'n_ensembles': 75,
        'observation_noise': 0.01,
        'alignment_strength':0.15,
        'noisestrength': 0.15,
        'velocity': 0.05,
        'sampling_rate': 1,
        'alignment_radius': 1,
        'observable_axis': (True,True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': False,
        'shuffle_measurements': False,
        'ensemble_theta_noise': 0.,
        'ensemble_pos_noise' : 0.00,
        'save_name' : "With_Artificall_Noise"
        }):
    
    parameters['theta_observerd'] = parameters['observable_axis'][-1] 
    
    t0 = time.time()
    for seed in parameters['seeds']:
        experimentname = parameters['name']
        # print(f'Running experiment {experimentname} with seed {seed}')

        # experiment_path = f'/saves/{experimentname}/'
        experiment_path = f'../saves/{experimentname}/'
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
    

if __name__ =="__main__":
    
    
    parameters_for_given_data = {
        'name': 'GivenData',
        'seeds': [np.random.randint(1,1000)],
        'steps': 200,
        'timestepsize': 1,
        'n_particles': 361,
        'n_ensembles': 2,
        'observation_noise': 0.000,
        'alignment_strength':0.05,
        'noisestrength': 0.15,
        'velocity': 0.09,
        'sampling_rate': 1,
        'alignment_radius': 1,
        'observable_axis': (True,True,True,True),
        'x_axis': 50,
        'y_axis': 50,
        'find_velocities': False,
        'shuffle_measurements': False,
        }
    execute_experiment()