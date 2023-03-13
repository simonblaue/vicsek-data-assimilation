from generate import execute_experiment
from analyze.analyze import read_and_eval
from misc import bools2str
import os
from tqdm import tqdm
import experiments.run_phase_experiments as run_phase_experiments
import numpy as np

parameters = {
        'name': 'Baseline',
        'seeds': [1,2,3,4,5],
        'steps': 300,
        'timestepsize': 1,
        'n_particles': 50,
        'n_ensembles': 100,
        'observation_noise': 0.001,
        'alignment_strength':0.05,
        'noisestrength': 0.15,
        'velocity': 0.05,
        'sampling_rate': 1,
        'alignment_radius': 1,
        'observable_axis': (True,True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': False,
        'shuffle_measurements': True, 
        'ensemble_theta_noise': 0.2,
        'ensemble_pos_noise' : 0.03,
        }


test_ensembles = [10,25,50,75]
test_observation_noise = [0.01,0.05,0.1]
ensemble_theta_noises = [0.08, 0.2 ,0.5] 
ensemble_pos_noises = [0.002, 0.1, 0.3]

# test_ensembles = [10]
# test_observation_noise = [0.01]
# ensemble_theta_noises = [0.08] 
# ensemble_pos_noises = [0.002, 0.1, 0.3]

#### GRID SUCHEN !!!! ###

def grid_search(parameters,phaseparameters, kind):

    # Noisestrength and Alignment strength are set to specific phase
    # for p in phaseparameters:
    #     parameters[p] = phaseparameters[p]


    for ensembles in tqdm(test_ensembles, position=0, leave=False):
        for observation_noise in tqdm(test_observation_noise, position=1, leave=False):
            for ensemble_pos_noise in tqdm(ensemble_pos_noises, position=2, leave=False):
                for ensamble_theta_noise in tqdm(ensemble_theta_noises, position=3, leave=False):
                    name = f"{kind}_50_{ensembles}_{observation_noise}_{ensemble_pos_noise}_{ensamble_theta_noise}"
                    parameters['n_ensembles'] = ensembles
                    parameters['observation_noise'] = observation_noise
                    parameters['ensemble_theta_noise'] = ensamble_theta_noise
                    parameters['ensemble_pos_noise'] = ensemble_pos_noise
                    parameters['name'] = name

                    
                    execute_experiment(parameters)
                    read_and_eval(name)

####### Experiment for Observation Noise ######## 

def observation_noise_exp():
    
    test_values = [0.0001 ,0.001, 0.01, 0.1, 1]
    names = [f"Obsv_noise_{v}" for v in test_values]
    
    for name, test_value in zip(names,test_values):
        parameters['observation_noise'] = test_value
        parameters['name'] = name
        execute_experiment(parameters)
        read_and_eval(name)
    
    
    
def ensemble_size_exp():
    test_values = [10,50,100,150,200]
    names = [f"ensamble_size_{v}" for v in test_values]
    
    for name, test_value in zip(names,test_values):
        parameters['n_ensambles'] = test_value
        parameters['name'] = name
        execute_experiment(parameters)
        read_and_eval(name)
        
if __name__ == "__main__":
    grid_search(parameters,run_phase_experiments.phase_1_flocking, "Flocking")
    grid_search(parameters,run_phase_experiments.phase_2_random, "Random")