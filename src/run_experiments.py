from generate import execute_experiment
from analyze import read_and_eval
from misc import bools2str
import os
from tqdm import tqdm


parameters = {
        'name': 'Baseline',
        'seeds': [1,2,3,4,5,6,7,8,9,10],
        'steps': 200,
        'timestepsize': 1,
        'n_particles': 50,
        'n_ensembles': 100,
        'observation_noise': 0.01,
        'viscec_noise': 0.05,
        'xi' : 0.8,
        'noisestrength':0.5,
        'velocity': 0.003,
        'sampling_rate': 2,
        'alignment_radius': 1.,
        'observable_axis': (True,True,True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': True,
        }


#### GRID SUCHEN !!!! ###

def grid_search():
    test_observable_axis = [(True,True,True,True,True),(True,True,True,True,False),(True,True,False,False,False)]
    test_agents = [50,100]
    test_ensembles = [50,100,150,200,250]
    test_observation_noise = [0.0001 ,0.001, 0.01, 0.1, 1]
    test_sampling_rate = [1,2,4]

    for observable_axis in tqdm(test_observable_axis, position=0, leave=False):
        # for agents in tqdm(test_agents, position=1, leave=False):
        for ensembles in tqdm(test_ensembles, position=2, leave=False):
            for observation_noise in tqdm(test_observation_noise, position=3, leave=False):
                for sampling_rate in tqdm(test_sampling_rate, position=4, leave=False):
                    name = f"{bools2str(observable_axis)}_50_{ensembles}_{observation_noise}_{sampling_rate}"
                    parameters['observable_axis'] = observable_axis
                    # parameters['agents'] = agents
                    parameters['observation_noise'] = observation_noise
                    parameters['sampling_rate']  = sampling_rate
                    parameters['name'] = name
                    parameters['n_ensambles'] = test_ensembles
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
    grid_search()