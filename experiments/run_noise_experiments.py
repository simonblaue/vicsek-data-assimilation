from src.generate import execute_experiment
from analyze.analyze import read_and_eval

from tqdm.auto import tqdm
import numpy as np


test_ensembles = [10,25,50,75]
test_observation_noise = [0.01,0.05,0.1]
ensemble_theta_noises = [0.08, 0.2 ,0.5] 
ensemble_pos_noises = [0.002, 0.1, 0.3]


#### GRID SUCHEN !!!! ###

def grid_search(parameters,phaseparameters, kind):
    
    for p in phaseparameters:
        parameters[p] = phaseparameters[p]
        
    parameters["seed"] = [0,1,2,3,4,5]
    
    for ensembles in tqdm(test_ensembles, position=0, leave=False):
        for observation_noise in tqdm(test_observation_noise, position=1, leave=False):
            for ensemble_pos_noise in tqdm(ensemble_pos_noises, position=2, leave=False):
                for ensemble_theta_noise in tqdm(ensemble_theta_noises, position=3, leave=False):
                    name = f"{kind}_50_{ensembles}_{observation_noise}_{ensemble_pos_noise}_{ensemble_theta_noise}"
                    parameters['n_ensembles'] = ensembles
                    parameters['observation_noise'] = observation_noise
                    parameters['ensemble_theta_noise'] = ensemble_theta_noise
                    parameters['ensemble_pos_noise'] = ensemble_pos_noise
                    parameters['name'] = name

                    
                    execute_experiment(parameters)
                    read_and_eval(name)

