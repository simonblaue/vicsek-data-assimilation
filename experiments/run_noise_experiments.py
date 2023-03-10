from src.generate import execute_experiment
from analyze.analyze import read_and_eval

from tqdm.auto import tqdm
import numpy as np


test_observation_noise = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.12, 0.15, 0.2]
ensemble_theta_noises = [0.08,0.1,0.15,0.2,0.23, 0.3, 0.4, 0.5, 0.6, 0.7] 
ensemble_pos_noises = [0.002, 0.004, 0.006, 0.008, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


#### GRID SUCHEN !!!! ###

def grid_search(parameters,phaseparameters, kind):
    
    for p in phaseparameters:
        parameters[p] = phaseparameters[p]
        
    parameters["seeds"] = [0,1,2,3]
    
    for observation_noise in tqdm(test_observation_noise, position=0, leave=False):
        for ensemble_pos_noise in tqdm(ensemble_pos_noises, position=1, leave=False):
            for ensemble_theta_noise in tqdm(ensemble_theta_noises, position=2, leave=False):
                name = f"{kind}_50_50_{observation_noise}_{ensemble_pos_noise}_{ensemble_theta_noise}"
                parameters['observation_noise'] = observation_noise
                parameters['ensemble_theta_noise'] = ensemble_theta_noise
                parameters['ensemble_pos_noise'] = ensemble_pos_noise
                parameters['name'] = name
                
                execute_experiment(parameters)
                read_and_eval(name)

