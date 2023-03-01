# analyze a set of eperiments (given paths) with metrics
import json
import numpy as np 
from misc import metric_lost_particles, metric_hungarian_precision

def read_experiment(experiment_name : str):
    folder = "saves/"+ experiment_name + "/"
    experiment_params  = json.load(open(folder + "params.json"))
    seeds = experiment_params['seeds']
    states = []
    for seed in seeds:
        states.append(
            np.load(f"folder{seed}.model.npy"), 
            np.load(f"folder{seed}.filter.npy")
        )
    return states, experiment_params


def evaluate_experiment(states, experiment_params):
    
    metrics = {
        'Hungarian Precision':[],
        'Lost Particle Precison':[],
    }
    
    measure_freq  = experiment_params['sampling_rate']
    for model_states, filter_states in states:
        model_pos = model_states[::measure_freq, 0:2]
        filter_pos = filter_states[:,0:2]
        
        metrics = 
        
        for step in zip(model_pos,filter_pos):
            metric_lost_particles(model_pos, filter_pos, experiment_params['particles'], experiment_params['alignment_radius']/5)
            metric_hungarian_precision(model_pos, filter_pos, experiment_params['particles'])
        
    