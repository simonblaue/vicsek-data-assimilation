from generate import execute_experiment
from analyze import read_and_eval
from misc import bools2str
import os
from tqdm import tqdm
import numpy as np

parameters = {
        'name': 'Baseline',
        'seeds': [np.random.randint(1,1000)],
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
        'observable_axis': (True,True,True,True),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': False,
        'shuffle_measurements': False
        }
        
        
phase_1_flocking = {
        'name': 'Flocking',
        'noisestrength': 0.15,
        'alignment_strength':0.1,
}

phase_2_random = {
        'name': 'Random',
        'noisestrength':0.2,
        'alignment_strength':0.025,
}

phase_3_jonas = {
        'name': 'Jonas',
        'n_particles':70,
        #'x_axis':50,
        #'y_axis':50,
        'noisestrength':0.1,
        'alignment_strength':0.05,
}


def run_phase(phasedict):
    for p in phasedict:
        parameters[p] = phasedict[p]
    execute_experiment(parameters)




if __name__ == "__main__":
    run_phase(phase_1_flocking)