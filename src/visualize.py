# visualize an experient given a path
import numpy as np
from pathlib import Path
import json
from scrips import get_animation

def visualize_experiment():
    parameters = {
        'loadexperiment': 'None',
        'save_name': 'None',
        'seed': 1, # TODO: set seed in random functions!!
        'steps': 100,
        'timestepsize': 1,
        'particles': 100,
        'ensembles': 50,
        'observation_noise': 0.001,
        'viscec_noise': 0.8,
        'velocities': [0.03],
        'sampling_rate': 2,
        'alignment_radius': 1.,
        'observable_axis': (True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        'simulation_frequency': 1,
        'sample_frequency': 1,
        'plot_interval': 10,
        'frames': 100,
        'boundary': 0.5,
        'steps_per_metrics_update': 10,
    }
    
    # TODO: set seeds
    
    animation = get_animation(parameters)
    animation()
    
if __name__ =="__main__":
    visualize_experiment()