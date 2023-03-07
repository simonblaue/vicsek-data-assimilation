# visualize an experient given a path
import numpy as np
from pathlib import Path
import json
from animation import Animation

"""
This script is called to animate the simulation with a given set of parameters.
It can also be used to animate simulation data when it is given a filepath.
The animations can be saved.
"""

def visualize_experiment(experiment_name):

    experiment = f'../saves/{experiment_name}/'
    parameters = json.load(open(f'{experiment}params.json'))
    parameters['save_name'] = 'None'
    seed_number = 0
    seed = int(parameters['seeds'][seed_number])
    parameters['experimentid'] = f'{experiment}{seed}'
    parameters['loadexperiment'] = f'experiment'
    parameters['frames'] = parameters['steps']

    parameters['plot_interval'] = 20
    parameters['boundary'] = 0.5
    parameters['lpp_thres'] = parameters['alignment_radius']/2
    
    animation = Animation(parameters)
    animation()
    
def visualize_dataset(dataset):
    parameters = {'Name':'Dataset'}
    parameters['experimentid'] = f'../saves/dataset/ds'
    parameters['loadexperiment'] = f'experiment'
    parameters['frames'] = 100

    parameters['plot_interval'] = 20
    parameters['boundary'] = 0.5
    parameters['lpp_thres'] = 0.5
    
    parameters['x_axis'] = 50
    parameters['y_axis'] = 50

    parameters['n_particles'] = 361
    
    parameters['save_name'] = "None"
    parameters['sampling_rate'] = 1
    

    animation = Animation(parameters)
    animation()

if __name__ =="__main__":
    #visualize_experiment("Flocking")
    visualize_experiment("Random")
    #visualize_dataset('dataset')
    
    
    