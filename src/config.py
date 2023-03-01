from dataclasses import dataclass
import numpy as np
from typing import Tuple, List
from vicsek import ViszecSimulation
from animation import Animation
from kalman import EnsembleKalman

"""
This script contains all config dataclasses
"""


@dataclass
class SharedConfig:
    
    # particles number
    n_particles: int = 50
    
    # field
    x_axis = 10
    y_axis = 10


@dataclass
class BaseSimulationConfig(SharedConfig):

    exec_ref = ViszecSimulation

    # particles
    n_particles: int = 50
    # repulsion_radius: float = 0.5
    alignment_radius: float = 1.0

    # field
    x_axis: float = 25
    y_axis: float = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 2.0
    xi: float = 0

    endtime: float = 200
    timestepsize: float = 1.0


@dataclass
class RandomSimulationConfig(BaseSimulationConfig):
    # field
    x_axis: float = 10
    y_axis: float = 10
    xi: float = 0.8
    # simulation
    noisestrength: float = 0.5


@dataclass
class GroupingSimulationConfig(BaseSimulationConfig):

    # field
    x_axis: float = 25
    y_axis: float = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 0.1
    
    
@dataclass
class OrderedSimulationConfig(BaseSimulationConfig):

    # field
    x_axis: float = 7
    y_axis: float = 7

    # simulation
    noisestrength: float = 0.5
    
@dataclass 
class AnimationConfig:

    exec_ref = Animation
    
    viscecmodel: ViszecSimulation = None
    filtermodel: EnsembleKalman = None

    # simulation steps before plotting
    simulation_frequency: int = 1
    
    # simulation steps before sampling
    sample_frequency: int = 1

    # delay between frames in ms
    plot_interval: int = 10

    # frames per simulation
    frames: int = 100

    # boundary around plots
    boundary: float = 0.5
    
    # steps per metrics update
    steps_per_metrics_update: int = 10
    
    experimentname: str = 'None'
    
    save_name: str = 'None'
    
# @dataclass
# class VicsekAnimationConfig(AnimationConfig):
#     exec_ref = VicsekAnimation



@dataclass
class EnsembleKalmanConfig(SharedConfig):
    
    exec_ref = EnsembleKalman
    
    seed: int = 1
    
    n_ensembles: int = 100
    
    noise_ratio: float = 0.0001
    # noise_ratio: float = 0.05
    
    n_particles: int = 50
    
    x_axis: int = 25
    y_axis: int = 25
    
    state: np.ndarray = np.random.rand(n_particles, 3)
    
    # At the moment only the last to False is possible and can only have False beginning from the left to the right
    observable_axis: Tuple[bool, bool, bool] = (True,True,False)
    
    model_forecast: callable = None
    epsilon: np.ndarray = np.ones((SharedConfig.n_particles, SharedConfig.n_particles))*1e-11
    
    ### Need to be overwritten on call when overwritten in another config
    n_particles = SharedConfig.n_particles
    x_axis = SharedConfig.x_axis
    y_axis = SharedConfig.y_axis
    state: np.ndarray = np.random.rand(SharedConfig.n_particles, 3)

