
from dataclasses import dataclass
import numpy as np
from vicsek import ViszecSimulation
from animation import VicsekAnimation
from kalman import EnsembleKalman

@dataclass
class BaseSimulationConfig:

    exec_ref = ViszecSimulation

    # particles
    n_particles: int = 100
    # repulsion_radius: float = 0.5
    alignment_radius: float = 1.0

    # field
    x_axis = 25
    y_axis = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 2.0
    xi = 0

    endtime: float = 200
    timestepsize: float = 1.0


@dataclass
class RandomSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 10
    y_axis = 10
    xi = 0.8
    # simulation
    noisestrength: float = 0.5


@dataclass
class GroupingSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 25
    y_axis = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 0.1
    
    
@dataclass
class OrderedSimulationConfig(BaseSimulationConfig):

    # field
    x_axis = 7
    y_axis = 7

    # simulation
    noisestrength: float = 0.5
    
    

@dataclass
class VicsekAnimationConfig:

    exec_ref = VicsekAnimation

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



@dataclass
class EnsembleKalmanConfig:
    
    exec_ref = EnsembleKalman
    
    n_ensembles: int = 100
    
    noise_ratio: float = 0.00000001
    
    n_ensembles: int = 50
    n_particles: int = 100
    x_axis: int = 25
    y_axis: int = 25
    
    state: np.ndarray = np.random.rand(n_particles, 3)
    
    model_forecast: callable = None
    epsilon: np.ndarray = np.ones((n_particles, n_particles))*1e-11