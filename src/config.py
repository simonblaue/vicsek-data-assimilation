from dataclasses import dataclass
import numpy as np
from vicsek import ViszecSimulation
from animation import VicsekAnimation
from kalman import EnsembleKalman


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

    # repulsion_radius: float = 0.5
    alignment_radius: float = 1.0

    # simulation
    velocity: float = 0.03
    noisestrength: float = 2.0
    xi = 0

    endtime: float = 200
    timestepsize: float = 1.0


@dataclass
class RandomSimulationConfig(BaseSimulationConfig):
    
    # field
    xi = 0.8
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
    
    # steps per metrics update
    steps_per_metrics_update: int = 10



@dataclass
class EnsembleKalmanConfig(SharedConfig):
    
    exec_ref = EnsembleKalman
    
    noise_ratio: float = 0.0001
    # noise_ratio: float = 0.05
    n_ensembles: int = 100
    
    # At the moment only the last to False is possible and can only have False beginning from the left to the right
    observable_axis = [True,True,False]
    
    model_forecast: callable = None
    epsilon: np.ndarray = np.ones((SharedConfig.n_particles, SharedConfig.n_particles))*1e-11
    
    ### Need to be overwritten on call when overwritten in another config
    n_particles = SharedConfig.n_particles
    x_axis = SharedConfig.x_axis
    y_axis = SharedConfig.y_axis
    state: np.ndarray = np.random.rand(SharedConfig.n_particles, 3)
    
    
    
if __name__ =="__main__":
    anim = VicsekAnimationConfig.exec_ref(
        animation_config=VicsekAnimationConfig,
        simulation_config=RandomSimulationConfig,
        kalman_config=EnsembleKalmanConfig
    )
    anim(save_name=False)


############# DO NOT USE THIS WAY ! #####


# @dataclass
# class GroupingSimulationConfig(BaseSimulationConfig):

#     # field
#     x_axis = 25
#     y_axis = 25

#     # simulation
#     velocity: float = 0.03
#     noisestrength: float = 0.1
    
    
# @dataclass
# class OrderedSimulationConfig(BaseSimulationConfig):

#     # field
#     x_axis = 7
#     y_axis = 7

#     # simulation
#     noisestrength: float = 0.5