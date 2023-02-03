from dataclasses import dataclass, field
import numpy as np



# polygon angle
THETA = 2*np.pi/360*150
# POLYGONSIZE/size of polygons
POLYGONSIZE = 0.2


@dataclass
class SimulationConfig:
    # particles
    n_particles: int = 300
    # repulsion_radius: float = 0.5
    alignment_radius: float = 1

    # field
    x_axis = 25
    y_axis = 25

    # simulation
    velocity: float = 0.03
    noisestrength: float = 0.1

    endtime: float = 100
    timestepsize: float = 1.0



###

# Das sind base parameter für random movement:

# @dataclass
# class SimulationConfig:
#     # particles
#     n_particles: int = 300
#     # repulsion_radius: float = 0.5
#     alignment_radius: float = 1

#     # field
#     x_axis = 7
#     y_axis = 7

#     # simulation
#     velocity: float = 0.03
#     noisestrength: float = 2.0

#     endtime: float = 100
#     timestepsize: float = 1.0


######

# Das sind parameter für orderd movement:

# @dataclass
# class SimulationConfig:
#     # particles
#     n_particles: int = 300
#     # repulsion_radius: float = 0.5
#     alignment_radius: float = 1

#     # field
#     x_axis = 25
#     y_axis = 25

#     # simulation
#     velocity: float = 0.03
#     noisestrength: float = 0.1

#     endtime: float = 100
#     timestepsize: float = 1.0