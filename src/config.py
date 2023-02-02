from dataclasses import dataclass, field
import numpy as np



# polygon angle
THETA = 2*np.pi/360*150
# POLYGONSIZE/size of polygons
POLYGONSIZE = 0.2


@dataclass
class SimulationConfig:
    # particles
    n_particles: int = 10
    repulsion_radius: float = 0.5
    alignment_radius: float = 3.0

    # field
    x_axis = 10
    y_axis = 10

    # simulation
    velocity: float = 1.0

    runtimesteps: int = 100
    timestepsize: float = 0.1

    examplearray: list[int] = field(default_factory=list)
