from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from config import POLYGONSIZE, THETA
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon
from vicsek import OrderedSimulationConfig, RandomSimulationConfig


def get_cmap(n: int, cmapname:str = 'hsv') -> Colormap:
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmapname, n)

def n_colors(n: int, cmapname:str = 'hsv') -> List:
    cmap = get_cmap(n, cmapname)
    return [cmap(i) for i in range(n)]


def triangle(x: float, y: float, phi: float) -> np.ndarray:
    ax = (x+POLYGONSIZE*np.cos(phi))
    ay = (y+POLYGONSIZE*np.sin(phi))

    bx = (x+POLYGONSIZE*np.cos(phi+THETA))
    by = (y+POLYGONSIZE*np.sin(phi+THETA))

    cx = (x+POLYGONSIZE*np.cos(phi-THETA))
    cy = (y+POLYGONSIZE*np.sin(phi-THETA))

    triangle = np.array([[ax, ay], [bx, by], [cx, cy]])
    return triangle


def polygon(datapoint: Tuple[float, float, float] = (0,0,0)) -> Polygon:
    x, y, phi = datapoint
    polygon_points = triangle(x,y,phi,)
    return Polygon(polygon_points, facecolor = 'green', alpha=0.5)


class VicsekAnimation():
    def __init__(self, config):

        self.config = config

        # initializing the Simulation
        self.simulation = RandomSimulationConfig.exec_ref(RandomSimulationConfig)

        self.fig, (self.axis_simulation, self.axis_tracking) = plt.subplots(1, 2,)
        self.set_axis(self.axis_simulation, 'Model')
        self.set_axis(self.axis_tracking, 'Tracking')

        self.init_vicsek()
        # TODO: init kalman


    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config.boundary, self.simulation.config.x_axis+self.config.boundary)
        ax.set_ylim(-self.config.boundary, self.simulation.config.x_axis+self.config.boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function_triangle(self):
        return self.vicsek_polygons

    def animate_triangle(self, i: int):

        # run simulation for <FREQUENCY steps>
        for _ in range(self.config.simulation_frequency):
            self.simulation.step()

        self.update_vicsek()

        # TODO: update kalmann

        return self.vicsek_polygons

    def init_vicsek(self):
        self.vicsec_colors = n_colors(self.simulation.config.n_particles)
        vicsek_polygon_coors = [
            triangle(w[0],w[1], w[2]) for w in self.simulation.walkers
        ]
        self.vicsek_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(vicsek_polygon_coors, self.vicsec_colors)
        ]
        for p in self.vicsek_polygons:
            self.axis_simulation.add_patch(p)

    def update_vicsek(self):
        for w, p in zip(self.simulation.walkers, self.vicsek_polygons):
            t = triangle(w[0], w[1], w[2])
            p.set_xy(t)

    def init_kalmann(self):
        # colors green red
        pass

    def update_kalmann(self):
        # change color of filter if filter is assigned to other particle
        pass


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_triangle, 
            init_func=self.init_function_triangle,
            # frames=np.arange(1, 10, 0.05), 
            frames=self.config.frames, 
            interval=self.config.plot_interval, 
            blit=True
        )
        plt.show()


@dataclass
class VicsekAnimationConfig:

    exec_ref = VicsekAnimation

    # simulation steps before plotting
    simulation_frequency: int = 2
    
    # simulation steps before sampling
    sample_frequency: int = 1

    # delay between frames in ms
    plot_interval = 20

    # frames per simulation
    frames = 100

    # boundary around plots
    boundary = 0.5
    




if __name__ == "__main__":
    anim = VicsekAnimationConfig.exec_ref(VicsekAnimationConfig)
    anim()

