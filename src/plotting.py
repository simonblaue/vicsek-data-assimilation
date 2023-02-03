from typing import List, Tuple

import numpy as np
from config import FREQUENCY, POLYGONSIZE, RESOLUTION, THETA
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon
from model import OrderedSimulationConfig, RandomSimulationConfig


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

class Animation():
    def __init__(self):        
        # initializing the Simulation
        self.simulation = RandomSimulationConfig.exec_ref(RandomSimulationConfig)

        self.fig, (self.axis_simulation, self.axis_tracking) = plt.subplots(1, 2,)
        self.set_axis(self.axis_simulation, 'Model')
        self.set_axis(self.axis_tracking, 'Tracking')

        # initializing all polygons and add them to axes
        # TODO: create function for this and use function for kalmann polygons as well
        self.colors = n_colors(self.simulation.config.n_particles)
        self.polygon_coors = [
            triangle(w[0],w[1], w[2]) for w in self.simulation.walkers
        ]
        self.polygons = [Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(self.polygon_coors, self.colors)]
        for p in self.polygons:
            self.axis_simulation.add_patch(p)


    def set_axis(self, ax: Axes, title: str):
        boundary = 0.5
        ax.set_xlim(-boundary, self.simulation.config.x_axis+boundary)
        ax.set_ylim(-boundary, self.simulation.config.x_axis+boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function_triangle(self):
        return self.polygons

    def animate_triangle(self, i: int):

        # run simulation for <FREQUENCY steps>
        for j in range(FREQUENCY):
            self.simulation.step()

        # update plot
        # TODO: new function to call for kalmann polygons
        for w, p in zip(self.simulation.walkers, self.polygons):
            t = triangle(w[0], w[1], w[2])
            p.set_xy(t)

        return self.polygons


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_triangle, 
            init_func=self.init_function_triangle,
            frames=np.arange(1, 10, 0.05), 
            interval=20, 
            blit=True
        )
        plt.show()


if __name__ == "__main__":
    anim = Animation()
    anim()

