import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from config import THETA, POLYGONSIZE, FREQUENCY, RESOLUTION
from model import Simulation

def get_cmap(n, cmapname='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmapname, n)

def n_colors(n, cmapname='hsv'):
    cmap = get_cmap(n, cmapname)
    return [cmap(i) for i in range(n)]


def triangle(x,y, phi):
    ax = (x+POLYGONSIZE*np.cos(phi))
    ay = (y+POLYGONSIZE*np.sin(phi))

    bx = (x+POLYGONSIZE*np.cos(phi+THETA))
    by = (y+POLYGONSIZE*np.sin(phi+THETA))

    cx = (x+POLYGONSIZE*np.cos(phi-THETA))
    cy = (y+POLYGONSIZE*np.sin(phi-THETA))

    triangle = np.array([[ax, ay], [bx, by], [cx, cy]])
    return triangle


def polygon(datapoint = (0,0,0)):
    x, y, phi = datapoint
    polygon_points = triangle(x,y,phi,)
    return Polygon(polygon_points, facecolor = 'green', alpha=0.5)

class Animation():
    def __init__(self):        
        # here we initialize the starting positions and colors of our polygons
        self.simulation = Simulation()
        
        print(self.simulation.walkers.shape)

        # First set up the figure, the axis, and the plot element we want to animate
        self.fig, (self.axis_simulation, self.axis_tracking) = plt.subplots(1, 2,)
        self.set_axis(self.axis_simulation, 'Model')
        self.set_axis(self.axis_tracking, 'Tracking')

        self.colors = n_colors(self.simulation.config.n_particles)
        self.polygon_coors = [
            triangle(w[0],w[1], w[2]) for w in self.simulation.walkers
        ]
        # now we initialize the polygons and add them to our axes
        self.polygons = [Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(self.polygon_coors, self.colors)]
        for p in self.polygons:
            self.axis_simulation.add_patch(p)


    def set_axis(self, ax, title):
        boundary = 0.5
        ax.set_xlim(-boundary, self.simulation.config.x_axis+boundary)
        ax.set_ylim(-boundary, self.simulation.config.x_axis+boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function_triangle(self):
        return self.polygons

    def animate_triangle(self, i):


        for i in range(FREQUENCY):
            self.simulation.step()
        # iterate over polygons and update their positions
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

