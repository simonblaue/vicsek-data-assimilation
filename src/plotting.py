import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from config import THETA, POLYGONSIZE

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
        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(-2,2)
        self.ax.grid()
        
        # here we initialize the starting positions and colors of our polygons
        self.steps = [0,1]
        self.colors = ['darkred', 'darkgreen']
        self.coordinates = [
            triangle(0,np.sin(s),np.arcsin(np.sin(s+np.pi/2))) for s in self.steps
        ]
        # now we initialize the polygons and add them to our axes
        self.polygons = [Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(self.coordinates, self.colors)]
        for p in self.polygons:
            self.ax.add_patch(p)

    # initialization function: plot the background of each frame
    def init_function_triangle(self):
        return self.polygons

    def animate_triangle(self, i):

        # iterate over polygons and update their positions
        for t, p, s in zip(self.coordinates, self.polygons, self.steps):
            s += i 
            y = np.sin(s)
            phi = np.arcsin(np.sin(s+np.pi/2))
            t = triangle(0, y, phi)
            p.set_xy(t)

        return self.polygons


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_triangle, 
            init_func=self.init_function_triangle,
            frames=np.arange(1, 10, 0.05), 
            interval=50, 
            blit=True
        )
        plt.show()

anim = Animation()
anim()

