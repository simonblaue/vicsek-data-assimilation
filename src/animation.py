from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from misc import n_colors, xyphi_to_abc
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from vicsek import OrderedSimulationConfig, RandomSimulationConfig


class VicsekAnimation():
    def __init__(self, config):

        self.config = config

        # initializing the Simulation
        self.simulation = RandomSimulationConfig.exec_ref(RandomSimulationConfig)

        self.fig, (self.axis_simulation, self.axis_tracking) = plt.subplots(1, 2, figsize=(10,7))
        self.set_axis(self.axis_simulation, 'Model')
        self.set_axis(self.axis_tracking, 'Tracking')

        self.init_vicsek()
        # TODO: init kalman


    # initialize plot
    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config.boundary, self.simulation.config.x_axis+self.config.boundary)
        ax.set_ylim(-self.config.boundary, self.simulation.config.y_axis+self.config.boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function(self):
        return self.vicsek_polygons


    def init_vicsek(self):
        '''initializes polygons in vicsek plot'''
        self.vicsek_colors = n_colors(self.simulation.config.n_particles)
        vicsek_polygon_coors = [
            xyphi_to_abc(w[0],w[1], w[2]) for w in self.simulation.walkers
        ]
        self.vicsek_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(vicsek_polygon_coors, self.vicsek_colors)
        ]
        for p in self.vicsek_polygons:
            self.axis_simulation.add_patch(p)

    def update_vicsek(self):
        '''updates polygons in vicsek plot'''
        for w, p in zip(self.simulation.walkers, self.vicsek_polygons):
            t = xyphi_to_abc(w[0], w[1], w[2])
            p.set_xy(t)

    def init_kalmann(self):
        '''initializes polygons in kalmann plot'''
        # colors green red
        pass

    def update_kalmann(self):
        '''updates polygons in kalmann plot'''
        # change color of filter if filter is assigned to other particle
        pass

    # TODO:
    def return_metrics(self):
        pass


    def animate_step(self, i: int):
            # run simulation for <FREQUENCY steps>
            for _ in range(self.config.simulation_frequency):
                self.simulation.step()

            self.update_vicsek()

            # TODO: update kalmann

            return self.vicsek_polygons


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_step, 
            init_func=self.init_function,
            # frames=np.arange(1, 10, 0.05), 
            frames=self.config.frames, 
            interval=self.config.plot_interval, 
            # blit=True
        )
        anim.save("saves/test.gif")
        # plt.show()
        self.return_metrics()




@dataclass
class VicsekAnimationConfig:

    exec_ref = VicsekAnimation

    # simulation steps before plotting
    simulation_frequency: int = 2
    
    # simulation steps before sampling
    sample_frequency: int = 1

    # delay between frames in ms
    plot_interval = 40

    # frames per simulation
    frames = 100

    # boundary around plots
    boundary = 0.5
    




if __name__ == "__main__":
    anim = VicsekAnimationConfig.exec_ref(VicsekAnimationConfig)
    anim()

