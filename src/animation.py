from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from misc import n_colors, xyphi_to_abc
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from vicsek import OrderedSimulationConfig, RandomSimulationConfig
from kalman import EnsembleKalmanConfig


class VicsekAnimation():
    def __init__(self, animation_config, simulation_config = OrderedSimulationConfig):

        self.config = animation_config
        # initializing the Simulation
        self.simulation = simulation_config.exec_ref(simulation_config)
        self.filter = EnsembleKalmanConfig.exec_ref(
            EnsembleKalmanConfig(
                n_particles=self.simulation.config.n_particles,
                state=self.simulation.walkers,
                model_forecast=self.simulation._step,
                x_axis=simulation_config.x_axis,
                y_axis=simulation_config.y_axis,
            )
        )

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10,7))
        self.set_axis(self.axes[0][0], 'Model')
        self.set_axis(self.axes[0][1], 'Tracking')

        self.init_vicsek_plot()
        self.init_kalman_plot()
        self.init_metrics_plot()


    # initialize plot
    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config.boundary, self.simulation.config.x_axis+self.config.boundary)
        ax.set_ylim(-self.config.boundary, self.simulation.config.y_axis+self.config.boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function(self):
        return self.vicsek_polygons, self.kalman_polygons, self.mean_errline


    def init_vicsek_plot(self):
        '''initializes polygons in vicsek plot'''
        self.vicsek_colors = n_colors(self.simulation.config.n_particles)
        vicsek_polygon_coors = [
            xyphi_to_abc(w[0],w[1], w[2]) for w in self.simulation.walkers
        ]
        self.vicsek_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(vicsek_polygon_coors, self.vicsek_colors)
        ]
        for p in self.vicsek_polygons:
            self.axes[0][0].add_patch(p)

    def init_kalman_plot(self):
        '''initializes polygons in vicsek plot'''
        self.kalman_colors = n_colors(self.simulation.config.n_particles)
        kalman_polygon_coors = [
            xyphi_to_abc(w[0],w[1], w[2]) for w in self.filter.state
        ]
        self.kalman_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(kalman_polygon_coors, self.kalman_colors)
        ]
        for p in self.kalman_polygons:
            self.axes[0][1].add_patch(p)
            
    def init_metrics_plot(self):
        self.error = []
        self.time = 1
        self.axes[1][0].grid()
        # self.max_errline, = self.axes[1][0].plot([], [], lw=2)
        self.mean_errline, = self.axes[1][0].plot([0], [0], lw=2)

    def update_vicsek_plot(self):
        '''updates polygons in vicsek plot'''
        for w, p in zip(self.simulation.walkers, self.vicsek_polygons):
            t = xyphi_to_abc(w[0], w[1], w[2])
            p.set_xy(t)


    def update_kalmann_plot(self):
        '''updates polygons in kalmann plot'''
        for w, p in zip(self.filter.state, self.kalman_polygons):
            t = xyphi_to_abc(w[0], w[1], w[2])
            p.set_xy(t)
    

    # TODO:
    def update_metrics(self):
        diff = np.mean(np.abs(self.filter.state - self.simulation.walkers))
        # print(diff)
        self.error.append(diff)
        self.time += 1
        self.mean_errline.set_data(np.arange(0, self.time-1, 1), self.error)



    def animate_step(self, i: int):
            # run simulation for <FREQUENCY steps>
            for _ in range(self.config.simulation_frequency):
                self.simulation.update()

            self.filter.update(self.simulation.walkers)
            self.update_vicsek_plot()
            self.update_kalmann_plot()
            
            self.update_metrics()
            
            

            return self.vicsek_polygons, self.kalman_polygons, self.mean_errline


    def __call__(self, save_name: bool = False):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_step, 
            init_func=self.init_function,
            # frames=np.arange(1, 10, 0.05), 
            frames=self.config.frames, 
            interval=self.config.plot_interval, 
            blit=False
        )
        if save_name:
            anim.save(f"saves/{save_name}.gif")
        plt.show()
        # self.return_metrics()
        return anim




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


if __name__ =="__main__":
    anim = VicsekAnimationConfig.exec_ref(
        animation_config=VicsekAnimationConfig,
        simulation_config=RandomSimulationConfig
    )
    anim(save_name=False)