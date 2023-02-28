from typing import List, Tuple
import matplotlib.gridspec as gridspec
import numpy as np
from misc import n_colors, xyphi_to_abc, format_e
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from decimal import Decimal




class VicsekAnimation():
    def __init__(self, animation_config, simulation_config, kalman_config):

        self.config = animation_config
        # initializing the Simulation
        self.simulation = simulation_config.exec_ref(simulation_config)
        self.filter = kalman_config.exec_ref(
            kalman_config(
                n_particles=self.simulation.config.n_particles,
                state=self.simulation.walkers,
                model_forecast=self.simulation._step,
                x_axis=simulation_config.x_axis,
                y_axis=simulation_config.y_axis,
            )
        )

        self.init_figure()
        self.set_axis(self.axes[0], 'Model')
        self.set_axis(self.axes[1], 'Kalman')

        self.init_vicsek_plot()
        self.init_kalman_plot()
        self.init_metrics_plot()


    def init_figure(self):
        self.fig = plt.figure(figsize=(10,7),)
        self.gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.5,hspace=0.6, height_ratios=[2,1])
        self.axes = [0]*3
        self.axes[0]=self.fig.add_subplot(self.gs[0,0])
        self.axes[1]=self.fig.add_subplot(self.gs[0,1])
        self.axes[2]=self.fig.add_subplot(self.gs[1,:])
        
    # initialize plot
    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config.boundary, self.simulation.config.x_axis+self.config.boundary)
        ax.set_ylim(-self.config.boundary, self.simulation.config.y_axis+self.config.boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)


    # initialization function: plot the background of each frame
    def init_function(self):
        return self.vicsek_polygons, self.kalman_polygons, self.errline_mean, self.errline_max


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
            self.axes[0].add_patch(p)

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
            self.axes[1].add_patch(p)
            
    def init_metrics_plot(self):
        self.step = 1
        self.error_mean = []
        self.error_max = []
        # self.axes[1][0].set_title('Error')
        self.axes[2].set_xlabel('Steps')
        self.axes[2].set_ylabel('Error')
        self.axes[2].grid()
        self.errline_max, = self.axes[2].plot([0], [0], lw=1, c='blue', label=f'Maximum')
        self.errline_mean, = self.axes[2].plot([0], [0], lw=1, c='orange', label=f'Mean')
        self.axes[2].legend()
        
        

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
        diff = np.abs(self.filter.state - self.simulation.walkers)
        self.error_mean.append(np.mean(diff))
        self.error_max.append(np.max(diff))
        self.step += self.config.steps_per_metrics_update
        self.errline_mean.set_data(np.arange(0, self.step-1, self.config.steps_per_metrics_update), self.error_mean)
        self.errline_max.set_data(np.arange(0, self.step-1, self.config.steps_per_metrics_update), self.error_max)
        self.axes[2].set_xlim(0, self.step+1)
        self.axes[2].set_ylim(0, np.max(self.error_max))
        self.axes[2].set_xlabel(
            f'Steps,  Total Max: {format_e(np.max(self.error_max))},  Mean avg: {format_e(np.mean(self.error_mean))}'
        )


    def animate_step(self, i: int):
            # run simulation for <FREQUENCY steps>
            for _ in range(self.config.simulation_frequency):
                self.simulation.update()

            self.filter.update(self.simulation.walkers, self.filter.config.observable_axis)
            self.update_vicsek_plot()
            self.update_kalmann_plot()
            
            if i % self.config.steps_per_metrics_update  == 0: 
                self.update_metrics()
            
            

            return self.vicsek_polygons, self.kalman_polygons, self.errline_mean, self.errline_max


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



if __name__ =="__main__":
    import config 
    anim = config.VicsekAnimationConfig.exec_ref(
        animation_config=config.VicsekAnimationConfig,
        simulation_config=config.RandomSimulationConfig(n_particles=10),
        kalman_config=config.EnsembleKalmanConfig
    )
    anim(save_name=False)