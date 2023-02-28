from typing import List, Tuple
import matplotlib.gridspec as gridspec
import numpy as np
from misc import n_colors, xyphi_to_abc, format_e, metric_hungarian_precision
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from decimal import Decimal


class Animation():
    def __init__(
        self,
        animation_config,
    ):
        self.config = animation_config
        
        # TODO: improve none cases
        if self.config.experimentname != 'None':
            self.animate_step = self._step_visualize           
        
        else:
            viscecmodel = self.config.viscecmodel
            filtermodel = self.config.filtermodel
            self.viscecmodel = viscecmodel
            self.filtermodel = filtermodel
            self.animate_step = self._step_test
            self.metrics = {'Hungarian Precision': []}
            
        
        self.init_figure()
        self.set_axis(self.axes[0], 'Model')
        self.set_axis(self.axes[1], 'Filter')

        self.init_vicsek_plot()
        self.init_filter_plot()
        self.init_metrics_plot()

    # TODO:
    def loadexperiment(self,):
        pass
        # self.config.experimentname
        viscecdata, filterdata = 1, 1
        return viscecdata, filterdata

    def init_figure(self):
        self.fig = plt.figure(figsize=(10,7),)
        self.gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.25,hspace=0.6, height_ratios=[2,1])
        self.axes = [0]*3
        self.axes[0]=self.fig.add_subplot(self.gs[0,0])
        self.axes[1]=self.fig.add_subplot(self.gs[0,1])
        self.axes[2]=self.fig.add_subplot(self.gs[1,:])
        
    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config.boundary, self.viscecmodel.config.x_axis+self.config.boundary)
        ax.set_ylim(-self.config.boundary, self.viscecmodel.config.y_axis+self.config.boundary)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)

    # initialization function: plot the background of each frame
    def init_function(self):
        return self.vicsek_polygons, self.filter_polygons, self.hungarian_precision_line

    def init_vicsek_plot(self):
        self.vicsek_colors = n_colors(self.viscecmodel.config.n_particles)
        vicsek_polygon_coors = [
            xyphi_to_abc(w[0],w[1], w[2]) for w in self.viscecmodel.walkers
        ]
        self.vicsek_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(vicsek_polygon_coors, self.vicsek_colors)
        ]
        for p in self.vicsek_polygons:
            self.axes[0].add_patch(p)

    def init_filter_plot(self):
        self.filter_colors = n_colors(self.viscecmodel.config.n_particles)
        filter_polygon_coors = [
            xyphi_to_abc(w[0],w[1], w[2]) for w in self.filtermodel.state
        ]
        self.filter_polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(filter_polygon_coors, self.filter_colors)
        ]
        for p in self.filter_polygons:
            self.axes[1].add_patch(p)
            
    def init_metrics_plot(self):
        self.step = 1
        self.error_mean = []
        self.error_hungarian = []
        self.axes[2].set_xlabel('Steps')
        self.axes[2].set_ylabel('Precision')
        self.axes[2].set_title('Metrics')
        self.axes[2].grid()
        self.hungarian_precision_line, = self.axes[2].plot([0], [0], lw=1, c='blue', label=f'Hungarian Precision')
        self.axes[2].legend()
        

    def update_vicsek_plot(self):
        for w, p in zip(self.viscecmodel.walkers, self.vicsek_polygons):
            t = xyphi_to_abc(w[0], w[1], w[2])
            p.set_xy(t)

    def update_filter_plot(self):
        for w, p in zip(self.filtermodel.state, self.filter_polygons):
            t = xyphi_to_abc(w[0], w[1], w[2])
            p.set_xy(t)
    
    def update_metrics(self):
        precision = metric_hungarian_precision(
            self.viscecmodel.walkers,
            self.filtermodel.state,
            self.viscecmodel.config.n_particles
        )
        
        self.metrics['Hungarian Precision'].append(precision)
        self.step += self.config.steps_per_metrics_update
        
        self.hungarian_precision_line.set_data(
            np.arange(0, self.step-1, self.config.steps_per_metrics_update),
            self.metrics['Hungarian Precision'],
        )
        self.axes[2].set_xlim(0, self.step+1)
        self.axes[2].set_ylim(0, np.max(self.metrics['Hungarian Precision']))
        p = format_e(np.mean(self.metrics['Hungarian Precision']))
        self.axes[2].set_xlabel(
            f'Steps,  Mean Hungarian Precision: {p}'
        )

    def _step_test(self, i: int):
            # run simulation for <FREQUENCY steps>
            for _ in range(self.config.simulation_frequency):
                self.viscecmodel.update()

            self.filtermodel.update(self.viscecmodel.walkers)
            self.update_vicsek_plot()
            self.update_filter_plot()
            
            if i % self.config.steps_per_metrics_update  == 0: 
                self.update_metrics()
            
            return self.vicsek_polygons, self.filter_polygons, self.hungarian_precision_line

    
    def _step_visualize(self, i: int):
        pass


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_step, 
            init_func=self.init_function,
            # frames=np.arange(1, 10, 0.05), 
            frames=self.config.frames, 
            # interval=100, 
            interval=self.config.plot_interval, 
            blit=False
        )
        # TODO: cover none cases
        if self.config.save_name != 'None':
            anim.save(f"saves/{self.config.save_name}.gif")
        plt.show()
        return anim
