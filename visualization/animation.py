from typing import List, Tuple
import matplotlib.gridspec as gridspec
import numpy as np
from misc import n_colors, xyphi_to_abc, format_e, metric_hungarian_precision, metric_lost_particles, metric_flocking
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

"""
This script contains the animation class for visualization
"""



class Animation():
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.loadexperiment(self.config['experimentid'])
        self.animate_step = self._step_visualize
        
        self.modelagents = self.viscecdata[0]
        self.filteragents = self.filterdata[0]
            
        self.metrics = {
            'Hungarian Precision': [],
            'LPP': [],
            'Flocking' : [],
        }
            
        self.init_figure()
        self.set_axis(self.axes[0], 'Model')
        self.set_axis(self.axes[1], 'Filter')

        self.model_polygons = self.init_polygon_plot(self.modelagents, self.axes[0])

        self.filter_polygons = self.init_polygon_plot(self.filteragents, self.axes[1])

        self.init_metrics_plot()

    def loadexperiment(self, experimentid):
        self.viscecdata = np.load(f'{experimentid}_model.npy')
        try:
            self.filterdata = np.load(f'{experimentid}_filter.npy')
        except FileNotFoundError:
            self.filterdata = None
        try:
            self.assignmentdata = np.load(f'{experimentid}_assignments.npy')
        except FileNotFoundError:
            self.assignmentdata = None

    def init_figure(self):
        self.fig = plt.figure(figsize=(10,7),)
        self.gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.25,hspace=0.6, height_ratios=[2,1])
        self.axes = [0]*3
        self.axes[0]=self.fig.add_subplot(self.gs[0,0])
        self.axes[1]=self.fig.add_subplot(self.gs[0,1])
        self.axes[2]=self.fig.add_subplot(self.gs[1,:])
        
    def set_axis(self, ax: Axes, title: str):
        ax.set_xlim(-self.config['boundary'], self.config['box_size']+self.config['boundary'])
        ax.set_ylim(-self.config['boundary'], self.config['box_size']+self.config['boundary'])
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.25)

    # initialization function: plot the background of each frame
    def init_function(self):
        return self.model_polygons, self.filter_polygons, self.hungarian_precision_line, self.lpp_line

    def init_polygon_plot(self, agents, axis):
        self.agent_colors = n_colors(self.config['n_particles'])
        polygon_coors = [
            xyphi_to_abc(w[0],w[1],w[3]) for w in agents
        ]
        polygons = [
            Polygon(t, closed=True, fc=c, ec=c) for t, c in zip(polygon_coors, self.agent_colors)
        ]
        for p in polygons:
            axis.add_patch(p)
        return polygons
            
    def init_metrics_plot(self):
        self.step = 1
        self.error_mean = []
        self.error_hungarian = []
        self.error_flocking = []
        self.axes[2].set_xlabel('Steps')
        self.axes[2].set_ylabel('')
        self.axes[2].set_title('Metrics')
        self.axes[2].grid()
        self.hungarian_precision_line, = self.axes[2].plot([0], [0], lw=1, c='blue', label=f'Hungarian Precision')
        self.lpp_line, = self.axes[2].plot([0], [0], lw=1, c='orange', label=f'LPP')
        self.flocking_line, = self.axes[2].plot([0], [0], lw=1, c='darkred', label=f'Flocking')
        self.axes[2].legend()
        

    def update_polygons(self, agents, polygons):
        for a, p in zip(agents, polygons):
            t = xyphi_to_abc(a[0], a[1],a[3])
            p.set_xy(t)
    
    def update_metrics(self, step, step_assignment_idxs):
        model_positions = self.modelagents[:,0:2][step_assignment_idxs]
        hungarian_precision = metric_hungarian_precision(
            model_positions,
            self.filteragents[:,0:2],
            boundary=self.config["box_size"],
        )
        lpp = metric_lost_particles(
            model_positions, 
            self.filteragents[:,0:2], 
            self.config['lpp_thres']
        )
        flocking = metric_flocking(self.modelagents[:,3][step_assignment_idxs],self.config["n_particles"])
        
        if step == 0:
            self.metrics['Hungarian Precision'] = [hungarian_precision]
            self.metrics['LPP'] = [lpp]
            self.metrics['Flocking'] = [flocking]
        else:
            self.step = step+1
            self.metrics['Hungarian Precision'].append(hungarian_precision)
            self.metrics['LPP'].append(lpp)
            self.metrics['Flocking'].append(flocking)
        
        self.hungarian_precision_line.set_data(
            np.arange(0, self.step, self.config['sampling_rate']),
            self.metrics['Hungarian Precision'],
        )
        self.lpp_line.set_data(
            np.arange(0, self.step, self.config['sampling_rate']),
            self.metrics['LPP'],
        )
        self.flocking_line.set_data(
            np.arange(0, self.step, self.config['sampling_rate']),
            self.metrics['Flocking'],
        )
        self.axes[2].set_xlim(0, self.step+1)
        ylimit = max(self.metrics['Hungarian Precision']+self.metrics['LPP'])
        self.axes[2].set_ylim(
            0,
            1.05#ylimit+0.05
        )
        _hp = format_e(np.mean(self.metrics['Hungarian Precision']))
        _lpp = format_e(np.mean(self.metrics['LPP']))
        _fl = format_e(np.mean(self.metrics['Flocking']))
        self.axes[2].set_xlabel(
            f'Steps,  Mean HP: {_hp},  '
            f'Mean LPP: {_lpp}'
            f'Flocking: {_fl}'
        )     

    # TODO:
    def _step_visualize(self, i: int):
        self.modelagents = self.viscecdata[i]
        self.update_polygons(self.modelagents, self.model_polygons)
        if i % self.config['sampling_rate'] == 0:
            filter_step = i // self.config['sampling_rate']
            
            if np.any(self.assignmentdata):
                step_assignment_idxs = self.assignmentdata[filter_step]
            else:
                step_assignment_idxs = list(range(self.config['n_particles']))
            # print(step_assignment_idxs)

            self.filteragents = self.filterdata[filter_step]
            
            self.update_polygons(self.filteragents, [self.filter_polygons[i] for i in step_assignment_idxs])
           
            self.update_metrics(i, step_assignment_idxs)
            
        return self.model_polygons, self.filter_polygons, self.hungarian_precision_line, self.lpp_line


    def __call__(self):
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_step, 
            init_func=self.init_function,
            # frames=np.arange(1, 10, 0.05), 
            frames=self.config['frames'], 
            # interval=100, 
            interval=self.config['plot_interval'], 
            blit=False
        )

        # if self.config['save_name']:
        #     filename = f"../vicsek-data-assimilation/saves/animations/{self.config['save_name']}.gif"
        #     print(f'Saving as {filename}')
        #     anim.save(filename)
        plt.show()
        return anim
