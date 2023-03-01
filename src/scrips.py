import numpy as np
from typing import Dict, Tuple, List
from config import (
    EnsembleKalmanConfig, 
    AnimationConfig, 
    RandomSimulationConfig, 
    BaseSimulationConfig,
)

from misc import metric_hungarian_precision, metric_lost_particles

def get_models(parameters: Dict):
    viscecmodelconfig = RandomSimulationConfig(
        n_particles=parameters['particles'],
        alignment_radius=parameters['alignment_radius'],
        noisestrength=parameters['viscec_noise'],
        endtime=parameters['steps'],
        timestepsize=parameters['timestepsize'],
        x_axis=parameters['x_axis'],
        y_axis=parameters['y_axis'],
    )
    viscecmodel = viscecmodelconfig.exec_ref(viscecmodelconfig)
    initfilteragents = np.random.rand(parameters['particles'], 3)
    initfilteragents[:,0] *= parameters['x_axis']
    initfilteragents[:,1] *= parameters['y_axis']
    initfilteragents[:,2] *= 2*np.pi
    filterconfig = EnsembleKalmanConfig(
        n_ensembles=parameters['ensembles'],
        noise_ratio=parameters['observation_noise'],
        n_particles=parameters['particles'],
        x_axis=parameters['x_axis'],
        y_axis=parameters['y_axis'],
        observable_axis=parameters['observable_axis'],
        model_forecast=viscecmodel._step,
        agents=initfilteragents
    )
    
    velocities = parameters['velocities']
    if len(velocities) == 1:
        viscecmodelconfig.velocity=velocities[0],
        
    
    filtermodel = filterconfig.exec_ref(filterconfig)
    
    return viscecmodel, filtermodel

def get_animation(parameters: Dict):
    viscecmodel, filtermodel = get_models(parameters)
    animationconfig = AnimationConfig(
        viscecmodel=viscecmodel,
        filtermodel=filtermodel,
        simulation_frequency=parameters['simulation_frequency'],
        sample_frequency=parameters['sample_frequency'],
        plot_interval=parameters['plot_interval'],
        frames=parameters['frames'],
        boundary=parameters['boundary'],
        steps_per_metrics_update=parameters['steps_per_metrics_update'],
        experimentname=parameters['loadexperiment'],
        save_name=parameters['save_name'],
        seed=parameters['seed'],
        x_axis=parameters['x_axis'],
        y_axis=parameters['y_axis'],
        n_particles=parameters['particles'],
    )
    
    return animationconfig.exec_ref(animationconfig)

def simulate(parameters: Dict) -> Tuple[List, List, Dict]:    
    viscecmodel, filtermodel = get_models(parameters=parameters)
    viscecstates = []
    filterstates = []
    # metrics = {
    #     'Hungarian Precision':[],
    #     'Lost Particle Precison':[],
    # }
    
    for t in range(parameters['steps']):
        viscecmodel.update()
        viscecstates.append(viscecmodel.agents)
        if t % parameters['sampling_rate'] == 0:
            filtermodel.update(viscecmodel.agents)
            filterstates.append(filtermodel.agents)
            # metrics['Hungarian Precision'].append(
            #     metric_hungarian_precision(viscecmodel.agents[:,0:2] ,filtermodel.state[:,0:2], )
            # )
            # metrics['Lost Particle Precison'].append(
            #     metric_lost_particles(viscecmodel.agents[:,0:2] ,filtermodel.state[:,0:2])
            # )
    
    # TODO: end metric
    # avg tracking interval
    # avg tracking steps
    # metrics['tracking_interval'] = tracking_interval
    # metrics['tracking_steps'] = tracking_steps
    
    return viscecstates, filterstates
