import numpy as np
from typing import Dict, Tuple, List
from config import (
    EnsembleKalmanConfig, 
    AnimationConfig, 
    RandomSimulationConfig, 
    BaseSimulationConfig,
)

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
    initfilterstate = np.random.rand(parameters['particles'], 3)
    initfilterstate[:,0] *= parameters['x_axis']
    initfilterstate[:,1] *= parameters['y_axis']
    initfilterstate[:,2] *= 2*np.pi
    filterconfig = EnsembleKalmanConfig(
        n_ensembles=parameters['ensembles'],
        noise_ratio=parameters['observation_noise'],
        n_particles=parameters['particles'],
        x_axis=parameters['x_axis'],
        y_axis=parameters['y_axis'],
        observable_axis=parameters['observable_axis'],
        model_forecast=viscecmodel._step,
        state=initfilterstate
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
    )
    
    return animationconfig.exec_ref(animationconfig)

def simulate(parameters: Dict) -> Tuple[List, List, Dict]:    
    viscecmodel, filtermodel = get_models(parameters=parameters)
    viscecstates = []
    filterstates = []
    metrics = {
        'Hungarian Precision':[],
    }
    
    for t in range(parameters['steps']):
        viscecmodel.update()
        viscecstates.append(viscecmodel.walkers)
        if t % parameters['sampling_rate'] == 0:
            filtermodel.update(viscecmodel.walkers)
            filterstates.append(filtermodel.state)
            metrics['Hungarian Precision'].append(
                
            )
    
    # TODO: end metric
    # avg tracking interval
    # avg tracking steps
    # metrics['tracking_interval'] = tracking_interval
    # metrics['tracking_steps'] = tracking_steps
    
    return viscecstates, filterstates, metrics
