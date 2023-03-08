# analyze a set of eperiments (given paths) with metrics
import json
import numpy as np 
from misc import metric_lost_particles, metric_hungarian_precision
import os
from typing import Dict, List
from pathlib import Path
from misc import bools2str
from parameters import load_parameters
import matplotlib.pyplot as plt

p = '/home/henrik/projects/nonlineardynamics23/Flocking_1100/Flocking_1111_50_50_0.1_True/'

def read_and_eval(experiment_name):
    filter_states, model_states, assignments_states, params = read_experiment(experiment_name)
    evaluate_experiment(filter_states, model_states, assignments_states, params)

def read_experiment(experiment_name : str):
    # folder = "saves/"+ experiment_name + "/"
    folder = experiment_name
    experiment_params  = json.load(open(folder + "params.json"))
    seeds = experiment_params['seeds']
    model_states = []
    filter_states = []
    assignments_states = []
    for seed in seeds:
        model_states.append(np.load(f"{folder}{seed}_model.npy"))
        filter_states.append(np.load(f"{folder}{seed}_filter.npy"))
        assignments_states.append(np.load(f"{folder}{seed}_assignments.npy"))
    return model_states, filter_states, assignments_states, experiment_params


def evaluate_experiment(model_states, filter_states, assignments_states, experiment_params):
    
    experiment_name = experiment_params['name']
    experiment_path = f'saves/{experiment_name}/'
    if os.path.exists(f'{experiment_path}metrics.json'):
        return
    
    metrics = {
        'Hungarian Precision':[],
        'Lost Particle Precision':[],
        'Average Hungarian Precision': [],
        'Average LPP': [],
        'Var Hungarian Precision': [],
        'Var LPP': []
    }
    
    seeds = experiment_params['seeds']
    sampling_rate  = experiment_params['sampling_rate']
    
    measure_steps = int(experiment_params['steps']/sampling_rate)
    
    average_hung = np.zeros(measure_steps)
    average_lpp = np.zeros(measure_steps)
    
    std_hung = np.zeros(measure_steps)
    std_lpp = np.zeros(measure_steps)

    for seed in range(len(seeds)):
        assignments = assignments_states[seed]
        model_pos = model_states[seed][::sampling_rate,:, 0:2]
        
        filter_pos = filter_states[seed][:,:,0:2]
        
        metrics['Hungarian Precision']
        
        hung_metric = []
        lpp_metric = []
        
        for i in range(measure_steps) :
            current_model_pos = model_pos[i]
            current_assignment = assignments[i]
            reassigned_model_pos = current_model_pos[current_assignment]
            m_hung = metric_hungarian_precision(reassigned_model_pos, filter_pos[i], experiment_params['x_axis'])
            m_lpp = metric_lost_particles(reassigned_model_pos, filter_pos[i], experiment_params['alignment_radius']/2)
            hung_metric.append(m_hung)
            lpp_metric.append(m_lpp)
            
            average_hung[i] += m_hung
            average_lpp[i] += m_lpp
            
            std_hung[i] += m_hung**2
            std_lpp[i] += m_lpp**2
            
        metrics['Hungarian Precision'].append(hung_metric)
        metrics['Lost Particle Precision'].append(lpp_metric)
        
    assignmentsT = np.vstack(
        # [np.load(str(file)).T for file in Path(experiment_path).glob('*assignments.npy')]
        a.T for a in assignments_states
    )
    trajs = assignments_to_binary_trajectories(assignmentsT, experiment_params['steps'])
    trajs = [''.join(map(str, t)) for t in trajs]
    max_lengths = [max([len(s) for s in ts.split('0')]) for ts in trajs]
    metrics['max_length_analysis'] = (np.mean(max_lengths), np.std(max_lengths)) 
        
    std_hung -=   (average_hung * average_hung)/len(seeds)
    std_hung /= len(seeds)    
    
    std_lpp -=   (average_lpp * average_lpp)/len(seeds)
    std_lpp /= len(seeds)    
        
    average_hung /= len(seeds)
    average_lpp /= len(seeds)
    
    metrics['Average Hungarian Precision'] = average_hung.tolist()
    metrics['Average LPP'] = average_lpp.tolist()
    metrics['Var Hungarian Precision'] = std_hung.tolist()
    metrics['Var LPP'] = std_lpp.tolist()
    
    
    with open(f'{p}metrics.json', 'w') as fp:
        # json.dump(experiment_params, fp, indent=4)
        json.dump(metrics, fp, indent=4)
        

def name_from_parameters(
    datatype,
    observable_axis,
    n_agents,
    ensembles,
    observation_noise,
    sampling_rate,
):
    return (
        f"{datatype}_{bools2str(observable_axis)}"
        f"_{n_agents}_{ensembles}_{observation_noise}"
        f"_{sampling_rate}"
    )
    
def experiments_to_analyze(datatype: str) -> List[Path]:
    print(f'Datatype: {datatype}')
    (
        test_observable_axis,
        test_agents,
        test_ensembles,
        test_observation_noise,
        test_sampling_rate,
    ) = load_parameters(datatype=datatype)
    names = []
    for observable_axis in test_observable_axis:
        for ensembles in test_ensembles:
            for observation_noise in test_observation_noise:
                for sampling_rate in test_sampling_rate:
                    name = Path(
                        name_from_parameters(
                            datatype=datatype,
                            n_agents=test_agents,
                            observable_axis=observable_axis,
                            ensembles=ensembles,
                            observation_noise=observation_noise,
                            sampling_rate=sampling_rate,
                        )
                    )
                    if name.is_dir():
                        names.append(name)
                    else:
                        raise Exception(
                            f'The Experiment {name} does not exist.'
                            'Please check your hyperparameters.'
                            )
    
    return names

def assignments_to_binary_trajectories(assignmentsT, steps):
    state_old = assignmentsT[:,:steps-1]
    state_new = assignmentsT[:,1:steps]
    return np.array(state_new==state_old, dtype=np.byte)

def analyze_single_experiment(path: str,):
    metrics = {
        'Hungarian Precision': [],
        'LPP': [],
        'Filter Consistency': []
    }
    
    metrics  = json.load(open(path + "metrics.json"))
    # print(metrics['max_length_analysis'])
    experiment_params  = json.load(open(path + "params.json"))
    seed = experiment_params['seeds'][0]
    assignmentsT = np.load(f"{path}{seed}_assignments.npy").T
    steps = experiment_params['steps']

    trajs = assignments_to_binary_trajectories(assignmentsT, steps)
    
    metrics['Filter Consistency'] = np.sum(trajs, axis=0)/50

    assignmentdata = assignmentsT.T    

    fig, axs = plt.subplots(1, 2, figsize=(15, 5),)
    axs[0].plot(assignmentdata)
    axs[1].plot(metrics['Filter Consistency'], label='Avg Max Consistency')
    axs[1].plot(metrics['Average Hungarian Precision'], label='Avg Hungarian Precision')
    axs[1].plot(metrics['Average LPP'], label='Avg LPP')
    axs[1].legend()
    axs[0].set_xlabel('Step')
    axs[1].set_xlabel('Step')
    axs[0].set_ylabel('Assignments')
    axs[1].set_ylabel('Metric')
    fig.suptitle(f'Experiment: {str(Path(path).name)}')
    plt.savefig('../nonlineardynamics23/vicsek-data-assimilation/saves/plots/single_experiment.jpg')
    plt.show()

        
if __name__ == "__main__":
    # read_and_eval('Obsv_noise_0.1')
    p = '/home/henrik/projects/nonlineardynamics23/Flocking_1100/Flocking_1111_50_50_0.1_True/'
    read_and_eval(p)
    analyze_single_experiment(p)
        
        
    