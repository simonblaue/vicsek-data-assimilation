# analyze a set of eperiments (given paths) with metrics
import json
import numpy as np 
from misc import metric_lost_particles, metric_hungarian_precision


def read_and_eval(experiment_name):
    filter_states, model_states, params = read_experiment(experiment_name)
    evaluate_experiment(filter_states, model_states, params)

def read_experiment(experiment_name : str):
    folder = "saves/"+ experiment_name + "/"
    experiment_params  = json.load(open(folder + "params.json"))
    seeds = experiment_params['seeds']
    model_states = []
    filter_states = []
    for seed in seeds:
        model_states.append(np.load(f"{folder}{seed}_model.npy"))
        filter_states.append(np.load(f"{folder}{seed}_filter.npy"))
    return model_states, filter_states, experiment_params


def evaluate_experiment(model_states, filter_states, experiment_params):
    
    metrics = {
        'Hungarian Precision':[],
        'Lost Particle Precison':[],
        'Average Hungarian': [],
        'Average LPP': []
    }
    
    seeds = experiment_params['seeds']
    measure_freq  = experiment_params['sampling_rate']
    
    measure_steps = int(experiment_params['steps']/measure_freq)
    
    average_hung = np.zeros(measure_steps)
    average_lpp = np.zeros(measure_steps)

    for seed in range(len(seeds)):
        model_pos = model_states[seed][::measure_freq,:, 0:2]
        
        filter_pos = filter_states[seed][:,:,0:2]
        
        metrics['Hungarian Precision']
        
        hung_metric = []
        lpp_metric = []
        
        for i in range(measure_steps) :
            m_hung = metric_hungarian_precision(model_pos[i], filter_pos[i])
            m_lpp = metric_lost_particles(model_pos[i], filter_pos[i], experiment_params['alignment_radius']/5)
            hung_metric.append(m_hung)
            lpp_metric.append(m_lpp)
            
            average_hung[i] += m_hung
            average_lpp[i] += m_lpp
            
        metrics['Hungarian Precision'].append(hung_metric)
        metrics['Lost Particle Precison'].append(lpp_metric)
        
    average_hung /= len(seeds)
    average_lpp /= len(seeds)
    
    metrics['Average Hungarian'] = average_hung.tolist()
    metrics['Average LPP'] = average_lpp.tolist()
    
    experiment_name = experiment_params['name']
    experiment_path = f'saves/{experiment_name}/'
    with open(f'{experiment_path}metrics.json', 'w') as fp:
        # json.dump(experiment_params, fp, indent=4)
        json.dump(metrics, fp, indent=4)
        
        
        
if __name__ == "__main__":
    read_and_eval('Baseline')
        
        
        
        
    