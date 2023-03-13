import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


##############################################################################
#Obtaining Data, Experiments etc. 
##############################################################################

def get_params_metrics(exp):
    """
    Input: Name of the folder with the Experiment
    Returns: Parameter and Metric file 
    """
    s = exp.split('_')
    folder = f'{s[0]}_{s[1]}'
    basepath = f'saves/{folder}/{exp}'
    
    with open(f'{basepath}/params.json') as paramfile:
        params = json.load(paramfile)

    with open(f'{basepath}/metrics.json') as metricfile:
        metrics = json.load(metricfile)
        
    return params,metrics


def get_experiments(basepath='saves/',folder='Flocking1111'):
    """
    Input: Path of the Saves Folder
    Output: A list of all Experiment names
    """
    path = f'{basepath}/{folder}'

    experiments = os.listdir(path)

    removes = ['Baseline','.DS_Store','animations','__MACOSX']

    for rem in removes:
        try:
            experiments.remove(rem)
        except:
            None
        
    return experiments


def get_avg_metric(experiment,startidx=176,metric='Average Hungarian Precision'):
    """
    Input: Experiment (folder name) and Metric (Hungarian or LPP)
    Output: Mean Precision starting at Timestep 176
    """
    p,m = get_params_metrics(experiment)
    
    x = np.arange(0,round(p['steps']/p['timestepsize']),p['sampling_rate']*p['timestepsize'])
    y = np.array(m[metric])
    avg_metric = np.mean(y[list(x).index(startidx):])
    
    return avg_metric

def get_metric_length(experiment):

    p,m = get_params_metrics(experiment)

    metric = m['max_length_analysis']
    mean = metric[0]
    variance = metric[1]

    return mean,variance 


##############################################################################
#Plotting routines 
##############################################################################

def plot_metric(ax,exp,folder='Flocking1111',metric='Average Hungarian Precision',color='Royalblue',title=True,legend=False,errors=True,label=None,yscale=True):
    p,m = get_params_metrics(exp)
    
    x = np.arange(0,round(p['steps']/p['timestepsize']),p['sampling_rate']*p['timestepsize'])
    y = np.array(m[metric])
    
    ax.plot(x,y,label=label,color=color)
    ax.set_xlabel('Time')
    ax.set_ylabel('Precision')
    
    metric_error = metric.replace('Average','Var')
    
    if errors:
        error = 1*np.array(m[metric_error])
        ax.fill_between(x,y-error,y+error,alpha=0.5,color=color)
    
    if legend:
        ax.legend()
        
    if title:
        header = get_header(exp)
        ax.set_title(header)
        
    if yscale:
        ax.set_ylim(0,1.1)
    
    return None



def get_header(exp):
    """
    Input: Folder Name of the Experiment
    Returns: The Title including the Experiment Parameters
    """
    s = exp.split('_')
    phase = s[0]
    obs_axis = s[1]
    nr_agents = s[2]
    nr_ensembles = s[3]
    obs_noise = s[4]
    shuffling = s[5]

    if obs_axis == '1100':
        obs_axis='Only Positions'
    elif obs_axis == '1101':
        obs_axis = 'Positions, Angles'
    elif obs_axis == '1110':
        obs_axis = 'Positions, Velocity'
    else:
        obs_axis = 'Positions, Velocity, Angles'

    if shuffling == 'True':
        shuffling = 'On'
    else:
        shuffling = 'Off'

    title = f"Phase: {phase}, Observed: {obs_axis}, \n Agents:{nr_agents}, Ensembles:{nr_ensembles},\n Obs. Noise:{obs_noise}, Shuffling:{shuffling}"
    return title 

# Flocking_50_10_0.01_0.1_0.08
# experiment particles ensembles obs position theta

def plot_all(
    path = "saves/", # path to experiment folder
    experiment = 'Flocking',
    ensembles = [10, 25, 50, 75][3],
    colors = ['navy', 'darkred', 'orange',
    ]
):
    max_lengths = {}

    for folder in list(Path(path).iterdir()):
        parameters = json.load(open(str(folder)+'/params.json'))
        metrics = json.load(open(str(folder)+'/metrics.json'))
        max_lengths[parameters['name']] = metrics['max_length_analysis']
        
    scale = 2
    test_ensembles = [10, 25, 50, 75]
    test_observation_noise = [0.01, 0.05, 0.1]
    ensemble_theta_noise = [0.08, 0.2, 0.5]
    ensemble_pos_noise = [0.002, 0.1, 0.3]
    fig, axs = plt.subplots(
        len(ensemble_theta_noise), len(ensemble_pos_noise), figsize=(len(ensemble_pos_noise)*3, len(ensemble_theta_noise)*3),
        sharey=True
    )
    for j in range(len(ensemble_theta_noise)):
        for i in range(len(ensemble_pos_noise)):
            # Flocking_50_10_0.01_0.1_0.08
            # experiment particles ensembles obs position theta
            names = [
                f'{experiment}_50_{ensembles}_{noise}_{ensemble_pos_noise[i]}_{ensemble_theta_noise[j]}' for noise in test_observation_noise
            ]
            mls = np.array([max_lengths[n] for n in names]).T
            axs[j][i].set_xlabel('Observation Noise')
            axs[j][i].grid(which='major', axis='y', linestyle='--')
            
            axs[j][i].set_ylim((0, 350))
            axs[j][i].scatter(test_observation_noise, mls[0],)
            axs[j][i].set_xscale('log')
            ob_axis = f'Angle Noise: {ensemble_theta_noise[j]}'
            axs[j][i].errorbar(test_observation_noise, mls[0],yerr=mls[1], capsize=5, elinewidth=1, fmt='o', c=colors[j], label=ob_axis)
            if j == 0:
                axs[j][i].set_title(f'Position Noise: {ensemble_pos_noise[i]}')
            if i == 0:
                axs[j][i].set_ylabel('Tracking Consistency (steps)')
                axs[j][i].legend()
    plt.title(f'Ensebles: {ensembles}')
    plt.tight_layout()
    plt.savefig(f'saves/plots/grid_seach_{ensembles}.jpg')
    plt.show()
    
