import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path


##############################################################################
#Obtaining Data, Experiments etc. 
##############################################################################

def get_params_metrics(exp,folder='Flocking1111'):
    """
    Input: Name of the folder with the Experiment
    Returns: Parameter and Metric file 
    """
    basepath = f'../saves/{folder}/{exp}'
    
    with open(f'{basepath}/params.json') as paramfile:
        params = json.load(paramfile)

    with open(f'{basepath}/metrics.json') as metricfile:
        metrics = json.load(metricfile)
        
    return params,metrics


def get_experiments(basepath='../saves/',folder='Flocking1111'):
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


##############################################################################
#Plotting routines 
##############################################################################

def plot_metric(ax,exp,folder='Flocking1111',metric='Average Hungarian Precision',color='Royalblue',title=True,legend=False,errors=True,label=None,yscale=True):
    p,m = get_params_metrics(exp,folder)
    
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

def ob_axis_description(obs_axis):
    if obs_axis == '1100':
        return 'Only Positions'
    elif obs_axis == '1101':
        return 'Positions, Angles'
    elif obs_axis == '1110':
        return 'Positions, Velocity'
    else:
        return 'Positions, Velocity, Angles'


def plot_all(
    path = "/home/henrik/projects/nonlineardynamics23/saves/", # path to experiment folder
    experiment = 'Flocking',
    shuffle = True,
    test_observable_axis = ['1100','1101', '1110', '1111'],
    test_ensembles = [2,10,25,50,100],
    test_observation_noise = [0.0001,0.001,0.01,0.1,1],
    colors = ['navy', 'darkred',
    'darkgreen', 'orange',
    ]
):
    max_lengths = {}

    for folder in list(Path(path).iterdir()):
        parameters = json.load(open(str(folder)+'/params.json'))
        metrics = json.load(open(str(folder)+'/metrics.json'))
        max_lengths[parameters['name']] = metrics['max_length_analysis']
        
    scale = 2
    fig, axs = plt.subplots(
        len(test_observable_axis), len(test_ensembles), figsize=(len(test_ensembles)*3, len(test_observable_axis)*3),
        sharey=True
    )
    for j in range(len(test_observable_axis)):
        for i in range(len(test_ensembles)):
            names = [
                f'{experiment}_{test_observable_axis[j]}_50_{test_ensembles[i]}_{noise}_{shuffle}' for noise in test_observation_noise
            ]
            mls = np.array([max_lengths[n] for n in names]).T
            axs[j][i].set_xlabel('Observation Noise')
            axs[j][i].grid(which='major', axis='y', linestyle='--')
            
            axs[j][i].set_ylim((0, 350))
            axs[j][i].scatter(test_observation_noise, mls[0],)
            axs[j][i].set_xscale('log')
            ob_axis = ob_axis_description(f'{test_observable_axis[j]}')
            axs[j][i].errorbar(test_observation_noise, mls[0],yerr=mls[1], capsize=5, elinewidth=1, fmt='o', c=colors[j], label=ob_axis)
            if j == 0:
                axs[j][i].set_title(f'Ensembles: {test_ensembles[i]}')
            if i == 0:
                axs[j][i].set_ylabel('Tracking Consistency (steps)')
                axs[j][i].legend()
    plt.tight_layout()
    plt.savefig(f'../nonlineardynamics23/vicsek-data-assimilation/saves/plots/grid_seach_{shuffle}.jpg')
    plt.show()
    
if __name__ == "__main__":
    plot_all()