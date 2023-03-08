import numpy as np
import json
import os


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