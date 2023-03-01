from generate import execute_experiment
from analyze import read_and_eval
from misc import bools2str

parameters = {
        'name': 'Baseline',
        'seeds': [1,2,3,4,5,6,7,8,9,10],
        'steps': 200,
        'timestepsize': 1,
        'n_particles': 50,
        'n_ensembles': 100,
        'observation_noise': 0.01,
        'viscec_noise': 0.5,
        'xi' : 0.8,
        'noisestrength':0.5,
        'velocity': 0.03,
        'sampling_rate': 2,
        'alignment_radius': 1.,
        'observable_axis': (True,True,False),
        'x_axis': 10,
        'y_axis': 10,
        }


#### GRID SUCHEN !!!! ###

def grid_search():
    test_observable_axis = [(True,True,True),(True,True,False)]
    test_agents = [50,100]
    test_ensembles = [50,100,150,200,250]
    test_observation_noise = [0.0001 ,0.001, 0.01, 0.1, 1]
    test_sampling_rate = [1,2,4]

    for observable_axis in test_observable_axis:
        for agents in test_agents:
            for ensembles in test_ensembles:
                for observation_noise in test_observation_noise:
                    for sampling_rate in test_sampling_rate:
                        name = f"{bools2str(observable_axis)}_{agents}_{ensembles}_{observation_noise}_{sampling_rate}"
                        parameters['observable_axis'] = observable_axis
                        parameters['agents'] = agents
                        parameters['observation_noise'] = observation_noise
                        parameters['sampling_rate']  = sampling_rate
                        parameters['name'] = name
                        print(name)
                    execute_experiment(parameters)
                    read_and_eval(name)

####### Experiment for Observation Noise ######## 

def observation_noise_exp():
    
    test_values = [0.0001 ,0.001, 0.01, 0.1, 1]
    names = [f"Obsv_noise_{v}" for v in test_values]
    
    for name, test_value in zip(names,test_values):
        parameters['observation_noise'] = test_value
        parameters['name'] = name
        execute_experiment(parameters)
        read_and_eval(name)
    
    
    
def ensemble_size_exp():
    test_values = [10,50,100,150,200]
    names = [f"ensamble_size_{v}" for v in test_values]
    
    for name, test_value in zip(names,test_values):
        parameters['n_ensambles'] = test_value
        parameters['name'] = name
        execute_experiment(parameters)
        read_and_eval(name)
        
if __name__ == "__main__":
    grid_search()