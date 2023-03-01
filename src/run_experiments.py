from generate import execute_experiment
from analyze import read_and_eval

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
    ensemble_size_exp()