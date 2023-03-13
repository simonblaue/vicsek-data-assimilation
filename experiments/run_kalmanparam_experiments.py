
from tqdm.auto import tqdm




from analyze.analyze import read_and_eval
from src.generate import execute_experiment, base_parameters
from src.misc import bools2str

parameters = base_parameters
test_observable_axis = [(True,True,True,True),(True,True,True,False),(True,True,False,False),(True,True,False,True)]
test_ensembles = [2,10,25,50,100]
test_observation_noise = [0.0001,0.001,0.01,0.1,1]
test_shuffle = [True,False]

#### GRID SUCHEN !!!! ###

def grid_search(parameters,phaseparameters, kind):

    # Noisestrength and Alignment strength are set to specific phase
    for p in phaseparameters:
        parameters[p] = phaseparameters[p]

    parameters['seed'] = [1,2,3,4,5,6,7,8,9,10]
    
    for observable_axis in tqdm(test_observable_axis, position=0, leave=False):
        for ensembles in tqdm(test_ensembles, position=2, leave=False):
            for observation_noise in tqdm(test_observation_noise, position=3, leave=False):
                for shuffle in tqdm(test_shuffle, position=4, leave=False):
                    name = f"{kind}_{bools2str(observable_axis)}_50_{ensembles}_{observation_noise}_{shuffle}"
                    parameters['observable_axis'] = observable_axis
                    parameters['n_ensembles'] = ensembles
                    parameters['observation_noise'] = observation_noise
                    parameters['shuffle_measurements'] = shuffle 
                    parameters['name'] = name

                    if observable_axis[2] == False:
                        parameters['find_velocities'] = True 

                    
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
        

