import numpy as np

def load_parameters(datatype='selfgenerate'):
    if datatype == 'selfgenerate':
        observable_axis = [(True,True,True,True,True),(True,True,True,True,False),(True,True,False,False,False)]
        agents = [50]
        ensembles = [50,100,150,200,250]
        observation_noise = [0.0001 ,0.001, 0.01, 0.1, 1]
        sampling_rate = [1,2,4]
    elif datatype == 'provided':
        observable_axis = [(True,True,True,True,True),(True,True,True,True,False),(True,True,False,False,False)]
        agents = [361]
        ensembles = [50,100,150,200,250]
        observation_noise = [0.0001 ,0.001, 0.01, 0.1, 1]
        sampling_rate = [1,]
    else:
        raise Exception(
            'datatype must be either '
            '\'selfgenerate\' (Simulation Data was self-generated) '
            'or \'provided\' (Simulation Data was provided by Jonas) '
            
        )
    
    return (
        observable_axis,
        agents,
        ensembles,
        observation_noise,
        sampling_rate,
    )
    
    
base_parameters = {
    'name': 'Baseline',
    'seeds': [np.random.randint(1,5)],
    'steps': 200,
    'timestepsize': 1,
    'n_particles': 50,
    'n_ensembles': 50,
    'observation_noise': 0.1,
    'alignment_strength':0.05,
    'noisestrength': 0.15,
    'velocity': 0.05,
    'sampling_rate': 1,
    'alignment_radius': 1,
    'observable_axis': (True,True,True,True),
    'box_size': 10,
    'shuffle_measurements': False,
    'ensemble_pos_noise' : 0.03,
    'ensemble_theta_noise': 0.2,
    'save_name' : 'None'
    }

        
phase_1_flocking = {
        'name': 'Flocking',
        'noisestrength': 0.15,
        'alignment_strength':0.1,
}

phase_2_random = {
        'name': 'Random',
        'noisestrength':0.2,
        'alignment_strength':0.025,
}

phase_3_jonas = {
        'name': 'Jonas',
        'n_particles':70,
        #'x_axis':50,
        #'y_axis':50,
        'noisestrength':0.1,
        'alignment_strength':0.05,
}

parameters_for_given_data = {
    'name': 'GivenData',
    'seeds': [np.random.randint(1,5)],
    'steps': 200,
    'timestepsize': 1,
    'n_particles': 361,
    'n_ensembles': 2,
    'observation_noise': 0.000,
    'alignment_strength':0.05,
    'noisestrength': 0.15,
    'velocity': 0.09,
    'sampling_rate': 1,
    'alignment_radius': 1,
    'observable_axis': (True,True,True,True),
    'x_axis': 50,
    'y_axis': 50,
    'find_velocities': False,
    'shuffle_measurements': False,
    'save_name' : 'None'
    }