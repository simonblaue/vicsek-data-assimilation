import numpy as np

parameters = {
        'name': 'Baseline',
        'seeds': [np.random.randint()],
        'steps': 300,
        'timestepsize': 1,
        'n_particles': 50,
        'n_ensembles': 100,
        'observation_noise': 0.1,
        'alignment_strength':0.05,
        'noisestrength': 0.15,
        'velocity': 0.05,
        'sampling_rate': 1,
        'alignment_radius': 1,
        'observable_axis': (True,True,True,True),
        'x_axis': 10,
        'y_axis': 10,
        'find_velocities': False,
        'shuffle_measurements': False,
        'ensemble_pos_noise' : 0.03,
        'ensemble_theta_noise': 0.2,
        }


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
    