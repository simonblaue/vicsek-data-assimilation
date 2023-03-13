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
    