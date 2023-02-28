# generate an experiment with optional visualization
# parameters as dict
import time
from datetime import datetime
import numpy as np
from pathlib import Path
import json
from scrips import simulate

def execute_experiment():
    parameters = {
        'name': 'Baseline',
        'seed': 1,
        'steps': 100,
        'timestepsize': 1,
        'particles': 100,
        'ensembles': 50,
        'observation_noise': 0.0001,
        'viscec_noise': 0.8,
        'velocities': [0.03],
        'sampling_rate': 2,
        'alignment_radius': 1.,
        'observable_axis': (True,True,False),
        'x_axis': 10,
        'y_axis': 10,
    }

    experiment_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    experiment_id = parameters['name']+'_'+parameters['seed']+'_'+experiment_time
    # TODO
    savepath = f'/{experiment_id}/'
        
    # TODO: set seed

    t = time.time()
        
    viscecstates, filterstates, metrics = simulate(parameters)
        
    runtime = time.time()-t
    print(f'Runtime for {experiment_id}: \t {runtime}')
    print(f'Saving report and result to {savepath}')
    print()
        
    np.save(savepath+'model.npy', viscecstates)
    np.save(savepath+'filter.npy', filterstates)

    report_data = {**metrics, **parameters}
    report_data['runtime'] = runtime

    # saving parameters
    with open(f'{savepath}/{experiment_id}.json', 'w') as fp:
        json.dump(report_data, fp, indent=4)
    

if __name__ =="__main__":
    execute_experiment()