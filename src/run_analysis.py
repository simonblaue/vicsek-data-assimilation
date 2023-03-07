import numpy as np
import tqdm
import json
from analyze import experiments_to_analyze, assignments_to_binary_trajectories
    

def analysis():
    experiment_paths = experiments_to_analyze(datatype='selfgenerated')
    # experiment_paths = experiments_to_analyze(datatype='provided')
    results = {}
    for folder in tqdm(experiment_paths):
        # get params, id, assignments
        experiment_params  = json.load(open(str(folder) + "/params.json"))
        experiment_id = experiment_params["name"]
        assignmentsT = np.vstack(
            [np.load(str(file)).T for file in folder.glob('*assignments.npy')]
        )
        # shift to generate trajectories
        trajs = assignments_to_binary_trajectories(assignmentsT, experiment_params['steps'])
        # trajs to str to max len
        trajs = [''.join(map(str, t)) for t in trajs]
        max_lengths = [max([len(s) for s in ts.split('0')]) for ts in trajs]
        # result and save
        result = (np.mean(max_lengths), np.std(max_lengths))
        experiment_params['max_length_analysis'] = result
        with open(open(str(folder) + "/params.json"), 'w') as fp:
            json.dump(experiment_params, fp, indent=4)
        results[experiment_id] = result
        return results
    
    
# TODO: analyze single experient
