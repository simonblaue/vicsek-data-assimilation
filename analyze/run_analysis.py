import numpy as np
import tqdm
import json
from pathlib import Path
from analyze.analyze import experiments_to_analyze, assignments_to_binary_trajectories
    

def analysis(dir):
    results = {}
    for folder in tqdm.tqdm(list(Path(dir).iterdir())):
        # get params, id, assignments
        experiment_params  = json.load(open(str(folder) + "/params.json"))
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
        # print(str(folder) + "/metrics.json")
        metrics = json.load(open(str(folder) + "/metrics.json"))
        metrics['max_length_analysis'] = result
        with open(str(folder) + "/metrics.json", 'w') as fp:
                json.dump(metrics, fp, indent=4)
    return results
    
# TODO: analyze single experient

if __name__ == "__main__":
    analysis('/home/henrik/projects/nonlineardynamics23/Flocking1111/')

