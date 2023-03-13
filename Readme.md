# Viszeck Model Analyses with an Ensemble Kalman Filter

## How to install


- ```cd ``` into this directory
- run ``` pip install . ```
- this should install all dependencies and make python able to import between different folders.
  

## How to use

- Everything can be run through the main.py file. Make sure to be in the top most folder when running this so that the saves folder will be also in top level.

### Into main.py

In the main file the following functions are available:

```python
    execute_experiment(base_parameters)
    visualize_experiment(experiment_name=base_parameters["name"])
    
    kalmanparam_grid_search(base_parameters,phase_1_flocking, "Flocking")
    kalmanparam_grid_search(base_parameters,phase_2_random, "Random")
    
    noise_grid_search(base_parameters,phase_1_flocking, "Flocking")
```

- The parameters used can be found in the folder experiments and parameters.py. There are also so me other intersting choices for different phases of the Viscek model. 
- Be aware that the grid searches can take quite long an also require each about 2 GB of storage.


### Into Visualization
