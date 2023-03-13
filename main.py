
from src.generate import execute_experiment, parameters_for_given_data
from visualization.visualize import visualize_experiment, visualize_dataset

from experiments.parameters import base_parameters, parameters_for_given_data, phase_1_flocking, phase_2_random, phase_3_jonas


from experiments.run_kalmanparam_experiments import grid_search as kalmanparam_grid_search
from experiments.run_noise_experiments import grid_search as noise_grid_search

from visualization.plotting import plot_all

if __name__ == "__main__":
    """
    Uncomment the lines you want to execute
    """

    execute_experiment(base_parameters) # You can change the parameters to parameters_for_given_data, phase_1_flocking, phase_2_random, phase_3_jonas
    visualize_experiment(experiment_name=base_parameters["name"])
    
    # kalmanparam_grid_search(base_parameters,phase_1_flocking, "Flocking")
    # kalmanparam_grid_search(base_parameters,phase_2_random, "Random")
    
    #noise_grid_search(base_parameters,phase_1_flocking, "Flocking")
    
    #plot_all()