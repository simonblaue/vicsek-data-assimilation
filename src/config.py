



@dataclass
class EnsembleKalmanConfig:
    
    exec_ref = EnsembleKalman
    
    n_ensembles: int = 100
    
    noise_ratio: float = 0.00000001
    
    n_ensembles: int = 50
    n_particles: int = 100
    x_axis: int = 25
    y_axis: int = 25
    
    state: np.ndarray = np.random.rand(n_particles, 3)
    
    model_forecast: callable = None
    epsilon: np.ndarray = np.ones((n_particles, n_particles))*1e-11