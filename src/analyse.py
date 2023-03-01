from config import VicsekAnimationConfig, RandomSimulationConfig, EnsembleKalmanConfig

if __name__ =="__main__":
    anim = VicsekAnimationConfig.exec_ref(
        animation_config=VicsekAnimationConfig,
        simulation_config=RandomSimulationConfig,
        kalman_config=EnsembleKalmanConfig
    )
    anim(save_name=False)
    
    
