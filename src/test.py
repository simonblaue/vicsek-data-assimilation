
import config

if __name__ =="__main__":
    anim = config.VicsekAnimationConfig.exec_ref(
        animation_config=config.VicsekAnimationConfig,
        simulation_config=config.RandomSimulationConfig,
        kalman_config=config.EnsembleKalmanConfig
    )
    anim(save_name=False)


