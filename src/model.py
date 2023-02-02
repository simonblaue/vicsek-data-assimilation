import numpy as np


class WalkerConf():
    def __init__(self, radius, noise_strength) -> None:
        self.radius = radius
        self.noise_strength = noise_strength
        

def init_walkers(n):
    # Array with posx, posy, orientation 
    walkers = np.random.rand(n, 3)
    return walkers


def step(walkers):
    return 
