import pandas as pd
import numpy as np
import time

###################################################################################
###################################################################################

def compute_channel_gain(
    rayleigh_distributed_small_scale_fading: np.ndarray,
    distance: float,
    path_loss_exponent: int,  ## this is determined in
) -> np.ndarray:
    return rayleigh_distributed_small_scale_fading / np.power(distance, path_loss_exponent / 2)


### Generate a random complex normal distribution, which follows the normal distribution with mean 0 and variance 1
def generate_complex_normal_distribution(size: int = 1):
    return np.random.normal(loc=0, scale=1, size=size) + 1j * np.random.normal(loc=0, scale=1, size=size)