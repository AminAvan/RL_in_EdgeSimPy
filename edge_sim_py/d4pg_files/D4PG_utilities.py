import pandas as pd
import numpy as np
import time

###################################################################################
###################################################################################

def compute_channel_gain(
    rayleigh_distributed_small_scale_fading: np.ndarray,
    distance: float,
    path_loss_exponent: int,
) -> np.ndarray:
    return rayleigh_distributed_small_scale_fading / np.power(distance, path_loss_exponent / 2)