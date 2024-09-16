import pandas as pd
import numpy as np
import time

###################################################################################
###################################################################################

def compute_channel_gain(
    rayleigh_distributed_small_scale_fading: np.ndarray,
    distance: float,          ## I need to modify it to EdgeSimPy metric = latency or delay to edgeserver ==> edge_sim_py/components/user.py:207
    path_loss_exponent: int,  ## is determined in edge_sim_py/d4pg_files/D4PG_environmentConfig.py:48
) -> np.ndarray:
    return rayleigh_distributed_small_scale_fading / np.power(distance, path_loss_exponent / 2)


### Generate a random complex normal distribution, which follows the normal distribution with mean 0 and variance 1
def generate_complex_normal_distribution(size: int = 1):
    return np.random.normal(loc=0, scale=1, size=size) + 1j * np.random.normal(loc=0, scale=1, size=size)

## I should totally give the result of an equivalent def from EdgeSimPy to compute_transmission_rate()
def compute_transmission_rate(SINR, bandwidth) -> float:
    """
    :param SNR:   ==>>   SINR=vehicle_SINR[vehicle_index, e]=compute_SINR()
                                compute_SINR(
                                white_gaussian_noise=self._config.white_gaussian_noise,
                                channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                                transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                                intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][e],
                                inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][e],)
    :param bandwidth:
    :return: transmission rate measure by bit/s
    """
    return float(cover_MHz_to_Hz(bandwidth) * np.log2(1 + SINR))


def compute_SINR(
    white_gaussian_noise: int,
    channel_condition: float,
    transmission_power: float,
    intra_edge_interference: float,
    inter_edge_interference: float
) -> float:
    """
    Compute the SINR of a vehicle transmission
    Args:
        white_gaussian_noise: the white gaussian noise of the channel, e.g., -70 dBm
        channel_fading_gain: the channel fading gain, e.g., Gaussion distribution with mean 2 and variance 0.4
        distance: the distance between the vehicle and the edge, e.g., 300 meters
        path_loss_exponent: the path loss exponent, e.g., 3
        transmission_power: the transmission power of the vehicle, e.g., 10 mW
    Returns:
        SNR: the SNR of the transmission
    """
    # print("cover_dBm_to_W(white_gaussian_noise): ", cover_dBm_to_W(white_gaussian_noise))
    # print("intra_edge_interference: ", intra_edge_interference)
    # print("inter_edge_interference: ", inter_edge_interference)
    # print("channel_condition: ", channel_condition)
    # print("cover_mW_to_W(transmission_power): ", cover_mW_to_W(transmission_power))
    # print("noise plus interference: ", (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference + inter_edge_interference))
    # print("signal: ", channel_condition * cover_mW_to_W(transmission_power))
    return (1.0 / (cover_dBm_to_W(white_gaussian_noise) + intra_edge_interference + inter_edge_interference)) * \
        np.power(np.absolute(channel_condition), 2) * cover_mW_to_W(transmission_power)


def cover_mW_to_W(mW: float) -> float:
    return mW / 1000