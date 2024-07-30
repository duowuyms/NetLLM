import torch
import numpy as np


def normalize_data(data, dataset):
    """
    Normalize data into [-1, 1].
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    normalized_data = torch.zeros_like(data)
    # for 360 video, we only need to do the following operations:
    # as roll falls within [-180, 180], do roll / 180;
    # as pitch falls within [-90, 90], do pitch / 90;
    # as yaw falls within [-180, 180], do yaw / 180
    normalized_data[..., 0] = data[..., 0] / 180  # roll
    normalized_data[..., 1] = data[..., 1] / 90  # pitch
    normalized_data[..., 2] = data[..., 2] / 180 # yaw
    return normalized_data

def denormalize_data(normalized_data, dataset):
    """
    Denormalize the normalized data.
    In other words, this function implements the reverse operations of function normalize_data.
    """
    denormalized_data = torch.zeros_like(normalized_data)
    denormalized_data[..., 0] = normalized_data[..., 0] * 180  # roll
    denormalized_data[..., 1] = normalized_data[..., 1] * 90  # pitch
    denormalized_data[..., 2] = normalized_data[..., 2] * 180  # yaw
    return denormalized_data
