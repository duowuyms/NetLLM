import torch
import numpy as np


def normalize_data(data, dataset, dataset_type):
    """
    Normalize data into [-1, 1].
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    normalized_data = torch.zeros_like(data)
    if dataset_type == '360':
        # for 360 video, we only need to do the following operations:
        # as roll falls within [-180, 180], do roll / 180;
        # as pitch falls within [-90, 90], do pitch / 90;
        # as yaw falls within [-180, 180], do yaw / 180
        normalized_data[..., 0] = data[..., 0] / 180  # roll
        normalized_data[..., 1] = data[..., 1] / 90  # pitch
        normalized_data[..., 2] = data[..., 2] / 180 # yaw
    elif dataset_type == 'vv':
        # for translational postions, depends on specific datasets
        if dataset == 'Serhan2020':
            normalized_data[..., 3] = data[..., 3] / 180  # roll
            normalized_data[..., 4] = data[..., 4] / 90  # pitch
            normalized_data[..., 5] = data[..., 5] / 180 # yaw
            # Serhan2020 dataset does not provide the scene information
            # so we analyze the dataset, and find that positions fall within:
            # x in [-5.339652, 3.314852]
            # y in [-0.863402, 0.150379]
            # z in [-2.386547, 3.843916]
            # so we divide x by 5.5, y by 1.0, z by 4.0
            normalized_data[..., 0] = data[..., 0] / 5.5  # x
            normalized_data[..., 1] = data[..., 1] / 1.0  # y
            normalized_data[..., 2] = data[..., 2] / 4.0  # z
        elif dataset == 'Hu2023':
            normalized_data[..., 3:] = data[..., 3:] / 360  
            normalized_data[..., 0] = data[..., 0] / 5.5  # x
            normalized_data[..., 1] = data[..., 1] / 1.0  # y
            normalized_data[..., 2] = data[..., 2] / 4.0  # z
    return normalized_data

def denormalize_data(normalized_data, dataset, dataset_type):
    """
    Denormalize the normalized data.
    In other words, this function implements the reverse operations of function normalize_data.
    """
    denormalized_data = torch.zeros_like(normalized_data)
    if dataset_type == '360':
        denormalized_data[..., 0] = normalized_data[..., 0] * 180  # roll
        denormalized_data[..., 1] = normalized_data[..., 1] * 90  # pitch
        denormalized_data[..., 2] = normalized_data[..., 2] * 180  # yaw
    elif dataset_type == 'vv':
        if dataset == 'Serhan2020':
            denormalized_data[..., 3] = normalized_data[..., 3] * 180  # roll
            denormalized_data[..., 4] = normalized_data[..., 4] * 90  # pitch
            denormalized_data[..., 5] = normalized_data[..., 5] * 180  # yaw
            denormalized_data[..., 0] = normalized_data[..., 0] * 5.5  # x
            denormalized_data[..., 1] = normalized_data[..., 1] * 1.0  # y
            denormalized_data[..., 2] = normalized_data[..., 2] * 4.0  # z
        elif dataset == 'Hu2023':
            denormalized_data[..., 3:] = normalized_data[..., 3:] * 360 
            denormalized_data[..., 0] = normalized_data[..., 0] * 5.5  # x
            denormalized_data[..., 1] = normalized_data[..., 1] * 1.0  # y
            denormalized_data[..., 2] = normalized_data[..., 2] * 4.0  # z

    return denormalized_data
