import numpy as np


def _compute_error(data1, data2, rotation=False):
    error = np.abs(data2 - data1)
    if rotation:
        error = np.minimum(error, np.abs(data2 + 360 - data1))
        error = np.minimum(error, np.abs(data2 - 360 - data1))
    return error


def compute_mae(data1, data2, rotation=False):
    mae = np.mean(_compute_error(data1, data2, rotation))
    return mae

def compute_each_mae(data1, data2, rotation=False):
    mae = np.mean(_compute_error(data1, data2, rotation), axis=(1,2))
    return mae


def compute_mse(data1, data2, rotation=False):
    mse = np.mean(_compute_error(data1, data2, rotation) ** 2)
    return mse

def compute_each_mse(data1, data2, rotation=False):
    mse = np.mean(_compute_error(data1, data2, rotation) ** 2, axis=(1,2))
    return mse


def compute_rmse(data1, data2, rotation=False):
    rmse = np.sqrt(compute_mse(data1, data2))
    return rmse

def compute_each_rmse(data1, data2, rotation=False):
    rmse = np.sqrt(compute_each_mse(data1, data2))
    return rmse

