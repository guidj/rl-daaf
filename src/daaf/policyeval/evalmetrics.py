"""
Eval metrics for aggregation pipeline.
"""

import numpy as np


def rmse(v_pred, v_true):
    """Works with batch data"""
    return np.sqrt(np.sum(np.power(v_pred - v_true, 2.0), axis=1) / len(v_true))


def mean_absolute_error(v_pred, v_true, axis: int):
    """Works with batch data"""
    delta = np.abs(v_pred - v_true)
    return np.mean(delta, axis=axis)


def normd_rmse(v_pred, v_true):
    """Works with batch data"""
    rmse_ = rmse(v_pred, v_true=v_true)
    range_ = np.max(v_true) - np.min(v_true)
    return rmse_ / range_
