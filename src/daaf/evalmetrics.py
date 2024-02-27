"""
Eval metrics for aggregation pipeline.
"""

import numpy as np


def rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if v_pred.shape != v_true.shape:
        raise ValueError(
            f"Arrays have different shapes: {v_pred.shape} != {v_true.shape}"
        )
    return np.sqrt(
        np.sum(np.power(v_pred - v_true, 2.0), axis=axis) / v_pred.shape[axis]
    )


def mean_absolute_error(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if v_pred.shape != v_true.shape:
        raise ValueError(
            f"Arrays have different shapes: {v_pred.shape} != {v_true.shape}"
        )
    delta = np.abs(v_pred - v_true)
    return np.mean(delta, axis=axis)


def normd_rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    # `rmse` checks for shape, so we skip it here.
    rmse_ = rmse(v_pred, v_true=v_true, axis=axis)
    range_ = np.max(v_true, axis=axis) - np.min(v_true, axis=axis)
    return rmse_ / range_
