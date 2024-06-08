"""
Eval metrics for aggregation pipeline.
"""

import numpy as np
from scipy.spatial import distance


def rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if v_pred.shape != v_true.shape:
        raise ValueError(
            f"Tensors have different shapes: {v_pred.shape} != {v_true.shape}"
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
            f"Tensors have different shapes: {v_pred.shape} != {v_true.shape}"
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


def cosine_distance(v_pred: np.ndarray, v_true: np.ndarray):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if v_pred.shape != v_true.shape:
        raise ValueError(
            f"Tensors have different shapes: {v_pred.shape} != {v_true.shape}"
        )

    if len(v_pred.shape) != 2:
        raise ValueError(f"Tensors are not 2-dim: {v_pred.shape}, {v_true.shape}")

    cosines = []
    for row in range(v_pred.shape[0]):
        cosines.append(distance.cosine(v_pred[row], v_true[row]))
    return np.array(cosines, dtype=v_true.dtype)


def dotproduct(v_pred: np.ndarray, v_true: np.ndarray):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if v_pred.shape != v_true.shape:
        raise ValueError(
            f"Tensors have different shapes: {v_pred.shape} != {v_true.shape}"
        )

    if len(v_pred.shape) != 2:
        raise ValueError(f"Tensors are not 2-dim: {v_pred.shape}, {v_true.shape}")

    dps = []
    for row in range(v_pred.shape[0]):
        dps.append(np.dot(v_pred[row], v_true[row]))
    return np.array(dps, dtype=v_true.dtype)
