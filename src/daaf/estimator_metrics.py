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
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )
    return np.sqrt(
        np.sum(np.power(v_pred - v_true, 2.0), axis=axis) / np.shape(v_pred)[axis]
    )


def mean_absolute_error(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
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
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )

    if len(np.shape(v_pred)) != 2:
        raise ValueError(
            f"Tensors are not 2-dim: {np.shape(v_pred)}, {np.shape(v_true)}"
        )

    cosines = []
    for row in range(np.shape(v_pred)[0]):
        cosines.append(distance.cosine(v_pred[row], v_true[row]))
    return np.array(cosines, dtype=v_true.dtype)


def dotproduct(v_pred: np.ndarray, v_true: np.ndarray):
    """
    Args:
        v_pred: An array of shape [b, k]
        v_true: An array of shape [b, k]
    """
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )

    if len(np.shape(v_pred)) != 2:
        raise ValueError(
            f"Tensors are not 2-dim: {np.shape(v_pred)}, {np.shape(v_true)}"
        )

    dps = []
    for row in range(np.shape(v_pred)[0]):
        dps.append(np.dot(v_pred[row], v_true[row]))
    return np.array(dps, dtype=v_true.dtype)
