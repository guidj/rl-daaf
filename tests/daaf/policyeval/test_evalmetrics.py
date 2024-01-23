import numpy as np

from daaf.policyeval import evalmetrics


def test_rmse():
    evalmetrics.rmse(np.array([[1, 2], [0, 1], [3, 4]]), np.array([3, 1]))


def test_mean_absolute_error():
    evalmetrics.mean_absolute_error(
        np.array([[1, 2], [0, 1], [3, 4]]), np.array([3, 1])
    )


def test_normd_rmse():
    evalmetrics.normd_rmse(np.array([[1, 2], [0, 1], [3, 4]]), np.array([3, 1]))
