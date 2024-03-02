import numpy as np

from daaf import evalmetrics


def test_rmse():
    obs = np.tile(np.array([3, 1]), reps=(3, 1))
    axis_0 = evalmetrics.rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0)
    axis_1 = evalmetrics.rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1)

    np.testing.assert_allclose(axis_0, [np.sqrt(13 / 3), np.sqrt(10 / 3)])
    np.testing.assert_allclose(axis_1, [np.sqrt(5 / 2), np.sqrt(9 / 2), np.sqrt(9 / 2)])


def test_normd_rmse():
    obs = np.tile(np.array([3, 1]), reps=(3, 1))
    axis_0 = evalmetrics.normd_rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0)
    axis_1 = evalmetrics.normd_rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1)

    np.testing.assert_allclose(
        axis_0,
        [
            np.inf,
            np.inf,
        ],
    )
    np.testing.assert_allclose(
        axis_1,
        [
            np.sqrt(5 / 2) / 2,
            np.sqrt(9 / 2) / 2,
            np.sqrt(9 / 2) / 2,
        ],
    )


def test_mean_absolute_error():
    obs = np.tile(np.array([3, 1]), reps=(3, 1))
    axis_0 = evalmetrics.mean_absolute_error(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0
    )
    axis_1 = evalmetrics.mean_absolute_error(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1
    )

    np.testing.assert_allclose(axis_0, [5 / 3, 4 / 3])
    np.testing.assert_allclose(axis_1, [3 / 2, 3 / 2, 3 / 2])
