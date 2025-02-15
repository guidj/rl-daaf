import numpy as np

from daaf import estimator_metrics


def test_rmse():
    obs = np.tile(np.array([3, 1]), reps=(3, 1))
    axis_0 = estimator_metrics.rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0)
    axis_1 = estimator_metrics.rmse(np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1)

    np.testing.assert_allclose(axis_0, [np.sqrt(13 / 3), np.sqrt(10 / 3)])
    np.testing.assert_allclose(axis_1, [np.sqrt(5 / 2), np.sqrt(9 / 2), np.sqrt(9 / 2)])


def test_normd_rmse():
    obs = np.tile(np.array([3, 1]), reps=(3, 1))
    axis_0 = estimator_metrics.normd_rmse(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0
    )
    axis_1 = estimator_metrics.normd_rmse(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1
    )

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
    axis_0 = estimator_metrics.mean_absolute_error(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=0
    )
    axis_1 = estimator_metrics.mean_absolute_error(
        np.array([[1, 2], [0, 1], [3, 4]]), obs, axis=1
    )

    np.testing.assert_allclose(axis_0, [5 / 3, 4 / 3])
    np.testing.assert_allclose(axis_1, [3 / 2, 3 / 2, 3 / 2])


def test_cosine_distance():
    xs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ys = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    zs = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    output_1 = estimator_metrics.cosine_distance(xs, ys)
    output_2 = estimator_metrics.cosine_distance(ys, zs)
    np.testing.assert_allclose(output_1, [1.0, 1.0])
    np.testing.assert_allclose(output_2, [0.29289321881345254, 1.0])


def test_dotproduct():
    xs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ys = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    zs = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
    ws = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]])

    output_1 = estimator_metrics.dotproduct(xs, ys)
    output_2 = estimator_metrics.dotproduct(ys, zs)
    output_3 = estimator_metrics.dotproduct(zs, ws)
    np.testing.assert_allclose(output_1, [0.0, 0.0])
    np.testing.assert_allclose(output_2, [1.0, 0.0])
    np.testing.assert_allclose(output_3, [1.0, 2.0])
