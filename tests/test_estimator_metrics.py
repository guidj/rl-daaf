import numpy as np

from daaf import estimator_metrics


def test_rmse():
    output_unit = estimator_metrics.rmse(pred=np.array(7), actual=np.array(6))
    output_vector = estimator_metrics.rmse(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6])
    )
    output_matrix = estimator_metrics.rmse(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 1.0, significant=6)
    np.testing.assert_approx_equal(output_vector, 2.16024690, significant=6)
    np.testing.assert_approx_equal(output_matrix, 0.61237246, significant=6)


def test_rmse_with_mask():
    output_unit = estimator_metrics.rmse(
        pred=np.array(7), actual=np.array(6), mask=np.array(1)
    )
    output_vector = estimator_metrics.rmse(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6]), mask=np.array([0, 1, 0])
    )
    output_matrix = estimator_metrics.rmse(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
        mask=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 1.0, significant=6)
    np.testing.assert_approx_equal(output_vector, 2.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, 0.8660254, significant=6)


def test_rmsle():
    output_unit = estimator_metrics.rmsle(pred=np.array(7), actual=np.array(6))
    output_vector = estimator_metrics.rmsle(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6])
    )
    output_matrix = estimator_metrics.rmsle(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 0.13353145, significant=6)
    np.testing.assert_approx_equal(output_vector, 0.59289280, significant=6)
    np.testing.assert_approx_equal(output_matrix, np.nan, significant=6)


def test_rmsle_with_mask():
    output_unit = estimator_metrics.rmsle(
        pred=np.array(7), actual=np.array(6), mask=np.array(1)
    )
    output_vector = estimator_metrics.rmsle(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6]), mask=np.array([0, 1, 0])
    )
    output_matrix = estimator_metrics.rmsle(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
        mask=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 0.13353134, significant=6)
    np.testing.assert_approx_equal(output_vector, 0.51082562, significant=6)
    np.testing.assert_approx_equal(output_matrix, np.nan, significant=6)


def test_rmsle_with_translate():
    untranslated_output = estimator_metrics.rmsle(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
        translate=False,
    )
    translated_output = estimator_metrics.rmsle(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
        translate=True,
    )
    np.testing.assert_approx_equal(untranslated_output, np.nan, significant=6)
    np.testing.assert_approx_equal(translated_output, 0.42446423, significant=6)


def test_mean_error():
    output_unit = estimator_metrics.mean_error(pred=np.array(7), actual=np.array(6))
    output_vector = estimator_metrics.mean_error(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6])
    )
    output_matrix = estimator_metrics.mean_error(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 1.0, significant=6)
    np.testing.assert_approx_equal(output_vector, -2.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, 3 / 8, significant=6)


def test_mean_error_with_mask():
    output_unit = estimator_metrics.mean_error(
        pred=np.array(7), actual=np.array(6), mask=np.array(1)
    )
    output_vector = estimator_metrics.mean_error(
        pred=np.array([0, 2, 3]), actual=np.array([1, 4, 6]), mask=np.array([0, 1, 0])
    )
    output_matrix = estimator_metrics.mean_error(
        pred=np.array([[0, 0], [0, 0], [0, 0], [-1, 0]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
        mask=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    )
    np.testing.assert_approx_equal(output_unit, 1.0, significant=6)
    np.testing.assert_approx_equal(output_vector, -2.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, 3 / 4, significant=6)


def test_pearson_correlation():
    output_vector, _ = estimator_metrics.pearson_correlation(
        pred=np.array([1, 2, 3]), actual=np.array([4, 5, 6])
    )
    output_matrix, _ = estimator_metrics.pearson_correlation(
        pred=np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
    )
    np.testing.assert_approx_equal(output_vector, 1.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, 1.0, significant=6)


def test_pearson_correlation_with_mask():
    output_vector, _ = estimator_metrics.pearson_correlation(
        pred=np.array([1, 2, 3]), actual=np.array([4, 5, 6]), mask=np.array([0, 1, 1])
    )
    output_matrix, _ = estimator_metrics.pearson_correlation(
        pred=np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        actual=np.array([[-1, 0], [-2, 0], [-3, 0], [-4, 0]]),
        mask=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    )
    np.testing.assert_approx_equal(output_vector, 1.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, -1, significant=6)


def test_spearman_correlation():
    output_vector, _ = estimator_metrics.spearman_correlation(
        pred=np.array([1, 2, 3]), actual=np.array([4, 5, 6])
    )
    output_matrix, _ = estimator_metrics.spearman_correlation(
        pred=np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        actual=np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]]),
    )
    np.testing.assert_approx_equal(output_vector, 1.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, 1.0, significant=6)


def test_spearman_correlation_with_mask():
    output_vector, _ = estimator_metrics.spearman_correlation(
        pred=np.array([1, 2, 3]), actual=np.array([4, 5, 6]), mask=np.array([0, 1, 1])
    )
    output_matrix, _ = estimator_metrics.spearman_correlation(
        pred=np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        actual=np.array([[-1, 0], [-2, 0], [-3, 0], [-4, 0]]),
        mask=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    )
    np.testing.assert_approx_equal(output_vector, 1.0, significant=6)
    np.testing.assert_approx_equal(output_matrix, -1, significant=6)
