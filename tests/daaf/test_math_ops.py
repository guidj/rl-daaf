import numpy as np

from daaf import math_ops


def test_least_squares_factors_present():
    examples = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert math_ops.least_squares_factors_present(examples)


def test_least_squares_factors_present_with_missing_entries():
    missing_0 = np.array([[0, 1, 0], [0, 0, 1]])
    missing_1 = np.array([[1, 0, 0], [0, 0, 1]])
    missing_2 = np.array([[1, 0, 0], [0, 1, 0]])

    assert not math_ops.least_squares_factors_present(missing_0)
    assert not math_ops.least_squares_factors_present(missing_1)
    assert not math_ops.least_squares_factors_present(missing_2)


def test_solve_least_squares():
    """
    Example 1:
    Two states (A, B), two actions (left, right)
    Table:
            Actions
    States  Left    Right
        A   0       1
        B   0       1

    events: (A, left, A, right) -> 0 + 1 = 1
            (B, left, B, right) -> 0 + 1 = 1
            (A, right, B, left) -> 1 + 0 = 1
            (A, right, B, right) -> 1 + 1 = 2

    matrix: (A, left), (A, right), (B, left), (B, right)
            1           1           0           0
            0           0           1           1
            0           1           1           0
            0           1           0           1
    rhs: 1, 1, 1, 2

    Example 2
    x = 1, y = 3
    ax + by = rhs
    3x + 2y = 3x1 + 2x3 = 9
    4x + 5y = 4x1 + 5x3 = 19
    0x + 1y = 0x1 + 1x3 = 3
    1x + 0y = 1x1 + 0x3 = 1

    """

    examples_0 = np.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float32
    )
    labels_0 = np.array([1, 1, 1, 2], np.float32)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float32)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float32)
    labels_1 = np.array([9, 19, 3, 1], np.float32)
    expected_1 = np.array([1.0, 3.0], np.float32)

    np.testing.assert_array_almost_equal(
        math_ops.solve_least_squares(examples_0, labels_0), expected_0, decimal=3
    )
    np.testing.assert_array_almost_equal(
        math_ops.solve_least_squares(examples_1, labels_1), expected_1, decimal=3
    )


def test_tf_solve_least_squares():
    """
    Example 1:
    Two states (A, B), two actions (left, right)
    Table:
            Actions
    States  Left    Right
        A   0       1
        B   0       1

    events: (A, left, A, right) -> 0 + 1 = 1
            (B, left, B, right) -> 0 + 1 = 1
            (A, right, B, left) -> 1 + 0 = 1
            (A, right, B, right) -> 1 + 1 = 2

    matrix: (A, left), (A, right), (B, left), (B, right)
            1           1           0           0
            0           0           1           1
            0           1           1           0
            0           1           0           1
    rhs: 1, 1, 1, 2

    Example 2
    x = 1, y = 3
    ax + by = rhs
    3x + 2y = 3x1 + 2x3 = 9
    4x + 5y = 4x1 + 5x3 = 19
    0x + 1y = 0x1 + 1x3 = 3
    1x + 0y = 1x1 + 0x3 = 1

    """

    examples_0 = np.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float32
    )
    labels_0 = np.array([1, 1, 1, 2], np.float32)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float32)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float32)
    labels_1 = np.array([9, 19, 3, 1], np.float32)
    expected_1 = np.array([1.0, 3.0], np.float32)

    np.testing.assert_array_almost_equal(
        math_ops._tf_solve_least_squares(examples_0, labels_0),
        expected_0,
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        math_ops._tf_solve_least_squares(examples_1, labels_1),
        expected_1,
        decimal=3,
    )


def test_np_solve_least_squares():
    """
    Example 1:
    Two states (A, B), two actions (left, right)
    Table:
            Actions
    States  Left    Right
        A   0       1
        B   0       1

    events: (A, left, A, right) -> 0 + 1 = 1
            (B, left, B, right) -> 0 + 1 = 1
            (A, right, B, left) -> 1 + 0 = 1
            (A, right, B, right) -> 1 + 1 = 2

    matrix: (A, left), (A, right), (B, left), (B, right)
            1           1           0           0
            0           0           1           1
            0           1           1           0
            0           1           0           1
    rhs: 1, 1, 1, 2

    Example 2
    x = 1, y = 3
    ax + by = rhs
    3x + 2y = 3x1 + 2x3 = 9
    4x + 5y = 4x1 + 5x3 = 19
    0x + 1y = 0x1 + 1x3 = 3
    1x + 0y = 1x1 + 0x3 = 1

    """

    examples_0 = np.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float32
    )
    labels_0 = np.array([1, 1, 1, 2], np.float32)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float32)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float32)
    labels_1 = np.array([9, 19, 3, 1], np.float32)
    expected_1 = np.array([1.0, 3.0], np.float32)

    np.testing.assert_array_almost_equal(
        math_ops._np_solve_least_squares(examples_0, labels_0),
        expected_0,
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        math_ops._np_solve_least_squares(examples_1, labels_1),
        expected_1,
        decimal=3,
    )
