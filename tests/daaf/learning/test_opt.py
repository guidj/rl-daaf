import hypothesis
import numpy as np
from hypothesis import strategies as st

from daaf.learning import opt


def test_least_squares_factors_present():
    examples = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert opt.least_squares_factors_present(examples)


def test_least_squares_factors_present_with_missing_entries():
    missing_0 = np.array([[0, 1, 0], [0, 0, 1]])
    missing_1 = np.array([[1, 0, 0], [0, 0, 1]])
    missing_2 = np.array([[1, 0, 0], [0, 1, 0]])

    assert not opt.least_squares_factors_present(missing_0)
    assert not opt.least_squares_factors_present(missing_1)
    assert not opt.least_squares_factors_present(missing_2)


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
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float64
    )
    labels_0 = np.array([1, 1, 1, 2], np.float64)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float64)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float64)
    labels_1 = np.array([9, 19, 3, 1], np.float64)
    expected_1 = np.array([1.0, 3.0], np.float64)

    np.testing.assert_array_almost_equal(
        opt.solve_least_squares(examples_0, labels_0), expected_0, decimal=3
    )
    np.testing.assert_array_almost_equal(
        opt.solve_least_squares(examples_1, labels_1), expected_1, decimal=3
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
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float64
    )
    labels_0 = np.array([1, 1, 1, 2], np.float64)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float64)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float64)
    labels_1 = np.array([9, 19, 3, 1], np.float64)
    expected_1 = np.array([1.0, 3.0], np.float64)

    np.testing.assert_array_almost_equal(
        opt._tf_solve_least_squares(examples_0, labels_0),
        expected_0,
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        opt._tf_solve_least_squares(examples_1, labels_1),
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
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], np.float64
    )
    labels_0 = np.array([1, 1, 1, 2], np.float64)
    expected_0 = np.array([0.0, 1.0, 0.0, 1.0], np.float64)

    examples_1 = np.array([[3, 2], [4, 5], [0, 1], [1, 0]], np.float64)
    labels_1 = np.array([9, 19, 3, 1], np.float64)
    expected_1 = np.array([1.0, 3.0], np.float64)

    np.testing.assert_array_almost_equal(
        opt._np_solve_least_squares(examples_0, labels_0),
        expected_0,
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        opt._np_solve_least_squares(examples_1, labels_1),
        expected_1,
        decimal=3,
    )


@hypothesis.given(st.floats(allow_nan=False, allow_infinity=False))
def test_learning_rate_schedule_init(initial_learning_rate: float):
    def schedule(ilr: float, episode: int, step: int):
        del episode
        del step
        return ilr

    lrs = opt.LearningRateSchedule(
        initial_learning_rate=initial_learning_rate, schedule=schedule
    )

    assert lrs.initial_learning_rate == initial_learning_rate


@hypothesis.given(step=st.integers())
def test_learning_rate_schedule_call_with_episode_schedule(step: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del step
        return initial_learning_rate * (0.9**episode)

    assert (
        opt.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=1, step=step
        )
        == 0.9
    )
    # increase episode
    assert (
        opt.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=2, step=step
        )
        == 0.81
    )
    # change initial learning rate
    assert (
        opt.LearningRateSchedule(initial_learning_rate=100.0, schedule=schedule)(
            episode=2, step=step
        )
        == 81
    )


@hypothesis.given(episode=st.integers())
def test_learning_rate_schedule_call_with_step_schedule(episode: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del episode
        return initial_learning_rate * (0.9**step)

    assert (
        opt.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=episode, step=1
        )
        == 0.9
    )
    # increase step
    assert (
        opt.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=episode, step=2
        )
        == 0.81
    )
    # change initial learning rate
    assert (
        opt.LearningRateSchedule(initial_learning_rate=100.0, schedule=schedule)(
            episode=episode, step=2
        )
        == 81
    )


@hypothesis.given(step=st.integers())
def test_learning_rate_schedule_call_with_decaying_schedule(step: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del step
        if episode < 10:
            return initial_learning_rate
        return initial_learning_rate * 0.9

    lrs = opt.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)

    assert lrs(episode=1, step=step) == 1
    assert lrs(episode=9, step=step) == 1
    assert lrs(episode=10, step=step) == 0.9
    assert lrs(episode=21, step=step) == 0.9
