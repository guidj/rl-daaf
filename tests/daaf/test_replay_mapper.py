from typing import Any, Union

import hypothesis
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import strategies as st
from rlplg import core

from daaf import replay_mapper
from tests import defaults


def test_identity_mapper_apply():
    mapper = replay_mapper.IdentifyMapper()

    inputs = [
        # single step traj
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=True,
            truncated=True,
        ),
        core.Trajectory(
            observation=defaults.array(2),
            action=defaults.array(4),
            policy_info={"log_probability": defaults.array(np.log(0.7))},
            reward=defaults.array(5.0),
            terminated=True,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(3),
            action=defaults.array(6),
            policy_info={"log_probability": defaults.array(np.log(0.2))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=True,
        ),
    ]

    expectations = [
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=True,
            truncated=True,
        ),
        core.Trajectory(
            observation=defaults.array(2),
            action=defaults.array(4),
            policy_info={"log_probability": defaults.array(np.log(0.7))},
            reward=defaults.array(5.0),
            terminated=True,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(3),
            action=defaults.array(6),
            policy_info={"log_probability": defaults.array(np.log(0.2))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=True,
        ),
    ]

    outputs = list(mapper.apply(inputs))

    for output, expected in zip(outputs, expectations):
        assert_trajectory(output=output, expected=expected)


@hypothesis.given(reward_period=st.integers(min_value=2))
def test_average_reward_mapper_init(reward_period: int):
    mapper = replay_mapper.AverageRewardMapper(reward_period=reward_period)
    assert mapper.reward_period == reward_period


@hypothesis.given(reward_period=st.integers(max_value=0))
def test_average_reward_mapper_init_with_invalid_reward_period(reward_period: int):
    with pytest.raises(ValueError):
        replay_mapper.AverageRewardMapper(reward_period=reward_period)


def test_average_reward_mapper_apply():
    """
    Each step is unpacked into its own Trajectory object.
    The reward is divided equally.
    Everything else is the same.
    """
    mapper = replay_mapper.AverageRewardMapper(reward_period=2)

    inputs = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(1),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=False,
        ),
    ]

    expectactions = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-4.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(1),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-4.0),
            terminated=False,
            truncated=False,
        ),
    ]

    outputs = list(mapper.apply(inputs))

    assert len(outputs) == 2
    for output, expected in zip(outputs, expectactions):
        assert_trajectory(output=output, expected=expected)


@hypothesis.given(
    reward_period=st.integers(min_value=1),
    impute_value=st.floats(allow_nan=False, allow_infinity=False),
)
def test_impute_missing_reward_mapper_init(reward_period: int, impute_value: float):
    mapper = replay_mapper.ImputeMissingRewardMapper(
        reward_period=reward_period, impute_value=impute_value
    )
    assert mapper.reward_period == reward_period
    assert mapper.impute_value == impute_value


@hypothesis.given(reward_period=st.integers(max_value=0))
def test_impute_missing_reward_mapper_init_with_invalid_reward_period(
    reward_period: int,
):
    with pytest.raises(ValueError):
        replay_mapper.ImputeMissingRewardMapper(
            reward_period=reward_period, impute_value=0.0
        )


def test_impute_missing_reward_mapper_init_with_invalid_impute_value():
    with pytest.raises(ValueError):
        replay_mapper.ImputeMissingRewardMapper(reward_period=1, impute_value=np.nan)

    with pytest.raises(ValueError):
        replay_mapper.ImputeMissingRewardMapper(reward_period=1, impute_value=np.inf)


def test_impute_missing_reward_mapper_apply():
    mapper = replay_mapper.ImputeMissingRewardMapper(reward_period=2, impute_value=0.0)

    inputs = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(1),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=False,
        ),
    ]

    expectactions = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(0.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(1),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-8.0),
            terminated=False,
            truncated=False,
        ),
    ]

    outputs = list(mapper.apply(inputs))

    assert len(outputs) == 2
    for output, expected in zip(outputs, expectactions):
        assert_trajectory(output=output, expected=expected)


def test_cumulative_reward_mapper_init():
    mapper = replay_mapper.SkipMissingRewardMapper(reward_period=2)
    assert mapper.reward_period == 2
    assert len(mapper._event_buffer) == 0


def test_skip_missing_reward_mapper_apply():
    mapper = replay_mapper.SkipMissingRewardMapper(reward_period=2)

    inputs = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-7.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(2),
            action=defaults.array(4),
            policy_info={"log_probability": defaults.array(np.log(0.7))},
            reward=defaults.array(5.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(3),
            action=defaults.array(6),
            policy_info={"log_probability": defaults.array(np.log(0.2))},
            reward=defaults.array(7.0),
            terminated=True,
            truncated=False,
        ),
    ]

    expectations = [
        core.Trajectory(
            observation=defaults.array(0),
            action=defaults.array(0),
            policy_info={"log_probability": defaults.array(np.log(0.3))},
            reward=defaults.array(-1.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(1),
            action=defaults.array(2),
            policy_info={"log_probability": defaults.array(np.log(0.8))},
            reward=defaults.array(-8.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(2),
            action=defaults.array(4),
            policy_info={"log_probability": defaults.array(np.log(0.7))},
            reward=defaults.array(5.0),
            terminated=False,
            truncated=False,
        ),
        core.Trajectory(
            observation=defaults.array(3),
            action=defaults.array(6),
            policy_info={"log_probability": defaults.array(np.log(0.2))},
            reward=defaults.array(12.0),
            terminated=True,
            truncated=False,
        ),
    ]

    outputs = list(mapper.apply(inputs))

    assert len(outputs) == 4
    for output, expected in zip(outputs, expectations):
        assert_trajectory(output=output, expected=expected)


def test_least_squares_attribution_mapper_init():
    rtable = [[0, 1], [0, 1], [0, 1], [0, 1]]
    mapper = replay_mapper.LeastSquaresAttributionMapper(
        num_states=4,
        num_actions=2,
        reward_period=2,
        state_id_fn=item,
        action_id_fn=item,
        buffer_size=8,
        init_rtable=defaults.array([0, 1], [0, 1], [0, 1], [0, 1]),
    )

    assert mapper.num_states == 4
    assert mapper.num_actions == 2
    assert mapper.reward_period == 2
    assert mapper.buffer_size == 8
    np.testing.assert_array_equal(mapper.rtable, rtable)


def test_least_squares_attribution_mapper_init_with_mismatched_table():
    with pytest.raises(ValueError):
        replay_mapper.LeastSquaresAttributionMapper(
            num_states=4,
            num_actions=2,
            reward_period=2,
            state_id_fn=item,
            action_id_fn=item,
            buffer_size=8,
            init_rtable=defaults.array([0.0, 1.0]),
        )


def test_least_squares_attribution_mapper_init_with_small_buffer_size():
    with pytest.raises(ValueError):
        replay_mapper.LeastSquaresAttributionMapper(
            num_states=4,
            num_actions=2,
            reward_period=2,
            state_id_fn=item,
            action_id_fn=item,
            buffer_size=7,
            init_rtable=defaults.array([0, 1], [0, 1], [0, 1], [0, 1]),
        )


def test_least_squares_attribution_mapper_apply():
    """
    Initial events will have reward values from rtable.
    Once there are enough samples, Least Square Estimates are used instead.
    The estimates are updated at `update_steps` intervals.

    Problem: Two states (A, B), two actions (left, right)
    Table:
            Actions
    States  Left    Right
        A   0       1
        B   0       1

    events: (A, left, A, right) -> (0, 0), (0, 1) -> 0 + 1 = 1
            (B, left, B, right) -> (1, 0), (1, 1) -> 0 + 1 = 1
            (A, right, B, left) -> (0, 1), (1, 0) -> 1 + 0 = 1
            (A, right, B, right)-> (0, 1), (1, 1) -> 1 + 1 = 2

    matrix: (A, left), (A, right), (B, left), (B, right)
            1           1           0           0
            0           0           1           1
            0           1           1           0
            0           1           0           1
    rhs: 1, 1, 1, 2
    """

    def ctraj(states, actions, rewards, probs):
        trajs = []
        for state, action, reward, prob in zip(states, actions, rewards, probs):
            trajs.append(
                core.Trajectory(
                    observation=defaults.array(state),
                    action=defaults.array(action),
                    policy_info={"log_probability": defaults.array(np.log(prob))},
                    reward=defaults.array(reward),
                    terminated=False,
                    truncated=False,
                )
            )
        return trajs

    mapper = replay_mapper.LeastSquaresAttributionMapper(
        num_states=2,
        num_actions=2,
        reward_period=2,
        state_id_fn=item,
        action_id_fn=item,
        buffer_size=8,
        init_rtable=defaults.array([-1.0, -1.0], [-1.0, -1.0]),
    )

    # We are simulating cumulative rewards.
    # So we supply the actual rewards to the simulator to aggregate (sum).
    inputs = []
    inputs.extend(
        ctraj(states=(0, 0), actions=(0, 1), rewards=(0.0, 1.0), probs=(0.0, 1.0))
    )
    inputs.extend(
        ctraj(states=(1, 1), actions=(0, 1), rewards=(0.0, 1.0), probs=(0.0, 1.0))
    )
    # after the event above, all factors are present, but we still lack rows
    # to satisfy the condition m >= n
    inputs.extend(
        ctraj(states=(0, 1), actions=(1, 0), rewards=(1.0, 0.0), probs=(1.0, 0.0))
    )
    inputs.extend(
        ctraj(states=(0, 1), actions=(1, 1), rewards=(1.0, 1.0), probs=(1.0, 1.0))
    )
    # after the event above, m >= n
    # the events will below will be emitted with estimated rewards
    inputs.extend(
        ctraj(states=(0, 0), actions=(0, 1), rewards=(-7.0, -7.0), probs=(0.0, 1.0))
    )
    inputs.extend(
        ctraj(states=(1, 1), actions=(0, 1), rewards=(-7.0, -7.0), probs=(0.0, 1.0))
    )
    inputs.extend(
        ctraj(states=(0, 1), actions=(1, 0), rewards=(-7.0, -7.0), probs=(1.0, 0.0))
    )
    inputs.extend(
        ctraj(states=(0, 1), actions=(1, 1), rewards=(-7.0, -7.0), probs=(1.0, 1.0))
    )

    expectactions = []
    # the events below are emitted with the initial beliefs about rewards
    expectactions.extend(
        ctraj(states=(0,), actions=(0,), rewards=(-1.0,), probs=(0.0,))
    )
    expectactions.extend(
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,))
    )
    expectactions.extend(
        ctraj(states=(1,), actions=(0,), rewards=(-1.0,), probs=(0.0,))
    )
    expectactions.extend(
        ctraj(states=(1,), actions=(1,), rewards=(-1.0,), probs=(1.0,))
    )
    expectactions.extend(
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,))
    )
    expectactions.extend(
        ctraj(states=(1,), actions=(0,), rewards=(-1.0,), probs=(0.0,))
    )
    expectactions.extend(
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,))
    )
    expectactions.extend(
        ctraj(states=(1,), actions=(1,), rewards=(-1.0,), probs=(1.0,))
    )
    # the events below are emitted with estimated rewards
    expectactions.extend(ctraj(states=(0,), actions=(0,), rewards=(0.0,), probs=(0.0,)))
    expectactions.extend(ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)))
    expectactions.extend(ctraj(states=(1,), actions=(0,), rewards=(0.0,), probs=(0.0,)))
    expectactions.extend(ctraj(states=(1,), actions=(1,), rewards=(1.0,), probs=(1.0,)))
    expectactions.extend(ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)))
    expectactions.extend(ctraj(states=(1,), actions=(0,), rewards=(0.0,), probs=(0.0,)))
    expectactions.extend(ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)))
    expectactions.extend(ctraj(states=(1,), actions=(1,), rewards=(1.0,), probs=(1.0,)))

    outputs = list(mapper.apply(inputs))

    assert len(outputs) == 16
    for output, expected in zip(outputs, expectactions):

        # reward can only be approximately equal
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.policy_info, expected.policy_info)
        np.testing.assert_array_almost_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.terminated, expected.terminated)
        np.testing.assert_array_equal(output.truncated, expected.truncated)


def test_counter_init():
    counter = replay_mapper.Counter()
    assert counter.value is None


def test_counter_inc():
    counter = replay_mapper.Counter()
    counter.reset()
    counter.inc()
    assert counter.value == 1

    counter.inc()
    assert counter.value == 2


def test_counter_reset():
    counter = replay_mapper.Counter()
    assert counter.value is None

    counter.reset()
    assert counter.value == 0


def item(array: Union[np.ndarray, tf.Tensor]) -> Any:
    if isinstance(array, tf.Tensor):
        return array.numpy().item()
    return array.item()


def assert_trajectory(output: core.Trajectory, expected: core.Trajectory) -> None:
    np.testing.assert_array_equal(output.observation, expected.observation)
    np.testing.assert_array_equal(output.action, expected.action)
    np.testing.assert_array_equal(output.policy_info, expected.policy_info)
    np.testing.assert_array_equal(output.reward, expected.reward)
    np.testing.assert_array_equal(output.terminated, expected.terminated)
    np.testing.assert_array_equal(output.truncated, expected.truncated)
