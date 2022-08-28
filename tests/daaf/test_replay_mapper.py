import hypothesis
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import strategies as st
from tf_agents.trajectories import policy_step, time_step, trajectory
from tf_agents.typing.types import TensorOrArray

from daaf import replay_mapper
from tests import defaults


def test_identity_mapper_apply():
    mapper = replay_mapper.IdentifyMapper()

    inputs = [
        # single step traj
        trajectory.Trajectory(
            step_type=defaults.batch(
                time_step.StepType.MID,
            ),
            observation=defaults.batch(1),
            action=defaults.batch(2),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.8),
                )
            ),
            next_step_type=defaults.batch(
                time_step.StepType.MID,
            ),
            reward=defaults.batch(-7.0),
            discount=defaults.batch(1.0),
        ),
        # multi step traj
        trajectory.Trajectory(
            step_type=defaults.batch(
                time_step.StepType.FIRST,
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.MID,
            ),
            observation=defaults.batch(0, 1, 2, 3),
            action=defaults.batch(0, 2, 4, 6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                    np.log(0.8),
                    np.log(0.7),
                    np.log(0.2),
                )
            ),
            next_step_type=defaults.batch(
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.LAST,
            ),
            reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
            discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
        ),
    ]

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(
                time_step.StepType.MID,
            ),
            observation=defaults.batch(1),
            action=defaults.batch(2),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.8),
                )
            ),
            next_step_type=defaults.batch(
                time_step.StepType.MID,
            ),
            reward=defaults.batch(-7.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(
                time_step.StepType.FIRST,
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.MID,
            ),
            observation=defaults.batch(0, 1, 2, 3),
            action=defaults.batch(0, 2, 4, 6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                    np.log(0.8),
                    np.log(0.7),
                    np.log(0.2),
                )
            ),
            next_step_type=defaults.batch(
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.MID,
                time_step.StepType.LAST,
            ),
            reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
            discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
        ),
    ]

    outputs = [list(mapper.apply(input)) for input in inputs]

    for output, expected in zip(outputs, expectations):
        assert len(output) == 1
        output = next(iter(output))
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


def test_single_action_mapper_apply():
    mapper = replay_mapper.SingleStepMapper()

    input = trajectory.Trajectory(
        step_type=defaults.batch(
            time_step.StepType.FIRST,
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.MID,
        ),
        observation=defaults.batch(0, 1, 2, 3),
        action=defaults.batch(0, 2, 4, 6),
        policy_info=policy_step.PolicyInfo(
            log_probability=defaults.batch(
                np.log(0.3),
                np.log(0.8),
                np.log(0.7),
                np.log(0.2),
            )
        ),
        next_step_type=defaults.batch(
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.LAST,
        ),
        reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
        discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
    )

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.3))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-1.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(1),
            action=defaults.batch(2),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.8))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-7.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(2),
            action=defaults.batch(4),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.7))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(5.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(3),
            action=defaults.batch(6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.2))
            ),
            next_step_type=defaults.batch(time_step.StepType.LAST),
            reward=defaults.batch(7.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = list(mapper.apply(input))

    assert len(outputs) == 4
    for output, expected in zip(outputs, expectations):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


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

    input = trajectory.Trajectory(
        step_type=defaults.batch(time_step.StepType.FIRST, time_step.StepType.MID),
        observation=defaults.batch(0, 1),
        action=defaults.batch(0, 1),
        policy_info=policy_step.PolicyInfo(
            log_probability=defaults.batch(
                np.log(0.3),
                np.log(0.8),
            )
        ),
        next_step_type=defaults.batch(time_step.StepType.MID, time_step.StepType.MID),
        reward=defaults.batch(-1.0, -7.0),
        discount=defaults.batch(1.0, 1.0),
    )

    expectactions = [
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                )
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-4.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(1),
            action=defaults.batch(1),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.8),
                )
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-4.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = list(mapper.apply(input))

    assert len(outputs) == 2
    for output, expected in zip(outputs, expectactions):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


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

    input = trajectory.Trajectory(
        step_type=defaults.batch(time_step.StepType.FIRST, time_step.StepType.MID),
        observation=defaults.batch(0, 1),
        action=defaults.batch(0, 1),
        policy_info=policy_step.PolicyInfo(
            log_probability=defaults.batch(
                np.log(0.3),
                np.log(0.8),
            )
        ),
        next_step_type=defaults.batch(time_step.StepType.MID, time_step.StepType.MID),
        reward=defaults.batch(-1.0, -7.0),
        discount=defaults.batch(1.0, 1.0),
    )

    expectactions = [
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                )
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(0.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(1),
            action=defaults.batch(1),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.8),
                )
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-8.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = list(mapper.apply(input))

    assert len(outputs) == 2
    for output, expected in zip(outputs, expectactions):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


def test_cumulative_reward_mapper_init():
    mapper = replay_mapper.CumulativeRewardMapper(reward_period=2)
    assert mapper.reward_period == 2
    assert len(mapper._event_buffer) == 0


def test_cumulative_reward_mapper_apply():
    mapper = replay_mapper.CumulativeRewardMapper(reward_period=2)

    input = trajectory.Trajectory(
        step_type=defaults.batch(
            time_step.StepType.FIRST,
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.MID,
        ),
        observation=defaults.batch(0, 1, 2, 3),
        action=defaults.batch(0, 2, 4, 6),
        policy_info=policy_step.PolicyInfo(
            log_probability=defaults.batch(
                np.log(0.3),
                np.log(0.8),
                np.log(0.7),
                np.log(0.2),
            )
        ),
        next_step_type=defaults.batch(
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.MID,
            time_step.StepType.LAST,
        ),
        reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
        discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
    )

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.3))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-1.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(1),
            action=defaults.batch(2),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.8))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(-8.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(2),
            action=defaults.batch(4),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.7))
            ),
            next_step_type=defaults.batch(time_step.StepType.MID),
            reward=defaults.batch(5.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(time_step.StepType.MID),
            observation=defaults.batch(3),
            action=defaults.batch(6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.2))
            ),
            next_step_type=defaults.batch(time_step.StepType.LAST),
            reward=defaults.batch(12.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = list(mapper.apply(input))

    assert len(outputs) == 4
    for output, expected in zip(outputs, expectations):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


def test_least_squares_attribution_mapper_init():
    rtable = [[0, 1], [0, 1], [0, 1], [0, 1]]
    mapper = replay_mapper.LeastSquaresAttributionMapper(
        num_states=4,
        num_actions=2,
        reward_period=2,
        state_id_fn=item,
        action_id_fn=item,
        buffer_size=8,
        init_rtable=defaults.batch([0, 1], [0, 1], [0, 1], [0, 1]),
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
            init_rtable=defaults.batch([0.0, 1.0]),
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
            init_rtable=defaults.batch([0, 1], [0, 1], [0, 1], [0, 1]),
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
        return trajectory.Trajectory(
            step_type=defaults.batch(*[time_step.StepType.MID for _ in actions]),
            observation=defaults.batch(*states),
            action=defaults.batch(*actions),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(*[np.log(prob) for prob in probs])
            ),
            next_step_type=defaults.batch(*[time_step.StepType.MID for _ in actions]),
            reward=defaults.batch(*rewards),
            discount=defaults.batch(*[1.0 for _ in actions]),
        )

    mapper = replay_mapper.LeastSquaresAttributionMapper(
        num_states=2,
        num_actions=2,
        reward_period=2,
        state_id_fn=item,
        action_id_fn=item,
        buffer_size=8,
        init_rtable=defaults.batch([-1.0, -1.0], [-1.0, -1.0]),
    )

    # We are simulating cumulative rewards.
    # So we supply the actual rewards to the simulator to aggregate (sum).
    inputs = [
        ctraj(states=(0, 0), actions=(0, 1), rewards=(0.0, 1.0), probs=(0.0, 1.0)),
        ctraj(states=(1, 1), actions=(0, 1), rewards=(0.0, 1.0), probs=(0.0, 1.0)),
        # after the event above, all factors are present, but we still lack rows
        # to satisfy the condition m >= n
        ctraj(states=(0, 1), actions=(1, 0), rewards=(1.0, 0.0), probs=(1.0, 0.0)),
        ctraj(states=(0, 1), actions=(1, 1), rewards=(1.0, 1.0), probs=(1.0, 1.0)),
        # after the event above, m >= n
        # the events will below will be emitted with estimated rewards
        ctraj(states=(0, 0), actions=(0, 1), rewards=(-7.0, -7.0), probs=(0.0, 1.0)),
        ctraj(states=(1, 1), actions=(0, 1), rewards=(-7.0, -7.0), probs=(0.0, 1.0)),
        ctraj(states=(0, 1), actions=(1, 0), rewards=(-7.0, -7.0), probs=(1.0, 0.0)),
        ctraj(states=(0, 1), actions=(1, 1), rewards=(-7.0, -7.0), probs=(1.0, 1.0)),
    ]

    expectactions = [
        # the events below are emitted with the initial beliefs about rewards
        ctraj(states=(0,), actions=(0,), rewards=(-1.0,), probs=(0.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(0,), rewards=(-1.0,), probs=(0.0,)),
        ctraj(states=(1,), actions=(1,), rewards=(-1.0,), probs=(1.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(0,), rewards=(-1.0,), probs=(0.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(-1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(1,), rewards=(-1.0,), probs=(1.0,)),
        # the events below are emitted with estimated rewards
        ctraj(states=(0,), actions=(0,), rewards=(0.0,), probs=(0.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(0,), rewards=(0.0,), probs=(0.0,)),
        ctraj(states=(1,), actions=(1,), rewards=(1.0,), probs=(1.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(0,), rewards=(0.0,), probs=(0.0,)),
        ctraj(states=(0,), actions=(1,), rewards=(1.0,), probs=(1.0,)),
        ctraj(states=(1,), actions=(1,), rewards=(1.0,), probs=(1.0,)),
    ]

    outputs = []
    for traj in inputs:
        outputs.extend(list(mapper.apply(traj)))

    assert len(outputs) == 16
    for output, expected in zip(outputs, expectactions):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_almost_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


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


def item(array: TensorOrArray) -> int:
    if isinstance(array, tf.Tensor):
        return array.numpy().item()
    return array.item()
