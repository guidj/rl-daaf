from typing import Any, Mapping, Optional, Union

import hypothesis
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import strategies as st
from rlplg import core

from daaf import replay_mapper
from tests import defaults


def test_identity_mapper_apply():
    mapper = replay_mapper.IdentityMapper()

    inputs = [
        # single step traj
        traj_step(state=1, action=2, reward=-7.0, prob=0.8),
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(
            state=1, action=2, reward=-7.0, prob=0.8, terminated=True, truncated=True
        ),
        traj_step(state=2, action=4, reward=5.0, prob=0.7, terminated=True),
        traj_step(state=3, action=6, reward=-7.0, prob=0.2, truncated=True),
    ]

    expectations = [
        traj_step(state=1, action=2, reward=-7.0, prob=0.8),
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(
            state=1, action=2, reward=-7.0, prob=0.8, terminated=True, truncated=True
        ),
        traj_step(state=2, action=4, reward=5.0, prob=0.7, terminated=True),
        traj_step(state=3, action=6, reward=-7.0, prob=0.2, truncated=True),
    ]

    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 5
    for output, expected in zip(outputs, expectations):
        assert_trajectory(output=output, expected=expected)


@hypothesis.given(
    reward_period=st.integers(min_value=1),
    impute_value=st.floats(allow_nan=False, allow_infinity=False),
)
def test_daaf_impute_missing_reward_mapper_init(
    reward_period: int, impute_value: float
):
    mapper = replay_mapper.DaafImputeMissingRewardMapper(
        reward_period=reward_period, impute_value=impute_value
    )
    assert mapper.reward_period == reward_period
    assert mapper.impute_value == impute_value


@hypothesis.given(reward_period=st.integers(max_value=0))
def test_daaf_impute_missing_reward_mapper_init_with_invalid_reward_period(
    reward_period: int,
):
    with pytest.raises(ValueError):
        replay_mapper.DaafImputeMissingRewardMapper(
            reward_period=reward_period, impute_value=0.0
        )


def test_daaf_impute_missing_reward_mapper_init_with_invalid_impute_value():
    with pytest.raises(ValueError):
        replay_mapper.DaafImputeMissingRewardMapper(
            reward_period=1, impute_value=np.nan
        )

    with pytest.raises(ValueError):
        replay_mapper.DaafImputeMissingRewardMapper(
            reward_period=1, impute_value=np.inf
        )


def test_daaf_impute_missing_reward_mapper_apply():
    mapper = replay_mapper.DaafImputeMissingRewardMapper(
        reward_period=2, impute_value=0.0
    )

    inputs = [
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(state=1, action=1, reward=-7.0, prob=0.8),
        traj_step(
            state=0,
            action=0,
            reward=3.0,
            prob=0.3,
            info={"prior": "entry"},
        ),
        traj_step(state=1, action=1, reward=6.0, prob=0.8, truncated=True),
        traj_step(state=1, action=1, reward=11.0, prob=0.9, terminated=True),
    ]

    expectactions = [
        traj_step(state=0, action=0, reward=0.0, prob=0.3, info={"imputed": True}),
        traj_step(state=1, action=1, reward=-8.0, prob=0.8, info={"imputed": False}),
        traj_step(
            state=0,
            action=0,
            reward=0.0,
            prob=0.3,
            info={"imputed": True, "prior": "entry"},
        ),
        traj_step(
            state=1,
            action=1,
            reward=9.0,
            prob=0.8,
            truncated=True,
            info={"imputed": False},
        ),
        traj_step(
            state=1,
            action=1,
            reward=0.0,
            prob=0.9,
            terminated=True,
            info={"imputed": True},
        ),
    ]

    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 5
    for output, expected in zip(outputs, expectactions):
        assert_trajectory(output=output, expected=expected)


def test_daaf_lsq_reward_attribution_mapper_init():
    rtable = [[0, 1], [0, 1], [0, 1], [0, 1]]
    mapper = replay_mapper.DaafLsqRewardAttributionMapper(
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


def test_daaf_lsq_reward_attribution_mapper_init_with_mismatched_table():
    with pytest.raises(ValueError):
        replay_mapper.DaafLsqRewardAttributionMapper(
            num_states=4,
            num_actions=2,
            reward_period=2,
            state_id_fn=item,
            action_id_fn=item,
            buffer_size=8,
            init_rtable=defaults.array([0.0, 1.0]),
        )


def test_daaf_lsq_reward_attribution_mapper_init_with_small_buffer_size():
    with pytest.raises(ValueError):
        replay_mapper.DaafLsqRewardAttributionMapper(
            num_states=4,
            num_actions=2,
            reward_period=2,
            state_id_fn=item,
            action_id_fn=item,
            buffer_size=7,
            init_rtable=defaults.array([0, 1], [0, 1], [0, 1], [0, 1]),
        )


def test_daaf_lsq_reward_attribution_mapper_apply():
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

    mapper = replay_mapper.DaafLsqRewardAttributionMapper(
        num_states=2,
        num_actions=2,
        reward_period=2,
        state_id_fn=item,
        action_id_fn=item,
        buffer_size=8,
        init_rtable=defaults.array([-1.0, -1.0], [-1.0, -1.0]),
        impute_value=88,
    )

    # We are simulating cumulative rewards.
    # So we supply the actual rewards to the simulator to aggregate (sum).
    inputs = [
        traj_step(state=0, action=0, reward=0.0, prob=0.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=0, reward=0.0, prob=0.0),
        traj_step(state=1, action=1, reward=1.0, prob=1.0),
        # after the event above, all factors are present, but we still lack rows
        # to satisfy the condition m >= n
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=0, reward=0.0, prob=0.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=1, reward=1.0, prob=1.0),
        # after the event above, m >= n
        # the events will below will be emitted with estimated rewards
        traj_step(state=0, action=0, reward=-7.0, prob=0.0),
        traj_step(state=0, action=1, reward=-7.0, prob=1.0),
        traj_step(state=1, action=0, reward=-7.0, prob=0.0),
        traj_step(state=1, action=1, reward=-7.0, prob=1.0),
        traj_step(state=0, action=1, reward=-7.0, prob=1.0),
        traj_step(state=1, action=0, reward=-7.0, prob=0.0),
        traj_step(state=0, action=1, reward=-7.0, prob=1.0),
        traj_step(state=1, action=1, reward=-7.0, prob=1.0),
    ]
    expectactions = [
        # the events below are emitted with the impute value
        # or the aggregate feedback
        traj_step(state=0, action=0, reward=88, prob=0.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=0, reward=88, prob=0.0),
        traj_step(state=1, action=1, reward=1.0, prob=1.0),
        traj_step(state=0, action=1, reward=88, prob=1.0),
        traj_step(state=1, action=0, reward=1.0, prob=0.0),
        traj_step(state=0, action=1, reward=88, prob=1.0),
        traj_step(state=1, action=1, reward=2.0, prob=1.0),
        # the events below are emitted with estimated rewards
        traj_step(state=0, action=0, reward=0.0, prob=0.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=0, reward=0.0, prob=0.0),
        traj_step(state=1, action=1, reward=1.0, prob=1.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=0, reward=0.0, prob=0.0),
        traj_step(state=0, action=1, reward=1.0, prob=1.0),
        traj_step(state=1, action=1, reward=1.0, prob=1.0),
    ]

    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 16
    for output, expected in zip(outputs, expectactions):
        # reward can only be approximately equal
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.policy_info, expected.policy_info)
        np.testing.assert_array_almost_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.terminated, expected.terminated)
        np.testing.assert_array_equal(output.truncated, expected.truncated)


def test_daaf_mdp_with_options_mapper_apply_given_truncated_options():
    mapper = replay_mapper.DaafMdpWithOptionsMapper()
    inputs = [
        # three step option
        traj_step(
            state=1,
            action=0,
            reward=2.0,
            policy_info={"option_id": 7, "option_terminated": False},
        ),
        traj_step(
            state=2,
            action=1,
            reward=4.0,
            policy_info={"option_id": 7, "option_terminated": False},
        ),
        traj_step(
            state=3,
            action=2,
            reward=6.0,
            policy_info={"option_id": 7, "option_terminated": True},
        ),
        # two step option
        traj_step(
            state=4,
            action=1,
            reward=1.0,
            policy_info={"option_id": 4, "option_terminated": False},
        ),
        traj_step(
            state=5,
            action=3,
            reward=1.0,
            policy_info={"option_id": 4, "option_terminated": True},
        ),
        # single action option
        traj_step(
            state=6,
            action=8,
            reward=1.0,
            policy_info={"option_id": 0, "option_terminated": True},
        ),
        # unfinished option - omitted from trajectory
        traj_step(
            state=7,
            action=8,
            reward=1.0,
            policy_info={"option_id": 3, "option_terminated": False},
        ),
    ]

    output = mapper.apply(inputs)
    expectactions = [
        traj_step(
            state=1,
            action=7,
            reward=12.0,
        ),
        traj_step(
            state=4,
            action=4,
            reward=2.0,
        ),
        traj_step(state=6, action=0, reward=1.0),
        traj_step(
            state=7,
            action=3,
            reward=0.0,
            truncated=True,
        ),
    ]
    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 4
    for output, expected in zip(outputs, expectactions):
        # reward can only be approximately equal
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.policy_info, expected.policy_info)
        np.testing.assert_array_almost_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.terminated, expected.terminated)
        np.testing.assert_array_equal(output.truncated, expected.truncated)


def test_daaf_mdp_with_options_mapper_apply_given_terminating_option():
    mapper = replay_mapper.DaafMdpWithOptionsMapper()
    inputs = [
        # three step option
        traj_step(
            state=1,
            action=0,
            reward=2.0,
            policy_info={"option_id": 7, "option_terminated": False},
        ),
        traj_step(
            state=2,
            action=1,
            reward=4.0,
            policy_info={"option_id": 7, "option_terminated": False},
        ),
        traj_step(
            state=3,
            action=2,
            reward=6.0,
            policy_info={"option_id": 7, "option_terminated": True},
        ),
        # two step option
        traj_step(
            state=4,
            action=1,
            reward=1.0,
            policy_info={"option_id": 4, "option_terminated": False},
        ),
        traj_step(
            state=5,
            action=3,
            reward=1.0,
            policy_info={"option_id": 4, "option_terminated": True},
        ),
    ]

    output = mapper.apply(inputs)
    expectactions = [
        traj_step(
            state=1,
            action=7,
            reward=12.0,
        ),
        traj_step(
            state=4,
            action=4,
            reward=2.0,
        ),
    ]
    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 2
    for output, expected in zip(outputs, expectactions):
        # reward can only be approximately equal
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.policy_info, expected.policy_info)
        np.testing.assert_array_almost_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.terminated, expected.terminated)
        np.testing.assert_array_equal(output.truncated, expected.truncated)


def test_daaf_nstep_td_update_mark_mapper_init():
    mapper = replay_mapper.DaafNStepTdUpdateMarkMapper(
        reward_period=2, impute_value=0.0
    )
    assert mapper.reward_period == 2
    assert mapper.nstep == 2
    assert mapper.impute_value == 0.0


def test_daaf_nstep_td_update_mark_mapper_apply():
    mapper = replay_mapper.DaafNStepTdUpdateMarkMapper(
        reward_period=2, impute_value=0.0
    )

    inputs = [
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(state=1, action=1, reward=-7.0, prob=0.8),
        traj_step(
            state=0,
            action=0,
            reward=3.0,
            prob=0.3,
            info={"prior": "entry"},
        ),
        traj_step(state=1, action=1, reward=6.0, prob=0.8, truncated=True),
        traj_step(state=1, action=1, reward=11.0, prob=0.9, terminated=True),
    ]

    expectactions = [
        traj_step(
            state=0,
            action=0,
            reward=0.0,
            prob=0.3,
            info={"imputed": True, "ok_nstep_tau": True},
        ),
        traj_step(
            state=1,
            action=1,
            reward=-8.0,
            prob=0.8,
            info={"imputed": False, "ok_nstep_tau": False},
        ),
        traj_step(
            state=0,
            action=0,
            reward=0.0,
            prob=0.3,
            info={"prior": "entry", "imputed": True, "ok_nstep_tau": True},
        ),
        traj_step(
            state=1,
            action=1,
            reward=9.0,
            prob=0.8,
            truncated=True,
            info={"imputed": False, "ok_nstep_tau": False},
        ),
        traj_step(
            state=1,
            action=1,
            reward=0.0,
            prob=0.9,
            terminated=True,
            info={"imputed": True, "ok_nstep_tau": False},
        ),
    ]

    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 5
    for output, expected in zip(outputs, expectactions):
        assert_trajectory(output=output, expected=expected)


def test_daaf_drop_episode_with_truncated_feedback_mapper_init():
    mapper = replay_mapper.DaafDropEpisodeWithTruncatedFeedbackMapper(reward_period=2)
    assert mapper.reward_period == 2


def test_daaf_drop_episode_with_truncated_feedback_mapper_apply():
    mapper = replay_mapper.DaafDropEpisodeWithTruncatedFeedbackMapper(reward_period=2)

    inputs = [
        # single step traj
        traj_step(state=1, action=2, reward=-7.0, prob=0.8),
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(
            state=1, action=2, reward=-7.0, prob=0.8, terminated=True, truncated=True
        ),
        traj_step(state=2, action=4, reward=5.0, prob=0.7, terminated=True),
        traj_step(state=3, action=6, reward=-7.0, prob=0.2, truncated=True),
    ]

    expectations = [
        traj_step(state=1, action=2, reward=-7.0, prob=0.8),
        traj_step(state=0, action=0, reward=-1.0, prob=0.3),
        traj_step(
            state=1, action=2, reward=-7.0, prob=0.8, terminated=True, truncated=True
        ),
        traj_step(state=2, action=4, reward=5.0, prob=0.7, terminated=True),
    ]

    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 0
    outputs = tuple(mapper.apply(inputs[:4]))
    for output, expected in zip(outputs, expectations):
        assert_trajectory(output=output, expected=expected)


def test_collect_returns_mapper_apply():
    mapper = replay_mapper.CollectReturnsMapper()

    inputs = [
        traj_step(
            state=1,
            action=0,
            reward=-7.0,
        ),
        traj_step(
            state=2,
            action=1,
            reward=-1.0,
        ),
        traj_step(state=3, action=0, reward=-7.0, terminated=True, truncated=True),
        traj_step(state=4, action=1, reward=5.0, terminated=True),
        traj_step(state=5, action=0, reward=-7.0, truncated=True),
    ]

    assert len(mapper.traj_returns) == 0
    outputs = tuple(mapper.apply(inputs))
    assert len(outputs) == 5
    for output, expected in zip(outputs, inputs):
        assert_trajectory(output=output, expected=expected)
    assert mapper.traj_returns == [-17.0]

    # second pass, first three steps
    outputs = tuple(mapper.apply(inputs[:3]))
    assert len(outputs) == 3
    for output, expected in zip(outputs, inputs[:3]):
        assert_trajectory(output=output, expected=expected)

    assert mapper.traj_returns == [-17.0, -15.0]


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


def traj_step(
    state: int,
    action: int,
    reward: float,
    prob: Optional[float] = None,
    terminated: bool = False,
    truncated: bool = False,
    policy_info: Optional[Mapping[str, Any]] = None,
    info: Optional[Mapping[str, Any]] = None,
):
    prob_info = {"log_probability": defaults.array(np.log(prob))} if prob else {}
    return core.TrajectoryStep(
        observation=defaults.array(state),
        action=defaults.array(action),
        policy_info={**policy_info, **prob_info} if policy_info else prob_info,
        reward=defaults.array(reward),
        terminated=terminated,
        truncated=truncated,
        info=info if info else {},
    )


def assert_trajectory(
    output: core.TrajectoryStep, expected: core.TrajectoryStep
) -> None:
    np.testing.assert_array_equal(output.observation, expected.observation)
    np.testing.assert_array_equal(output.action, expected.action)
    np.testing.assert_array_equal(output.policy_info, expected.policy_info)
    np.testing.assert_array_equal(output.reward, expected.reward)
    np.testing.assert_array_equal(output.terminated, expected.terminated)
    np.testing.assert_array_equal(output.truncated, expected.truncated)
    np.testing.assert_array_equal(output.info, expected.info)
