from typing import Sequence

import hypothesis
import numpy as np
from hypothesis import strategies as st
from scipy.spatial import distance

from daaf.learning.tabular import policies
from tests import defaults


def test_random_policy_init():
    num_actions = 4
    policy = create_random_policy(num_actions=num_actions)

    assert policy._num_actions == num_actions
    assert policy.emit_log_probability is False


def test_random_policy_init_with_emit_log_probability():
    num_actions = 4
    policy = create_random_policy(num_actions=num_actions, emit_log_probability=True)

    assert policy._num_actions == num_actions
    assert policy.emit_log_probability is True


@hypothesis.given(num_actions=st.integers(min_value=1, max_value=100))
def test_random_policy_action(num_actions: int):
    policy = create_random_policy(num_actions=num_actions, emit_log_probability=True)
    _policy_step = policy.action(observation={})
    np.testing.assert_allclose(
        _policy_step.info["log_probability"], np.log(1.0 / num_actions)
    )


def test_random_policy_validity():
    num_actions = 4
    policy = create_random_policy(num_actions=num_actions)

    expected = np.array([1.0 / num_actions] * num_actions, np.float32)
    output = np.zeros(shape=(num_actions), dtype=np.int64)
    for _ in range(10_000):
        _policy_step = policy.action(observation={})
        output[_policy_step.action] += 1
    output = output / np.sum(output, dtype=np.float32)

    js = distance.jensenshannon(p=output, q=expected)
    assert js < 0.05


def test_qgreedy_policy_init():
    qgreedy_policy = create_qgreedy_policy(
        qtable=[[1, 2, 3, 4]],
    )

    assert qgreedy_policy.emit_log_probability is False
    np.testing.assert_array_equal(
        qgreedy_policy._state_action_value_table, np.array([[1, 2, 3, 4]], np.float32)
    )


def test_qgreedy_policy_init_with_emit_log_probability():
    qgreedy_policy = create_qgreedy_policy(
        qtable=[[1, 2, 3, 4]],
        emit_log_probability=True,
    )

    assert qgreedy_policy.emit_log_probability is True
    np.testing.assert_array_equal(
        qgreedy_policy._state_action_value_table, np.array([[1, 2, 3, 4]], np.float32)
    )


def test_qgreedy_policy_action():
    qgreedy_policy = create_qgreedy_policy(
        qtable=[[0, 0, 1, 0]],
    )

    # single state: 0
    _policy_step = qgreedy_policy.action(
        observation=np.array(0),
        policy_state=qgreedy_policy.get_initial_state(),
    )

    np.testing.assert_array_equal(_policy_step.action, 2)
    assert _policy_step.info == {}
    assert _policy_step.state == ()


def test_qgreedy_policy_action_with_emit_log_probability():
    qgreedy_policy = create_qgreedy_policy(
        qtable=[[0, 0, 1, 0]],
        emit_log_probability=True,
    )
    _policy_step = qgreedy_policy.action(
        observation=np.array(0),
        policy_state=qgreedy_policy.get_initial_state(),
    )

    np.testing.assert_array_equal(_policy_step.action, 2)
    assert _policy_step.info == defaults.policy_info(log_probability=np.math.log(1.0))
    assert _policy_step.state == ()


def test_qgreedy_policy_validity():
    num_actions = 4
    qgreedy_policy = create_qgreedy_policy(
        qtable=[[0, 0, 1, 0]],
        emit_log_probability=True,
    )

    # Only one action is chosen
    expected = np.array(
        [0, 0, 1.0, 0],
        np.float32,
    )
    output = np.zeros(shape=(num_actions), dtype=np.int64)
    for _ in range(10_000):
        _policy_step = qgreedy_policy.action(observation=np.array(0))
        output[_policy_step.action] += 1
    output = output / np.sum(output, dtype=np.float32)

    js = distance.jensenshannon(p=output, q=expected)
    assert js < 0.05


def test_egreedy_policy_init():
    epsilon_greedy_policy = create_epsilon_greedy_policy(
        qtable=[[0, 0, 1, 0]],
        epsilon=0.1,
    )

    assert epsilon_greedy_policy.emit_log_probability is False
    assert epsilon_greedy_policy._num_actions == 4
    assert epsilon_greedy_policy.epsilon == 0.1
    assert isinstance(epsilon_greedy_policy.explore_policy, policies.PyRandomPolicy)
    assert epsilon_greedy_policy.explore_policy._num_actions == 4
    assert epsilon_greedy_policy.explore_policy.emit_log_probability is False


def test_egreedy_policy_init_with_emit_log_probability():
    epsilon_greedy_policy = create_epsilon_greedy_policy(
        qtable=[[0, 0, 1, 0]],
        epsilon=0.1,
        emit_log_probability=True,
    )
    assert epsilon_greedy_policy.emit_log_probability is True
    assert epsilon_greedy_policy._num_actions == 4
    assert epsilon_greedy_policy.epsilon == 0.1
    assert isinstance(epsilon_greedy_policy.explore_policy, policies.PyRandomPolicy)
    assert epsilon_greedy_policy.explore_policy._num_actions == 4
    assert epsilon_greedy_policy.explore_policy.emit_log_probability is True


def test_egreedy_policy_action():
    epsilon_greedy_policy = create_epsilon_greedy_policy(
        qtable=[[0, 0, 1, 0]],
        epsilon=0.1,
    )
    _policy_step = epsilon_greedy_policy.action(
        observation=np.array(0),
        policy_state=epsilon_greedy_policy.get_initial_state(),
    )

    assert _policy_step.action in list(range(4))
    assert _policy_step.info == {}
    assert _policy_step.state == ()


def test_egreedy_policy_action_with_emit_log_probability():
    num_actions = 4
    epsilon_greedy_policy = create_epsilon_greedy_policy(
        qtable=[[0, 0, 1, 0]],
        epsilon=0.1,
        emit_log_probability=True,
    )
    _policy_step = epsilon_greedy_policy.action(
        observation=np.array(0),
        policy_state=epsilon_greedy_policy.get_initial_state(),
    )

    assert _policy_step.action in list(range(num_actions))
    assert np.allclose(
        _policy_step.info["log_probability"],
        np.math.log((0.1 / num_actions) + 0.9),
    ) or np.allclose(
        _policy_step.info["log_probability"], np.math.log((0.1 / num_actions))
    )
    assert _policy_step.state == ()


def test_egreedy_policy_validity():
    num_actions = 4
    epsilon = 0.2
    epsilon_greedy_policy = create_epsilon_greedy_policy(
        qtable=[[0, 0, 1, 0]],
        epsilon=epsilon,
        emit_log_probability=True,
    )

    # Avery action is chosen at least 1/explore times
    expected = np.array(
        [epsilon / num_actions] * num_actions,
        np.float32,
    )
    # Action 2 is the best, add exploited
    expected[2] += 1.0 - epsilon
    output = np.zeros(shape=(num_actions), dtype=np.int64)
    for _ in range(10_000):
        _policy_step = epsilon_greedy_policy.action(observation=np.array(0))
        output[_policy_step.action] += 1
    output = output / np.sum(output, dtype=np.float32)

    js_distance = distance.jensenshannon(p=output, q=expected)
    assert js_distance < 0.05


def create_random_policy(
    num_actions: int,
    emit_log_probability: bool = False,
) -> policies.PyRandomPolicy:
    return policies.PyRandomPolicy(
        num_actions=num_actions,
        emit_log_probability=emit_log_probability,
    )


def create_qgreedy_policy(
    qtable: Sequence[Sequence[float]],
    emit_log_probability: bool = False,
) -> policies.PyQGreedyPolicy:
    return policies.PyQGreedyPolicy(
        state_id_fn=defaults.identity,
        action_values=np.array(qtable, np.float32),
        emit_log_probability=emit_log_probability,
    )


def create_epsilon_greedy_policy(
    qtable: Sequence[Sequence[float]],
    epsilon: float,
    emit_log_probability: bool = False,
) -> policies.PyEpsilonGreedyPolicy:
    _, num_actions = np.array(qtable).shape
    return policies.PyEpsilonGreedyPolicy(
        policy=create_qgreedy_policy(
            qtable=qtable,
            emit_log_probability=emit_log_probability,
        ),
        num_actions=num_actions,
        epsilon=epsilon,
        emit_log_probability=emit_log_probability,
    )
