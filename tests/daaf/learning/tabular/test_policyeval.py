"""
Policy evaluation methods tests.

Note: once the terminal state is entered, the MDP trajectory ends.
So only rewards from the transitions up until the terminal state matter.
Hence, no actions taken in the terminal state are used
in policy evaluation algorithms.

"""

import gymnasium as gym
import numpy as np
import pytest

from daaf import core
from daaf.learning import opt
from daaf.learning.tabular import policies, policyeval
from tests.daaf import defaults


def test_onpolicy_first_visit_monte_carlo_action_values_with_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    At each step, except the last, there are value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    (undiscoutned) Returns: 0, -1, -1, -2
    (gamma=0.95) Returns: 0, 0, -1, -1.95

    """

    results = policyeval.onpolicy_first_visit_monte_carlo_action_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_array_almost_equal(
        snapshot.values, np.array([[0, -2.8525], [0, -1.95], [0, -1], [0, 0]])
    )


def test_onpolicy_first_visit_monte_carlo_action_values_with_one_episode_convering_every_action(
    environment: gym.Env,
):
    """
    At each step, except the last, there are value updates.

    First moves are wrong, the ones that follow are right.
    init Q(s,a) = 1
    G = 0
    Q(s,a) = Q(s,a) + [R + gamma * G]

    Trajectory: (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)
    Reversed: (3, 0), (2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)
    Reversed Rewards: 0, 0, -10, -1, -10, -1, -10
    (undiscoutned) Returns: 0, 0, -10, -11, -21, -22, -32
    (gamma=0.95) Returns: 0, 0, -10, -10.5, -19.975, -19.97625, -28.9774375

    ...
    """
    stochastic_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = policyeval.onpolicy_first_visit_monte_carlo_action_values(
        policy=stochastic_policy,
        environment=environment,
        num_episodes=1,
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 7
    np.testing.assert_array_almost_equal(
        snapshot.values,
        np.array(
            [[-29.751219, -20.790756], [-20.832375, -11.4025], [-10.95, -1], [0, 0]]
        ),
    )


def test_onpolicy_sarsa_action_values_with_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    At each step, except the last, there are value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    init Q(s,a) = 0
    Q(0,0) = Q(0,0) + 0.1 * [R + 0.95 * Q(0,1) - Q(0,0)]
    Q(0,0) = 0 + 0.1 * [-10 + 0.95 * 0 - 0]``
    Q(0,0) = 0 + 0.1 * (-10) = -0.1
    """

    results = policyeval.onpolicy_sarsa_action_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_array_almost_equal(
        snapshot.values, np.array([[0, -0.1], [0, -0.1], [0, -0.1], [0, 0]])
    )


def test_onpolicy_sarsa_action_values_with_one_episode_convering_every_action(
    environment: gym.Env,
):
    """
    At each step, except the last, there are value updates.

    First moves are wrong, the ones that follow are right.
    init Q(s,a) = 0
    Q(s,a) = Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
    Q(0,0) = Q(0,0) + 0.1 * [R + 0.95 * Q(0,1) - Q(0,0)]
    Q(0,0) = 0 + 0.1 * [-10 + 0.95 * 0 - 0]
    Q(0,0) = 0 + 0.1 * (-10) = -0.1
    ...
    """
    stochastic_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1, 0],
    )

    results = policyeval.onpolicy_sarsa_action_values(
        policy=stochastic_policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 7
    np.testing.assert_array_almost_equal(
        snapshot.values, np.array([[-1, -0.1], [-1, -0.1], [-1, -0.1], [0, 0]])
    )


def test_onpolicy_first_visit_monte_carlo_state_values_with_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    Reversed Returns: 0, -1, -2, -3
    """

    results = policyeval.onpolicy_first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [-3, -2, -1, 0])


def test_onpolicy_first_visit_monte_carlo_state_values_with_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    Reversed Returns: 0, -1, -2, -3
    """

    results = policyeval.onpolicy_first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [-3, -2, -1, 0])
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [-3, -2, -1, 0])


def test_onpolicy_one_step_td_state_values_with_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0
    """

    results = policyeval.onpolicy_one_step_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.1, -0.1, -0.1, 0])


def test_onpolicy_one_step_td_state_values_with_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0

    Episode 1:
    V(s) = V(s) + alpha * (r + gamma * V(s') - V(s))
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(0) = 0 + (-0.1) = -0.1
    ..
    V(1) = -0.1
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(2) = -0.1

    Episode 2:
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = -0.1 + 0.1 * (-1 + 1.0 * -0.1 - (-0.1))
    V(0) = -0.2
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = -0.1 + 0.1 * (-1 + 1.0 * 0 - (-0.1))
    V(2) = -0.19
    """

    results = policyeval.onpolicy_one_step_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.1, -0.1, -0.1, 0])
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.2, -0.2, -0.19, 0])


def test_onpolicy_nstep_td_state_values_with_one_nstep_and_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0
    """

    results = policyeval.onpolicy_nstep_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        nstep=1,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.1, -0.1, -0.1, 0])


def test_onpolicy_nstep_td_state_values_with_one_nstep_and_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0

    Episode 1:
    V(s) = V(s) + alpha * (r + gamma * V(s') - V(s))
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(0) = 0 + (-0.1) = -0.1
    ..
    V(1) = -0.1
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(2) = -0.1

    Episode 2:
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = -0.1 + 0.1 * (-1 + 1.0 * -0.1 - (-0.1))
    V(0) = -0.2
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = -0.1 + 0.1 * (-1 + 1.0 * 0 - (-0.1))
    V(2) = -0.19
    """

    results = policyeval.onpolicy_nstep_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        nstep=1,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.1, -0.1, -0.1, 0])
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.2, -0.2, -0.19, 0])


def test_onpolicy_nstep_td_state_values_with_two_nstep_and_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0
    """

    results = policyeval.onpolicy_nstep_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        nstep=2,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.2, -0.2, -0.1, 0])


def test_onpolicy_nstep_td_state_values_with_two_nstep_and_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0

    Episode 1:'
    G = n-step discounted rewards + discounted value
        of next state (if there is one)
    V(s_{tau}) = V(s_{tau}) + alpha * (G - V(s_{tau}))
    V(0) = V(0) + 0.1 * (-2 - V(0)) | G = -2 + 0
    V(0) = 0 + 0.1 * (-2 - 0)
    V(0) = 0 + (-0.2) = -0.2
    ..
    V(1) = -0.2 | G = -2
    ...
    V(2) = V(2) + 0.1 * (-1 - V(2)) | G = -1 + 0
    V(2) = 0 + 0.1 * -1
    V(2) = -0.1

    V(s) = [-0.2, -0.2, -0.1, 0]

    Episode 2:
    V(0) = V(0) + 0.1 * (G - V(0)) | G = -2 + -0.1
    V(0) = -0.2 + 0.1 * (-2.1 - (-0.2))
    V(0) = -0.39

    V(1) = V(1) + 0.1 * (G - V(1)) | G = -2
    V(1) = -0.2 + 0.1 * (-2 - (-0.2))
    V(1) = -0.38
    ...

    V(2) = V(2) + 0.1 * (G - V(2)) | G = -1
    V(2) = -0.1 + 0.1 * (-1 - (-0.1))
    V(2) = -0.19
    """

    results = policyeval.onpolicy_nstep_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        nstep=2,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.2, -0.2, -0.1, 0])
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_allclose(snapshot.values, [-0.39, -0.38, -0.19, 0.0])


def test_offpolicy_monte_carlo_action_values_with_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    """

    results = policyeval.offpolicy_monte_carlo_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [[0, -3], [0, -2], [0, -1], [0, 0]])


def test_offpolicy_monte_carlo_action_values_with_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.
    After one episode, the value for the best actions match.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    """

    results = policyeval.offpolicy_monte_carlo_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=2,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    output_iter = iter(output)
    assert len(output) == 2
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [[0, -3], [0, -2], [0, -1], [0, 0]])
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_equal(snapshot.values, [[0, -3], [0, -2], [0, -1], [0, 0]])


def test_offpolicy_monte_carlo_action_values_with_one_episode_covering_every_action(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Only the last two state-action pairs are updated because
    after that, the 2nd to last action mistmatches the policy,
    turning W into zero - i.e. the importance sample ratio collapses.
    """

    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )
    results = policyeval.offpolicy_monte_carlo_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 7
    # we only learn about the last two (state, action) pairs.
    np.testing.assert_array_equal(snapshot.values, [[0, 0], [0, 0], [-11, -1], [0, 0]])


def test_offpolicy_monte_carlo_action_values_step_with_reward_discount():
    """
    G <- 0.9*10 + 100 = 109
    C(s,a) <- 50 + 2 = 52
    Q(s,a) <- 20 + 2/52 * (109-20) = 255/13
    W <- 2 * 0.8 = 1.6

    """
    mc_update = policyeval.offpolicy_monte_carlo_action_values_step(
        reward=100, returns=10, cu_sum=50, weight=2.0, value=20, rho=0.8, gamma=0.9
    )

    np.testing.assert_approx_equal(mc_update.returns, 109)
    np.testing.assert_approx_equal(mc_update.cu_sum, 52)
    np.testing.assert_approx_equal(mc_update.value, 609 / 26)
    np.testing.assert_approx_equal(mc_update.weight, 1.6)


def test_offpolicy_nstep_sarsa_action_values_with_one_nstep_and_one_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Each step value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    """

    results = policyeval.offpolicy_nstep_sarsa_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=1,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 4
    np.testing.assert_array_almost_equal(
        snapshot.values, np.array([[0.0, -0.1], [0.0, -0.1], [0.0, -0.1], [0.0, 0.0]])
    )


def test_offpolicy_nstep_sarsa_action_values_with_two_nsteps_and_two_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every step is updated after n+2 steps, so the final step isn't updated.
    """
    results = policyeval.offpolicy_nstep_sarsa_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=2,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    output_iter = iter(output)
    assert len(output) == 2
    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_almost_equal(
        snapshot.values,
        np.array([[0.0, -0.195], [0.0, -0.195], [0.0, -0.1], [0.0, 0.0]]),
    )

    snapshot = next(output_iter)
    assert snapshot.steps == 4
    np.testing.assert_array_almost_equal(
        snapshot.values,
        np.array([[0.0, -0.379525], [0.0, -0.3705], [0.0, -0.19], [0.0, 0.0]]),
    )


def test_offpolicy_nstep_sarsa_action_values_with_one_nstep_and_one_episode_covering_every_action(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    Every step preceding a correct step is updated.
    Every step following a mistep isn't.
    """
    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = policyeval.offpolicy_nstep_sarsa_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=1,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 7
    np.testing.assert_array_almost_equal(
        snapshot.values,
        np.array([[-1.0, -0.1], [-1.0, -0.1], [-1.0, -0.1], [0.0, 0.0]]),
    )


def test_offpolicy_nstep_sarsa_action_values_with_two_nsteps_and_one_episode_covering_every_action(
    environment: gym.Env,
    policy: core.PyPolicy,
):
    """
    No step gets updated.
    """
    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = policyeval.offpolicy_nstep_sarsa_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        lrs=opt.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=2,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    snapshot = next(iter(output))
    assert snapshot.steps == 7
    np.testing.assert_array_almost_equal(
        snapshot.values,
        np.array([[-2.19, 0.0], [-2.19, 0.0], [-2.19, -0.1], [0.0, 0.0]]),
    )


def policy_prob_fn(policy: core.PyPolicy, traj: core.TrajectoryStep) -> float:
    """The policy we're evaluating is assumed to be greedy w.r.t. Q(s, a).
    So the best action has probability 1.0, and all the others 0.0.
    In case multiple actions have the same value, the probability is
    equal between them.
    """

    qtable = getattr(policy, "_state_action_value_table")
    action_values = qtable[traj.observation]
    candidate_actions = np.flatnonzero(action_values == np.max(action_values))
    if traj.action in candidate_actions:
        return 1.0 / len(candidate_actions)
    return 0.0


def collect_policy_prob_fn(policy: core.PyPolicy, traj: core.TrajectoryStep) -> float:
    """The behavior policy is assumed to be fixed over the evaluation window.
    We log probabilities when choosing actions, so we can just use that information.
    For a random policy on K arms, log_prob = log(1/K).
    We just have to return exp(log_prob).
    """
    del policy
    prob: float = np.exp(traj.policy_info["log_probability"])
    return prob


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    del episode
    del step
    return initial_lr


@pytest.fixture(scope="function")
def qtable() -> np.ndarray:
    """
    Q-table for optimal actions.
    """
    return np.array([[-1, 0], [-1, 0], [-1, 0], [0, 0]], np.float32)


@pytest.fixture(scope="function")
def policy(environment: gym.Env, qtable: np.ndarray) -> core.PyPolicy:
    return policies.PyQGreedyPolicy(
        state_id_fn=defaults.identity,
        action_values=qtable,
        emit_log_probability=True,
    )


@pytest.fixture(scope="function")
def environment() -> gym.Env:
    """
    Test environment.
    """
    return defaults.CountEnv()
