import numpy as np

from daaf import core
from daaf.learning.tabular import dynamicprog, policies
from tests.daaf import defaults


def test_dynamic_iterative_policy_evaluation():
    environment = defaults.CountEnv()
    mdp = core.EnvMdp(env_space=core.EnvSpace(4, 2), transition=environment.transition)
    policy = create_observable_random_policy(num_actions=mdp.env_space.num_actions)
    delta = 1e-8

    actual_state_values = np.array([-33.0, -22.0, -11.0, 0.0])
    actual_state_action_values = np.array(
        [[-43, -23.0], [-32.0, -12.0], [-21.0, -1.0], [0.0, 0.0]]
    )
    state_values = dynamicprog.iterative_policy_evaluation(mdp, policy, accuracy=delta)
    state_action_values = dynamicprog.action_values_from_state_values(mdp, state_values)

    np.testing.assert_array_almost_equal(actual_state_values, state_values, decimal=8)
    np.testing.assert_array_almost_equal(
        actual_state_action_values, state_action_values, decimal=8
    )


def create_observable_random_policy(
    num_actions: int,
    emit_log_probability: bool = False,
):
    return policies.PyRandomPolicy(
        num_actions=num_actions,
        emit_log_probability=emit_log_probability,
    )
