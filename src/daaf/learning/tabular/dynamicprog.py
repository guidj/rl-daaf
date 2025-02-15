"""
This module implements value function estimation
with dynamic programming.
"""

import copy
import logging

import numpy as np

from daaf import core
from daaf.learning.tabular import policies


def iterative_policy_evaluation(
    mdp: core.Mdp,
    policy: policies.SupportsStateActionProbability,
    gamma: float = 1.0,
    accuracy: float = 1e-8,
):
    """
    Implementation of dynamic programming for state value function computation.
    V(s)_{pi}.
    Vectorized implementation.
    """
    pi_action = np.zeros(
        (mdp.env_desc.num_states, mdp.env_desc.num_actions), dtype=np.float64
    )
    m_transition = np.zeros(
        (
            mdp.env_desc.num_states,
            mdp.env_desc.num_actions,
            mdp.env_desc.num_states,
        ),
        dtype=np.float64,
    )
    m_reward = np.zeros(
        (
            mdp.env_desc.num_states,
            mdp.env_desc.num_actions,
            mdp.env_desc.num_states,
        ),
        dtype=np.float64,
    )

    for state, action_transitions in mdp.transition.items():
        for action, transitions in action_transitions.items():
            pi_action[state, action] = policy.state_action_prob(state, action)
            for prob, next_state, reward, _ in transitions:
                m_transition[state, action, next_state] = prob
                m_reward[state, action, next_state] = reward

    m_state_values = np.tile(
        np.zeros(shape=mdp.env_desc.num_states), (mdp.env_desc.num_actions, 1)
    )
    while True:
        delta = np.zeros(shape=mdp.env_desc.num_states)
        current_state_values = copy.deepcopy(m_state_values[0])
        # |S| x |A| x |S'|
        # |S| x |A|
        values = np.sum(m_transition * (m_reward + gamma * m_state_values), axis=2)
        new_state_values = np.diag(np.dot(pi_action, np.transpose(values)))
        m_state_values = np.tile(new_state_values, (mdp.env_desc.num_actions, 1))

        delta = np.maximum(
            delta,
            np.abs(current_state_values - new_state_values),
        )
        if np.all(delta < accuracy):
            return m_state_values[0, :]


def action_values_from_state_values(
    mdp: core.Mdp, state_values: np.ndarray, gamma: float = 1.0
):
    """
    Compute Q(s,a) using V(s)
    """
    qtable = np.zeros(shape=(mdp.env_desc.num_states, mdp.env_desc.num_actions))
    for state, action_transitions in mdp.transition.items():
        for action, transitions in action_transitions.items():
            cu_value = 0
            for prob, next_state, reward, _ in transitions:
                next_state_value = gamma * state_values[next_state]
                logging.debug(
                    "state: %d, action: %d, next_state: %d (prob: %f), reward: %f, next_state_value: %d",
                    state,
                    action,
                    next_state,
                    prob,
                    reward,
                    next_state_value,
                )
                cu_value += prob * (reward + next_state_value)
            qtable[state, action] = cu_value
    return qtable
