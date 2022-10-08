"""
Policy evaluation functions modified for Cumulative Periodic Rewards.
"""


import copy
from typing import Any, Callable, Generator, Tuple

import numpy as np
from rlplg import envplay
from rlplg.learning.tabular import policies
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import Array


def cpr_sarsa_prediction(
    policy: policies.PyQGreedyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    reward_period: int,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, Array, float], None, None]:
    """
    On-policy Sarsa Prediction.
    Estimates Q (table) for a fixed policy pi.
    Source: https://homes.cs.washington.edu/~bboots/RL-Spring2020/Lectures/TD_notes.pdf
    In the document, they refer to Algorithm 15 as Algorithm 16.

    Note to self: As long you don't use the table you're updating,
    the current approach is fine

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        event_mapper: A function that generates trajectories from a given trajectory.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.identity_replay`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, num_episodes=1))
        for step in range(len(experiences) - 1):
            if (step + 1) % reward_period != 0:
                continue
            state_id = state_id_fn(experiences[step].observation)
            action_id = action_id_fn(experiences[step].action)
            reward = experiences[step].reward

            next_state_id = state_id_fn(experiences[step + 1].observation)
            next_action_id = action_id_fn(experiences[step + 1].action)

            qtable[state_id, action_id] += alpha * (
                reward
                + gamma * qtable[next_state_id, next_action_id]
                - qtable[state_id, action_id]
            )

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)

