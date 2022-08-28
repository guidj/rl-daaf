"""
Policy evaluation functions modified for Cumulative Periodic Rewards.
"""


import copy
from typing import Any, Callable, Generator, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import Array

from rlplg import envplay
from rlplg.learning.tabular import policies


def cpr_nstep_sarsa_prediction(
    policy: policies.PyQGreedyPolicy,
    collect_policy: py_policy.PyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    nstep: int,
    policy_probability_fn: Callable[
        [py_policy.PyPolicy, trajectory.Trajectory],
        float,
    ],
    collect_policy_probability_fn: Callable[
        [py_policy.PyPolicy, trajectory.Trajectory],
        float,
    ],
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
    Off-policy n-step Sarsa Prediction.
    Estimates Q (table) for a fixed policy pi.

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        collect_policy: A behavior policy, used to generate episodes.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        nstep: The number of steps before value updates in the MDP sequence.
        policy_probability_fn: returns action propensity for the target policy,
            given a trajectory.
        collect_policy_probability_fn: returns action propensity for the collect policy,
            given a trajectory.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        generate_episodes: A function that generates trajectories.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.collect_episodes`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    if nstep < 1:
        raise ValueError(f"nstep must be > 1: {nstep}")
    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)

    for _ in range(num_episodes):
        final_step = np.inf
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(
            generate_episodes(environment, collect_policy, num_episodes=1)
        )
        for step, _ in enumerate(experiences):
            if step % reward_period != 0:
                continue
            if step < final_step:
                # we don't need to transition because we already collected the experience
                # a better way to determine the next state is terminal one
                if np.array_equal(experiences[step].step_type, ts.StepType.LAST):
                    final_step = step + 1

            tau = step - nstep + 1
            if tau >= 0:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step - 1)
                rho = 1.0
                returns = 0.0

                for i in range(min_idx, max_idx + 1):
                    rho *= policy_probability_fn(
                        policy, experiences[i]
                    ) / collect_policy_probability_fn(collect_policy, experiences[i])
                    returns += (gamma ** (i - tau - 1)) * experiences[i - 1].reward
                if tau + nstep < final_step:
                    returns += (gamma**nstep) * qtable[
                        state_id_fn(experiences[tau + nstep].observation),
                        action_id_fn(experiences[tau + nstep].action),
                    ]

                state_id = state_id_fn(experiences[tau].observation)
                action_id = action_id_fn(experiences[tau].action)
                qtable[state_id, action_id] += (
                    alpha * rho * (returns - qtable[state_id, action_id])
                )
            if tau == final_step - 1:
                break

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)


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
