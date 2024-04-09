"""
In this module, we can do on-policy evaluation with
delayed aggregate feedback - for tabular problems.
"""

import copy
from typing import Generator

import gymnasium as gym
import numpy as np
from rlplg import core, envplay
from rlplg.core import MapsToIntId
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policyeval


def nstep_td_state_values_on_aggregate_start_steps(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    nstep: int,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[policyeval.PolicyEvalSnapshot, None, None]:
    """
    n-step TD learning.
    Estimates V(s) for a fixed policy pi.
    Source: https://en.wikipedia.org/wiki/Temporal_difference_learning

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the V(s) table.
        initial_values: Initial state-value estimates.
        generate_episodes: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract reward access references by one.
    """
    values = copy.deepcopy(initial_values)
    steps_counter = 0
    for episode in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        # TODO: refactor - memory efficiency
        experiences = list(generate_episode(environment, policy))
        final_step = np.iinfo(np.int64).max
        for step, _ in enumerate(experiences):
            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
                    final_step = step + 1
            tau = step - nstep + 1
            if tau >= 0 and experiences[tau].info["ok_nstep_tau"]:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step)
                returns = 0.0

                for i in range(min_idx, max_idx + 1):
                    returns += (gamma ** (i - tau - 1)) * experiences[i - 1].reward
                if tau + nstep < final_step:
                    returns += (gamma**nstep) * values[
                        state_id_fn(experiences[tau + nstep].observation),
                    ]
                state_id = state_id_fn(experiences[tau].observation)
                alpha = lrs(episode=episode, step=steps_counter)
                values[state_id] += alpha * (returns - values[state_id])
            steps_counter += 1
        # need to copy qtable because it's a mutable numpy array
        yield policyeval.PolicyEvalSnapshot(
            steps=len(experiences), values=copy.deepcopy(values)
        )
