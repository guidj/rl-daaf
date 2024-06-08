"""
This modules has baseline methods for
policy evaluation with DAAF.
"""

import collections
import copy
from typing import DefaultDict, Dict, Generator, List

import gymnasium as gym
import numpy as np
from rlplg import core, envplay
from rlplg.core import MapsToIntId
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policyeval


def onpolicy_one_step_td_state_values_only_aggregate_updates(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[policyeval.PolicyEvalSnapshot, None, None]:
    """
    TD(0) or one-step TD.
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
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    values = copy.deepcopy(initial_values)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        while True:
            try:
                traj_step = next(trajectory)
            except StopIteration:
                break
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - 2, None)

            if experiences[step].info["imputed"] is False:
                state_id = state_id_fn(experiences[step].observation)
                next_state_id = state_id_fn(experiences[step + 1].observation)
                alpha = lrs(episode=episode, step=steps_counter)
                if experiences[step].terminated:
                    values[state_id] = 0.0
                else:
                    values[state_id] += alpha * (
                        experiences[step].reward
                        + gamma * values[next_state_id]
                        - values[state_id]
                    )
            steps_counter += 1
            step += 1

        # need to copy values because it's a mutable numpy array
        yield policyeval.PolicyEvalSnapshot(
            steps=traj_step_idx, values=copy.deepcopy(values)
        )


def onpolicy_first_visit_monte_carlo_state_values_only_aggregate_updates(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    gamma: float,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[policyeval.PolicyEvalSnapshot, None, None]:
    """
    First-Visit Monte Carlo Prediction.
    Estimates V(s) for a fixed policy pi.
    Source: http://www.incompleteideas.net/book/ebook/node51.html

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the V(s) table.
        initial_values: Initial state-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    values = copy.deepcopy(initial_values)
    state_updates: DefaultDict[int, int] = collections.defaultdict(int)
    state_visits: DefaultDict[int, int] = collections.defaultdict(int)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences_ = list(generate_episode(environment, policy))
        num_steps = len(experiences_)
        # reverse list and ammortize state visits
        experiences: List[core.TrajectoryStep] = []
        while len(experiences_) > 0:
            experience = experiences_.pop()
            state_visits[state_id_fn(experience.observation)] += 1
            experiences.append(experience)

        episode_return = np.nan
        while True:
            try:
                # delete what we no longer need
                experience = experiences.pop(0)
            except IndexError:
                break
            state_id = state_id_fn(experience.observation)
            reward = experience.reward
            # update return
            if experience.info["imputed"] is False:
                if np.isnan(episode_return):
                    episode_return = 0.0
                episode_return = gamma * episode_return + reward
            state_visits[state_id] -= 1

            # no feedback; no updates
            if np.isnan(episode_return):
                continue
            if state_visits[state_id] == 0:
                if experience.terminated:
                    values[state_id] = 0.0
                else:
                    if state_updates[state_id] == 0:
                        # first value
                        values[state_id] = episode_return
                    else:
                        values[state_id] = values[state_id] + (
                            (episode_return - values[state_id])
                            / state_updates[state_id]
                        )
                state_updates[state_id] += 1

        # need to copy values because it's a mutable numpy array
        yield policyeval.PolicyEvalSnapshot(
            steps=num_steps, values=copy.deepcopy(values)
        )


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
        final_step = np.iinfo(np.int64).max
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        # In the absence of a step as terminal
        # or truncated, `empty_steps` prevents
        # infinite loops
        empty_steps = 0
        while True:
            if step > final_step or empty_steps > nstep:
                break
            try:
                traj_step = next(trajectory)
            except StopIteration:
                empty_steps += 1
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1

            # TD requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - nstep, None)

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
                if experiences[tau].terminated:
                    values[state_id] = 0.0
                else:
                    values[state_id] += alpha * (returns - values[state_id])
                steps_counter += 1
            step += 1
        # need to copy qtable because it's a mutable numpy array
        yield policyeval.PolicyEvalSnapshot(
            steps=traj_step_idx, values=copy.deepcopy(values)
        )
