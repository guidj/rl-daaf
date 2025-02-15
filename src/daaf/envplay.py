"""
This module contains utilities for rollouts, and
trajectory data.
"""

import math
from typing import Generator, Optional

import gymnasium as gym
from gymnasium import wrappers as gym_wrappers

from daaf import core
from daaf.core import TimeStep


def generate_episodes(
    environment: gym.Env,
    policy: core.PyPolicy,
    num_episodes: int,
    max_episode_steps: Optional[int] = None,
) -> Generator[core.TrajectoryStep, None, None]:
    """
    Generates `num_episodes` episodes using the environment
    and policy provided for rollout.

    A `TimeStep` from an env has the next step, and reward, SA(RS)A in SARSA.
    Thus, to generated a `Trajectory`, we need to have the previous
    state, the action, and reward.

    Args:
        enviroment: environment to use.
        policy: for rollout.
        num_episodes: number of rollout episodes.
        max_episode_steps: maximum steps before truncating an episode.

    Yields:
        Trajectory instances from episodic rollouts, one step at a time.
    """
    for _ in range(num_episodes):
        yield from generate_episode(
            environment, policy=policy, max_steps=max_episode_steps
        )


def generate_episode(
    environment: gym.Env,
    policy: core.PyPolicy,
    max_steps: Optional[int] = None,
) -> Generator[core.TrajectoryStep, None, None]:
    """
    Generates an episode using the environment
    and policy provided for rollout.

    A `TimeStep` from an env has the next step, and reward, SA(RS)A in SARSA.
    Thus, to generated a `Trajectory`, we need to have the previous
    state, the action, and reward.

    Args:
        enviroment: environment to use.
        policy: for rollout.
        max_steps: maximum steps before truncating an episode.

    Yields:
        Trajectory instances from episodic rollouts, one step at a time.
    """
    if max_steps is not None:
        environment = gym_wrappers.TimeLimit(environment, max_episode_steps=max_steps)
    obs, _ = environment.reset()
    policy_state = policy.get_initial_state()
    time_step: TimeStep = obs, math.nan, False, False, {}
    while True:
        obs, _, terminated, truncated, _ = time_step
        policy_step = policy.action(obs, policy_state)
        next_time_step = environment.step(policy_step.action)
        yield core.TrajectoryStep.from_transition(
            time_step, policy_step, next_time_step
        )
        if terminated or truncated:
            break
        policy_state = policy_step.state
        time_step = next_time_step


def identity_replay(
    event: core.TrajectoryStep,
) -> Generator[core.TrajectoryStep, None, None]:
    """
    Yields:
        The given trajectory step, as is.
    """
    yield event
