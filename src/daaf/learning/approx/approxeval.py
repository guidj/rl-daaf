"""
Policy evaluation methods with approximation.
"""

import copy
from typing import Iterator, Tuple

import gymnasium as gym
import numpy as np

from daaf import core, envplay
from daaf.core import GeneratesEpisode
from daaf.learning import opt, utils
from daaf.learning.approx import modelspec


def gradient_monte_carlo_state_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: opt.LearningRateSchedule,
    estimator: modelspec.ApproxFn,
    generate_episode: GeneratesEpisode = envplay.generate_episode,
) -> Iterator[Tuple[int, modelspec.ApproxFn]]:
    """
    Gradient monte-carlo based uses returns to
    approximate the value function of a policy.
    """
    steps_counter = 0
    for episode in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episode(environment, policy))
        episode_return = np.sum([experience.reward for experience in experiences])
        # while step isn't last
        for experience in experiences:
            alpha = lrs(episode=episode, step=steps_counter)
            state_value = estimator.predict(experience.observation)
            gradients = estimator.gradients(experience.observation)
            weights = estimator.weights()
            new_weights = weights + alpha * (episode_return - state_value) * gradients

            if utils.nan_or_inf(state_value):
                raise RuntimeError(f"Value estimate is NaN or Inf: {state_value}")
            if utils.nan_or_inf(gradients):
                raise RuntimeError(f"Gradients are NaN or Inf: {gradients}")
            estimator.assign_weights(new_weights)
            # update returns for the next state
            episode_return -= experience.reward
            steps_counter += 1
        # need to copy values because they can be mutable np.ndarrays or tf.tensors
        # we use shallow copy because tf doesn't play nicely with deepcopy
        yield len(experiences), copy.copy(estimator)
