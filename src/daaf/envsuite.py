"""
This module has utilities to load environments,
defined in either `rlplg` or gymnasium.
"""

import functools
from typing import Any, Callable, Mapping, SupportsInt

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from daaf import core
from daaf.core import EnvTransition
from daaf.environments import (
    abcseq,
    gridworld,
    iceworld,
    randomwalk,
    redgreen,
    towerhanoi,
)

FROZEN_LAKE = "FrozenLake-v1"
CLIFF_WALKING = "CliffWalking-v0"

SUPPORTED_LIB_ENVS = frozenset(
    (
        abcseq.ENV_NAME,
        gridworld.ENV_NAME,
        randomwalk.ENV_NAME,
        redgreen.ENV_NAME,
        towerhanoi.ENV_NAME,
        iceworld.ENV_NAME,
    )
)
SUPPORTED_GYM_ENVS = frozenset((FROZEN_LAKE, CLIFF_WALKING))


class DefaultGymEnvMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: SupportsInt) -> int:
        """
        Maps an observation to a state ID.
        """
        del self
        return int(observation)

    def action(self, action: SupportsInt) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        return int(action)


def load(name: str, **args) -> core.EnvSpec:
    """
    Creates an environment with the given arguments.

    Args:
        name: unique identifier.
        args: parameters that are passed to an environment constructor.

    Returns:
        An instantiated environment.

    Raises:
        A ValueError is the environment is unsupported.

    """
    constructors = __environment_spec_constructors()
    if name not in constructors:
        raise ValueError(f"Unsupported environment: {name}.")
    return constructors[name](**args)


def __environment_spec_constructors() -> Mapping[str, Callable[..., core.EnvSpec]]:
    """
    Creates a mapping of lib and gym environment names to their constructors.

    Returns:
        A mapping from a unique string identifier to a constructor.

    """
    lib_envs: Mapping[str, Callable[..., core.EnvSpec]] = {
        abcseq.ENV_NAME: abcseq.create_env_spec,
        gridworld.ENV_NAME: gridworld.create_envspec_from_grid_text,
        iceworld.ENV_NAME: iceworld.create_envspec_from_map,
        randomwalk.ENV_NAME: randomwalk.create_env_spec,
        redgreen.ENV_NAME: redgreen.create_env_spec,
        towerhanoi.ENV_NAME: towerhanoi.create_env_spec,
    }
    gym_envs = {
        name: __gym_environment_spec_constructor(name) for name in SUPPORTED_GYM_ENVS
    }
    return {**lib_envs, **gym_envs}


def __gym_environment_spec_constructor(
    name: str,
) -> Callable[..., core.EnvSpec]:
    """
    Partially initiatializes a gym environment spec with its name.

    Args:
        name: unique identifier.

    Returns:
        A callable that creates a gym environment.
    """

    return functools.partial(__gym_environment_spec, name)


def __gym_environment_spec(name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
    """
    Creates a gym environment spec.
    """
    environment = __make_gym_environment(name, **kwargs)
    discretizer = __gym_environment_discretizer(name)
    mdp = core.EnvMdp(
        env_desc=__parse_gym_env_desc(environment=environment),
        transition=__parse_gym_env_transition(environment),
    )
    return core.EnvSpec(
        name=name,
        level=__encode_env(**kwargs),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
    )


def __make_gym_environment(name: str, **kwargs: Mapping[str, Any]) -> gym.Env:
    """
    Creates discretizers for supported environments.
    """
    return gym.make(name, **kwargs)


def __parse_gym_env_desc(environment: gym.Env) -> core.EnvDesc:
    """
    Infers the EnvDesc from a `gym.Env`.
    """
    num_actions = (
        environment.action_space.n
        if isinstance(environment.action_space, spaces.Discrete)
        else np.inf
    )
    num_states = (
        environment.observation_space.n
        if isinstance(environment.action_space, spaces.Discrete)
        else np.inf
    )
    return core.EnvDesc(num_states=num_states, num_actions=num_actions)


def __parse_gym_env_transition(environment: gym.Env) -> EnvTransition:
    """
    Parses transition data from a `gym.Env`.
    """
    transition: EnvTransition = getattr(environment.unwrapped, "P")
    return transition


def __gym_environment_discretizer(name: str) -> core.MdpDiscretizer:
    """
    Creates discretizers for supported environments.
    """
    del name
    return DefaultGymEnvMdpDiscretizer()


def __encode_env(**kwargs: Mapping[str, Any]) -> str:
    """
    Encodes environment into a unique hash.
    """
    keys = []
    values = []
    for key, value in sorted(kwargs.items()):
        keys.append(key)
        values.append(value)

    hash_key = tuple(keys) + tuple(values)
    return core.encode_env(signature=hash_key)
