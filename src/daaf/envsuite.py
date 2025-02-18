"""
This module has utilities to load environments,
defined in either `rlplg` or gymnasium.
"""

import functools
import hashlib
from typing import Any, Callable, Mapping, Optional, SupportsInt

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rlplg.environments import (
    abcseq,
    gridworld,
    iceworld,
    randomwalk,
    redgreen,
    towerhanoi,
)

from daaf import core
from daaf.core import EnvTransition, MutableEnvTransition

FROZEN_LAKE = "FrozenLake-v1"
CLIFF_WALKING = "CliffWalking-v0"

SUPPORTED_RLPLG_ENVS = frozenset(
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
KEY_HALF_SIZE = 5


class DefaultEnvMdp(core.Mdp):
    """
    Environment Dynamics for Gynasium environments that are
    compliant with Toy Text examples.
    """

    def __init__(self, env_space: core.EnvSpace, transition: EnvTransition):
        """
        Creates an MDP using transition mapping.

        In some of the environments, there are errors in the implementation
        for terminal states.
        We correct them.
        """
        self.__env_space = env_space
        self.__transition: MutableEnvTransition = {}
        # collections.defaultdict(lambda: collections.defaultdict(lambda: (0.0, 0.0)))
        # Find terminal states
        terminal_states = core.infer_env_terminal_states(transition)
        # Create mapping with correct transition for terminal states
        # This necessary because `env.P` in Gymnasium toy text
        # examples are incorrect.
        for state, action_transitions in transition.items():
            self.__transition[state] = {}
            for action, transitions in action_transitions.items():
                self.__transition[state][action] = []
                for prob, next_state, reward, terminated in transitions:
                    # if terminal state, override prob and reward for different states
                    if state in terminal_states:
                        prob = 1.0 if state == next_state else 0.0
                        reward = 0.0
                    self.__transition[state][action].append(
                        (
                            prob,
                            next_state,
                            reward,
                            terminated,
                        )
                    )

    @property
    def env_space(self) -> core.EnvSpace:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """
        return self.__env_space

    @property
    def transition(self) -> EnvTransition:
        """
        Returns:
            The mapping of state-action transition.
        """
        return self.__transition


class RlplgEnvSpecParser:
    """
    Env parser for Rlplg environments.
    """

    class DefaultRlplgEnvMdpDiscretizer(core.MdpDiscretizer):
        """
        Creates an environment discrete maps for states and actions.
        """

        def state(self, observation: Any) -> int:
            """
            Maps an observation to a state ID.
            """
            del self
            return int(observation.get("id"))

        def action(self, action: SupportsInt) -> int:
            """
            Maps an agent action to an action ID.
            """
            del self
            return int(action)

    def __init__(self):
        self._constructors = rlplg_env_constructors()

    def parse(self, name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
        """
        Creates an environment space given the arguments
        provided.
        """
        return self.__environment_spec(name, **kwargs)

    def __environment_spec(
        self, name: str, **kwargs: Mapping[str, Any]
    ) -> core.EnvSpec:
        """
        Creates a gym environment spec.
        """

        environment = self._constructors[name](**kwargs)
        discretizer = self.__environment_discretizer(name)
        mdp = DefaultEnvMdp(
            env_space=self.__parse_env_space(environment=environment),
            transition=self.__parse_env_transition(environment),
        )
        return core.EnvSpec(
            name=name,
            level=encode_env(**kwargs),
            environment=environment,
            discretizer=discretizer,
            mdp=mdp,
        )

    def __environment_discretizer(self, name: str) -> core.MdpDiscretizer:
        """
        Creates discretizers for supported environments.
        """
        del name
        return self.DefaultRlplgEnvMdpDiscretizer()

    def __parse_env_space(self, environment: gym.Env) -> core.EnvSpace:
        """
        Infers the EnvDesc from a `gym.Env`.
        """
        del self
        num_actions = (
            environment.action_space.n
            if isinstance(environment.action_space, spaces.Discrete)
            else np.inf
        )
        num_states = (
            environment.observation_space["id"].n
            if isinstance(environment.observation_space["id"], spaces.Discrete)
            else np.inf
        )
        return core.EnvSpace(num_states=num_states, num_actions=num_actions)

    def __parse_env_transition(self, environment: gym.Env) -> EnvTransition:
        """
        Parses transition data from a `gym.Env`.
        """
        del self
        return getattr(environment.unwrapped, "transition")  # type: ignore


class GymEnvSpecParser:
    """
    Env parser for Gynamsnium Toy Text environments.
    """

    class DefaultGymEnvMdpDiscretizer(core.MdpDiscretizer):
        """
        Creates an environment discrete maps for states and actions.
        """

        def state(self, observation: Any) -> int:
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

    def parse(self, name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
        """
        Creates an environment space given the arguments
        provided.
        """
        return self.__environment_spec(name, **kwargs)

    def __environment_spec(
        self, name: str, **kwargs: Mapping[str, Any]
    ) -> core.EnvSpec:
        """
        Creates a gym environment spec.
        """
        environment = gym.make(name, **kwargs)
        discretizer = self.__environment_discretizer(name)
        mdp = DefaultEnvMdp(
            env_space=self.__parse_env_space(environment=environment),
            transition=self.__parse_env_transition(environment),
        )
        return core.EnvSpec(
            name=name,
            level=encode_env(**kwargs),
            environment=environment,
            discretizer=discretizer,
            mdp=mdp,
        )

    def __environment_discretizer(self, name: str) -> core.MdpDiscretizer:
        """
        Creates discretizers for supported environments.
        """
        del name
        return self.DefaultGymEnvMdpDiscretizer()

    def __parse_env_space(self, environment: gym.Env) -> core.EnvSpace:
        """
        Infers the EnvDesc from a `gym.Env`.
        """
        del self
        num_actions = (
            environment.action_space.n
            if isinstance(environment.action_space, spaces.Discrete)
            else np.inf
        )
        num_states = (
            environment.observation_space.n
            if isinstance(environment.observation_space, spaces.Discrete)
            else np.inf
        )
        return core.EnvSpace(num_states=num_states, num_actions=num_actions)

    def __parse_env_transition(self, environment: gym.Env) -> EnvTransition:
        """
        Parses transition data from a `gym.Env`.
        """
        del self
        return getattr(environment.unwrapped, "P")  # type: ignore


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
        name: functools.partial(parse_rlplg_env, name) for name in SUPPORTED_RLPLG_ENVS
    }
    gym_envs = {
        name: functools.partial(parse_gym_env, name) for name in SUPPORTED_GYM_ENVS
    }
    return {**lib_envs, **gym_envs}


def parse_gym_env(name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
    """
    Gym parser.
    """
    return GymEnvSpecParser().parse(name, **kwargs)


def parse_rlplg_env(name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
    """
    Rlplg parser.
    """
    return RlplgEnvSpecParser().parse(name, **kwargs)


def rlplg_env_constructors() -> Mapping[str, Callable[..., gym.Env]]:
    """
    Synthetic sugar to make rlplg environments.
    """

    def make_iceworld(
        map: Optional[str] = None, map_name: Optional[str] = None
    ) -> gym.Env:
        if map and map_name:
            raise ValueError("Both `map` and `map_name` can't be defined.")

        if map:
            spec = map.splitlines()
        elif map_name:
            if map_name not in iceworld.MAPS:
                raise ValueError(f"Unknown `map_name`: {map_name}")
            spec = iceworld.MAPS[map_name]
        else:
            raise ValueError("Either `map` or `map_name` must be provided.")
        size, lakes, goals, start = iceworld.parse_map_from_text(spec)
        return iceworld.IceWorld(size=size, lakes=lakes, goals=goals, start=start)

    def make_gridworld(grid: str) -> gym.Env:
        size, cliffs, exits, start = gridworld.parse_grid_from_text(grid.splitlines())
        return gridworld.GridWorld(size=size, cliffs=cliffs, exits=exits, start=start)

    return {
        abcseq.ENV_NAME: abcseq.ABCSeq,
        gridworld.ENV_NAME: make_gridworld,
        randomwalk.ENV_NAME: randomwalk.StateRandomWalk,
        redgreen.ENV_NAME: redgreen.RedGreenSeq,
        towerhanoi.ENV_NAME: towerhanoi.TowerOfHanoi,
        iceworld.ENV_NAME: make_iceworld,
    }


def encode_env(**kwargs: Mapping[str, Any]) -> str:
    """
    Encodes environment into a unique hash.
    """
    keys = []
    values = []
    for key, value in sorted(kwargs.items()):
        keys.append(key)
        values.append(value)
    hash_key = tuple(keys) + tuple(values)
    return hashlib.shake_256(str(hash_key).encode("UTF-8")).hexdigest(KEY_HALF_SIZE)
