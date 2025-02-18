"""
This module defines core abstractions.
"""

import abc
import dataclasses
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Tuple,
    Union,
)

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

NestedArray = Union[Mapping, np.ndarray]
TimeStep = Tuple[ObsType, SupportsFloat, bool, bool, Mapping[str, Any]]
InitState = Tuple[ObsType, Mapping[str, Any]]
RenderType = Optional[Union[RenderFrame, Sequence[RenderFrame]]]
StateTransition = Mapping[int, Sequence[Tuple[float, int, float, bool]]]
# Type: Mapping[state, Mapping[action, Sequence[Tuple[prob, next_state, reward, terminated]]]]
EnvTransition = Mapping[int, StateTransition]
MutableStateTransition = Dict[int, List[Tuple[float, int, float, bool]]]
MutableEnvTransition = Dict[int, MutableStateTransition]
MapsToIntId = Callable[[Any], int]


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    """
    Output of a policy's action function.
    Encapsulates the chosen action and policy state.
    """

    action: ActType
    state: Any
    info: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class TrajectoryStep:
    """
    A trajectory step for training RL agents.
    """

    observation: ObsType
    action: ActType
    policy_info: Mapping[str, Any]
    terminated: bool
    truncated: bool
    reward: float
    info: Mapping[str, Any]

    @staticmethod
    def from_transition(
        time_step: TimeStep,
        action_step: PolicyStep,
        next_time_step: TimeStep,
    ) -> "TrajectoryStep":
        """
        Builds a trajectory step given a state and action.
        """
        obs, _, terminated, truncated, _ = time_step
        _, next_reward, _, _, _ = next_time_step

        return TrajectoryStep(
            observation=obs,
            action=action_step.action,
            policy_info=action_step.info,
            terminated=terminated,
            truncated=truncated,
            reward=next_reward,
            info={},
        )


class PyPolicy(abc.ABC):
    """
    Base class for python policies.
    """

    def __init__(
        self,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        self.emit_log_probability = emit_log_probability
        self.seed = seed

    @abc.abstractmethod
    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """

    @abc.abstractmethod
    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> PolicyStep:
        """Implementation of `action`.

        Args:
          observation: An observation.
          policy_state: An Array, or a nested dict, list or tuple of Arrays
            representing the previous policy state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A `PolicyStep` named tuple containing:
            `action`: The policy's chosen action.
            `state`: A policy state to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """


@dataclasses.dataclass(frozen=True)
class EnvSpace:
    """
    Class contains properties of the environment.
    """

    num_states: int
    num_actions: int


class MdpDiscretizer:
    """
    Class contains functions to map state and action spaces
    to discrete range values.
    """

    @abc.abstractmethod
    def state(self, observation: Any) -> int:
        """
        Maps an observation to a state ID.
        """

    @abc.abstractmethod
    def action(self, action: SupportsInt) -> int:
        """
        Maps an agent action to an action ID.
        """


class Mdp:
    """
    Markov Decision Process.
    """

    @property
    @abc.abstractmethod
    def transition(self) -> EnvTransition:
        """
        Returns:
            The mapping of state-action transition.
        """

    @property
    @abc.abstractmethod
    def env_space(self) -> EnvSpace:
        """
        Returns:
            An instance of EnvSpace with properties of the environment.
        """


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Class holds environment variables.
    """

    name: str
    level: str
    environment: gym.Env
    discretizer: MdpDiscretizer
    mdp: Mdp


class EnvMdp(Mdp):
    """
    Environment Dynamics for Gynasium environments that are
    compliant with Toy Text examples.
    """

    def __init__(self, env_space: EnvSpace, transition: EnvTransition):
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
        terminal_states = infer_env_terminal_states(transition)
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
    def env_space(self) -> EnvSpace:
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


class GeneratesEpisode(typing.Protocol):
    def __call__(
        self, environment: gym.Env, policy: PyPolicy, max_steps: Optional[int] = None
    ) -> Iterator[TrajectoryStep]: ...


def infer_env_terminal_states(transition: EnvTransition) -> Set[int]:
    """
    Creates an MDP using transition mapping.

    In some of the environments, there are errors in the implementation
    for terminal states.
    We correct them.
    """
    # collections.defaultdict(lambda: collections.defaultdict(lambda: (0.0, 0.0)))
    # Find terminal states
    terminal_states = set()
    for _, action_transitions in transition.items():
        for _, transitions in action_transitions.items():
            for _, next_state, _, terminated in transitions:
                if terminated is True:
                    terminal_states.add(next_state)
    return terminal_states
