"""
#1 Taking other pills has no impact - other than the right pills
[0] - no medication
[1] - red
[2] - red, green
[3] - red, green, red...
[4] - red, green, red, green
[5] - red, green, red, green, wait
Done
[0, 1, 2, 3, 4, 5, TERMINAL_STATE]

S: integer indicating pos: from 0 to 5
A: integer 3 actions - red, green, wait
(S x A x S): (0, red, 1), (0, green/wait, 0)
R: penalty of -1 for every action
zero for terminal state.
"""

import copy
from typing import Any, Mapping, Optional, Sequence, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from daaf import core
from daaf.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "RedGreenSeq"
RED_PILL = 0
GREEN_PILL = 1
WAIT = 2
ACTIONS = [RED_PILL, GREEN_PILL, WAIT]

ACTION_NAME_MAPPING = {"red": RED_PILL, "green": GREEN_PILL, "wait": WAIT}
OBS_KEY_CURE_SEQUENCE = "cure_sequence"
OBS_KEY_POSITION = "pos"


class RedGreenSeq(gym.Env[Mapping[str, Any], int]):
    """
    An environment where an agent is meant to follow a pre-defined
    sequence of actions.
    """

    def __init__(self, cure: Sequence[str], render_mode: str = "rgb_array"):
        """
        Args:
            cure_sequence: sequence of actions to take to cure a patient.
                Each value is a string, one of `ACTION_NAME_MAPPING`.
        """
        super().__init__()

        if len(cure) < 1:
            raise ValueError(f"Cure sequence should be longer than one: {len(cure)}")

        self.cure_sequence = tuple([ACTION_NAME_MAPPING[action] for action in cure])
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict(
            {
                OBS_KEY_CURE_SEQUENCE: spaces.Sequence(spaces.Discrete(len(ACTIONS))),
                OBS_KEY_POSITION: spaces.Discrete(len(self.cure_sequence) + 1),
            }
        )
        self.transition: MutableEnvTransition = {}
        for state in range(len(self.cure_sequence) + 1):
            self.transition[state] = {}
            state_obs = state_observation(self.cure_sequence, pos=state)
            for action in range(len(ACTIONS)):
                self.transition[state][action] = []
                next_state_obs, reward = apply_action(
                    observation=state_obs, action=action
                )
                next_state_id = get_state_id(next_state_obs)
                for next_state in range(len(self.cure_sequence) + 1):
                    prob = 1.0 if next_state == next_state_id else 0.0
                    actual_reward = reward if next_state == next_state_id else 0.0
                    terminated = state != next_state and next_state == len(
                        self.cure_sequence
                    )
                    self.transition[state][action].append(
                        (prob, next_state, actual_reward, terminated)
                    )

        # env specific
        self._observation: Mapping[str, Any] = {}
        self._seed: Optional[int] = None
        self._rng: np.random.Generator = np.random.default_rng()

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.
        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        new_observation, reward = apply_action(self._observation, action)
        finished = is_terminal_state(new_observation)
        self._observation = new_observation
        return copy.copy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = state_observation(cure_sequence=self.cure_sequence, pos=0)
        return copy.copy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return state_representation(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)
        return self._seed


class RedGreenMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Mapping[str, Any]) -> int:
        """
        Maps an observation to a state ID.
        """
        del self
        return get_state_id(observation)

    def action(self, action: SupportsInt) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        return int(action)


def apply_action(observation: Mapping[str, Any], action: int) -> Tuple[Any, float]:
    """
    Computes a new observation and reward given the current state and action.

    Example:
        cure_sequence: [red, green, red, wait]
        starting_state: 0
        possible_states: [0, 1 (after red), 2 (after green), 3 (after red), 4 (after wait - terminal state)]
        len(possible states) = len(cure_sequence) + 1
        terminal state = len(cure_sequence)
        possible_actions: 0 (red), 1 (green), 2 (wait)


        S = 0 (starting pos), Action 1 (green treatment)
            new state = 0 (no change from starting pos)
            reward = -1
        S = 0 (starting pos), Action 0 (red treatment)
            new state = 1 (has taken the first treatment)
            reward = -1
        S = 1 (first treatment), Action 1 (green treatment)
            new state = 2 (has taken the second treatment)
            reward = -1
        S = 2 (second treatment), Action 0 (red treatment)
            new state = 3 (has taken the third treatment)
            reward = -1
        S = 3 (second treatment), Action 2 (wait treatment)
            new state = 4 (has taken the fourth treatment) - terminal state
            reward = -1

    """
    pos = observation[OBS_KEY_POSITION]
    new_observation = dict(observation)
    if is_terminal_state(observation):
        reward = 0.0
    else:
        if action == observation[OBS_KEY_CURE_SEQUENCE][pos]:
            new_observation[OBS_KEY_POSITION] += 1
        reward = -1.0
    return new_observation, reward


def is_terminal_state(observation: Mapping[str, Any]) -> bool:
    """
    Determines if the agent is in a terminal state.
    """
    terminal_state = len(observation[OBS_KEY_CURE_SEQUENCE])
    return observation[OBS_KEY_POSITION] == terminal_state  # type: ignore


def create_env_spec(cure: Sequence[str]) -> core.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = RedGreenSeq(cure=cure)
    discretizer = RedGreenMdpDiscretizer()
    num_states = len(cure) + 1
    num_actions = len(ACTIONS)
    return core.EnvSpec(
        name=ENV_NAME,
        level=core.encode_env(cure),
        environment=environment,
        discretizer=discretizer,
        mdp=core.EnvMdp(
            env_desc=core.EnvDesc(num_states=num_states, num_actions=num_actions),
            transition=environment.transition,
        ),
    )


def get_state_id(observation: Mapping[str, Any]) -> int:
    """
    Computes an integer ID that represents that state.
    """
    return observation[OBS_KEY_POSITION]  # type: ignore


def state_observation(cure_sequence: Sequence[int], pos: int) -> Mapping[str, Any]:
    """
    Generates a state observation, given an interger ID and the cure sequence.

    Args:
        state_id: An interger ID of the current state.
        cure_sequence: The sequence of actions required to end the game successfully.

    Returns:
        A mapping with the cure sequence and the current state.
    """
    if not 0 <= pos <= len(cure_sequence):
        raise ValueError(
            f"Position must be in range [0, {len(cure_sequence)}]. Got {pos}"
        )
    return {
        OBS_KEY_CURE_SEQUENCE: cure_sequence,
        OBS_KEY_POSITION: pos,
    }


def state_representation(observation: Mapping[str, Any]) -> Sequence[int]:
    """
    An array view of the state, where successful steps are marked
    with 1s and missing steps are marked with a 0.

    E.g.
    observation = {
        "pos": 2,
        "cure_sequnece": [0, 1, 2, 0, 1]
    }

    output = [1, 1, 1, 0, 0]
    """
    pos = observation[OBS_KEY_POSITION]
    return [
        1 if idx < pos else 0 for idx in range(len(observation[OBS_KEY_CURE_SEQUENCE]))
    ]
