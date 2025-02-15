"""
This is an episodic task.
  - Action space ~ state space.

The objective of the environment is to sort tokens.
Starting from the first token, the agent is suppose to choose the one that follows it,
and then the one that follows after.
Choosing tokens out of order incurs a negative reward, while choosing tokens
in order incurs the smallest penalty reward.
The player reaches the end of the game when they reach the final token,
regardless of how many they have gone through.

This is general problem of sorting items - the tokens
can represent anything.

Observation:
The agent can choose any token.
The observation is just what the agent has already mastered.

E.g. [A, B, C, D]

Transitions:
  - If an agent chooses a token that is further ahead, they get penalized by the distance
  and the state doesn't change.
  - If they choose a token that lies before, they also get penalized - the entire sequence
  of steps forward until the beginning, and from there to the token they chose.
  The state doesn't change.

The number of possible states is the number of tokens plus one:
E.g. With four tokens (first value is set to 1 to simplify calculations)
  - starting: 0
  - Picked the first token: 1
  - Picked the second token: 2
  - Picked the third token: 3
  - Picked the fourth token: 4 (terminal state)


Notes:
  - If an agent chooses the right action the first time in a given state,
  it won't get to explore other actions in that state.
"""

import copy
from typing import Any, Mapping, Optional, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlplg import core, npsci
from rlplg.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "ABCSeq"
MIN_SEQ_LENGTH = 1
LETTERS = [chr(value) for value in range(ord("A"), ord("Z") + 1)]
NUM_TOKENS = len(LETTERS)
OBS_KEY_POS = "pos"
OBS_KEY_LENGTH = "length"
OBS_KEY_DIST_PENALTY = "distance_penalty"


class ABCSeq(gym.Env[Mapping[str, int], int]):
    """
    Place a sequence of tokens in order.
    The goal is to follow then, from left to right, until the end, selecting
    the current state as an action in the current state.
    """

    metadata = {"render.modes": ["raw"]}

    def __init__(self, length: int, distance_penalty: bool, render_mode: str = "raw"):
        super().__init__()
        self.length = length
        self.distance_penalty = distance_penalty
        self.render_mode = render_mode
        if length > NUM_TOKENS or length < MIN_SEQ_LENGTH:
            raise ValueError(
                f"Length must be between {MIN_SEQ_LENGTH} and {NUM_TOKENS}: {length}"
            )
        self.action_space = spaces.Discrete(self.length)
        self.observation_space = spaces.Dict(
            {
                "length": spaces.Box(low=self.length, high=self.length, dtype=np.int64),
                "distance_penalty": spaces.Discrete(2),
                "pos": spaces.Discrete(self.length + 1),
            }
        )
        self.transition: MutableEnvTransition = {}
        for state in range(self.length + 1):
            self.transition[state] = {}
            state_obs = state_observation(
                state, length=self.length, distance_penalty=self.distance_penalty
            )
            for action in range(self.length):
                self.transition[state][action] = []
                next_state_obs, reward = apply_action(state_obs, action)
                next_state_id = get_state_id(next_state_obs)
                for next_state in range(self.length + 1):
                    prob = 1.0 if next_state == next_state_id else 0.0
                    actual_reward = reward if next_state == next_state_id else 0.0
                    terminated = state != next_state and next_state == self.length
                    self.transition[state][action].append(
                        (prob, next_state, actual_reward, terminated)
                    )

        # env specific
        self._observation: Mapping[str, int] = {}
        self._seed: Optional[int] = None
        self._rng: np.random.Generator = np.random.default_rng()

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.

        """
        new_observation, reward = apply_action(self._observation, action)
        finished = is_terminal_state(new_observation)
        self._observation = new_observation
        return copy.copy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = beginning_state(
            length=self.length, distance_penalty=self.distance_penalty
        )
        return copy.copy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
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


class ABCSeqMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Mapping[str, int]) -> int:
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


def apply_action(
    observation: Mapping[str, int], action: int
) -> Tuple[Mapping[str, int], float]:
    """
    Takes an action in a given state.

    Returns:
        New observation and reward
    """
    return _step_observation(observation, action), _step_reward(observation, action)


def _step_observation(observation: Mapping[str, int], action: int) -> Mapping[str, int]:
    """
    Transitions:
        - If an agent chooses a token that is further ahead, they get penalized by the distance
        and the state doesn't change.
        - If they choose a token that lies before, they also get penalized - the entire sequence
        of steps forward until the beginning, and from there to the letter they chose.
        The state doesn't change.
    """
    new_observation = dict(observation)
    # Check action in within bounds
    if action < observation[OBS_KEY_LENGTH] and observation[OBS_KEY_POS] == action:
        new_observation[OBS_KEY_POS] += 1
    return new_observation


def _step_reward(observation: Mapping[str, int], action: int) -> float:
    """
    One penalty per turn and one for the distance - except
    in the terminal state.
    """
    chosen_step = action + 1
    unplaced_tokens = observation[OBS_KEY_LENGTH] - observation[OBS_KEY_POS]
    # non terminal state
    if unplaced_tokens > 0:
        next_token = observation[OBS_KEY_POS] + 1
        if observation[OBS_KEY_DIST_PENALTY]:
            if chosen_step == next_token:
                distance = 0.0
            elif chosen_step > next_token:
                distance = chosen_step - next_token
            else:
                # distance from the current position to the chosen
                # one, going forward and shifting back to the starting
                # state
                distance = (observation[OBS_KEY_LENGTH] - next_token + 1) + chosen_step
        else:
            distance = 0.0
        turn_penalty = -1.0
    else:
        # terminal state
        distance = 0.0
        turn_penalty = 0.0

    return -distance + turn_penalty


def beginning_state(length: int, distance_penalty: bool) -> Mapping[str, int]:
    """
    Args:
        lenght: number of tokens in sequence.
    Returns:
        Initial observation for a starting game.
    """
    return {
        OBS_KEY_LENGTH: length,
        OBS_KEY_POS: 0,
        OBS_KEY_DIST_PENALTY: int(distance_penalty),
    }


def is_terminal_state(observation: Mapping[str, int]) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    return observation[OBS_KEY_POS] == observation[OBS_KEY_LENGTH]


def create_env_spec(length: int, distance_penalty: bool) -> core.EnvSpec:
    """
    Creates an env spec for ABCSeq.
    """
    environment = ABCSeq(length=length, distance_penalty=distance_penalty)
    discretizer = ABCSeqMdpDiscretizer()
    mdp = core.EnvMdp(
        env_desc=core.EnvDesc(num_states=length + 1, num_actions=length),
        transition=environment.transition,
    )
    return core.EnvSpec(
        name=ENV_NAME,
        level=str(length),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
    )


def get_state_id(observation: Mapping[str, int]) -> int:
    """
    Discretizes an observation to an `int` state Id.
    Returns:
        The int `state` corresponding to the observation.
    """
    return observation[OBS_KEY_POS]


def state_observation(
    state_id: int, length: int, distance_penalty: bool
) -> Mapping[str, int]:
    """
    Given a state ID for an environment, returns an observation.
    """
    if state_id > length:
        raise ValueError(f"State id should be <= length: {state_id} > {length}")
    return {
        OBS_KEY_LENGTH: length,
        OBS_KEY_DIST_PENALTY: distance_penalty,
        OBS_KEY_POS: state_id,
    }


def state_representation(observation: Mapping[str, Any]) -> np.ndarray:
    """
    An array view of the state, where the position of the
    agent is marked with an 1.

    E.g. {
        "pos": 2,
        "length": 4,
    }

    output = [1, 1, 1, 0, 0]
    """
    array = np.zeros(shape=(observation[OBS_KEY_LENGTH] + 1,), dtype=np.int64)
    array[0 : npsci.item(observation[OBS_KEY_POS]) + 1] = 1
    return array
