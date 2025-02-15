"""
This module contains the definition of the State-Random-Walk problem.

From Barto, Sutton, p.125

"...all episodes start in the center state, C,then proceed either
left or right by one state on each step, with equal probability.
Episodes terminate either on the extreme left or the extreme right.
When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero."

A distiction is that we introduce agent actions - going left or right.
Combined with a random policy, it should produce the same effect.
"""

import copy
import math
from typing import Any, Mapping, Optional, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from daaf import core
from daaf.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "StateRandomWalk"
GO_LEFT = 0
GO_RIGHT = 1
ACTIONS = [GO_LEFT, GO_RIGHT]
RIGHT_REWARD = 1
LEFT_REWARD = 0
STEP_REWARD = 0
OBS_KEY_POS = "pos"
OBS_KEY_STEPS = "steps"
OBS_KEY_LEFT_END_REWARD = "left_end_reward"
OBS_KEY_RIGHT_END_REWARD = "right_end_reward"
OBS_KEY_STEP_REWARD = "step_reward"


class StateRandomWalk(gym.Env[Mapping[str, Any], int]):
    """
    An environment where an agent is meant to go right, until the end.
    The environment can terminate on the last left or right states.
    By default, only the right direction yields a positive reward.

    Example:
    Steps: 5
    The starting state is the middle state, i.e. ceil(steps / 2) (without shifting by zero-index, since zero is a terminal state)
    Starting state = ceil(5 / 2) = 3
    Terminal states: 0 and 6.
    """

    def __init__(
        self,
        steps: int,
        left_end_reward: float = LEFT_REWARD,
        right_end_reward: float = RIGHT_REWARD,
        step_reward: float = STEP_REWARD,
        render_mode: str = "rgb_array",
    ):
        """
        Args:
            steps: number of states. The left and right ends are terminal states.
            left_reward: reward for terminating on the left.
            right_reward: reward for terminating on the right.
            step_reward: reward for any other move.
        """
        if steps <= 2:
            raise ValueError(f"Steps must be greater than 2. Got {steps}")
        super().__init__()

        self.steps = steps
        self.left_end_reward = left_end_reward
        self.right_end_reward = right_end_reward
        self.step_reward = step_reward
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict(
            {
                OBS_KEY_POS: spaces.Discrete(steps),
                OBS_KEY_STEPS: spaces.Box(low=steps, high=steps, dtype=np.int64),
                OBS_KEY_RIGHT_END_REWARD: spaces.Box(
                    low=right_end_reward, high=right_end_reward, dtype=np.float32
                ),
                OBS_KEY_LEFT_END_REWARD: spaces.Box(
                    low=left_end_reward, high=left_end_reward, dtype=np.float32
                ),
                OBS_KEY_STEP_REWARD: spaces.Box(
                    low=step_reward, high=step_reward, dtype=np.float32
                ),
            }
        )
        self.transition: MutableEnvTransition = {}
        for state in range(self.steps):
            self.transition[state] = {}
            state_obs = state_observation(
                pos=state,
                steps=self.steps,
                left_end_reward=self.left_end_reward,
                right_end_reward=self.right_end_reward,
                step_reward=self.step_reward,
            )
            for action in range(len(ACTIONS)):
                self.transition[state][action] = []
                next_state_obs, reward = apply_action(
                    observation=state_obs, action=action
                )
                next_state_id = get_state_id(next_state_obs)
                for next_state in range(self.steps):
                    prob = 1.0 if next_state == next_state_id else 0.0
                    actual_reward = reward if next_state == next_state_id else 0.0
                    terminated = state != next_state and next_state in set(
                        [0, self.steps - 1]
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
        finished = is_finished(new_observation)
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
        middle = math.floor(self.steps / 2)
        self._observation = state_observation(
            pos=middle - 1 if self.steps % 2 == 0 else middle,
            steps=self.steps,
            left_end_reward=self.left_end_reward,
            right_end_reward=self.right_end_reward,
            step_reward=self.step_reward,
        )
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


class StateRandomWalkMdpDiscretizer(core.MdpDiscretizer):
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
        steps: 5
        starting_state: 2
        possible_states: [0, 4]
        (0 is terminal, 1 is a left step, 3 is a right steps, 4 is a terminal state)
    """
    right_terminal_state = observation[OBS_KEY_STEPS] - 1
    terminal_states = set((0, right_terminal_state))
    new_observation = dict(observation)
    # default reward + for terminal states
    reward = 0.0
    if observation[OBS_KEY_POS] in terminal_states:
        # override step reward in terminal states
        step_reward = 0.0
    else:
        if action == GO_LEFT:
            new_observation[OBS_KEY_POS] -= 1
        elif action == GO_RIGHT:
            new_observation[OBS_KEY_POS] += 1

        step_reward = new_observation[OBS_KEY_STEP_REWARD]
        if new_observation[OBS_KEY_POS] == right_terminal_state:
            reward = new_observation[OBS_KEY_RIGHT_END_REWARD]
        elif new_observation[OBS_KEY_POS] == 0:  # left end
            reward = new_observation[OBS_KEY_LEFT_END_REWARD]
    return new_observation, reward + step_reward


def state_observation(
    pos: int,
    steps: int,
    left_end_reward: float,
    right_end_reward: float,
    step_reward: float,
) -> Mapping[str, Any]:
    """
    Generates the starting state.
    """
    return {
        OBS_KEY_POS: pos,
        OBS_KEY_STEPS: steps,
        OBS_KEY_RIGHT_END_REWARD: right_end_reward,
        OBS_KEY_LEFT_END_REWARD: left_end_reward,
        OBS_KEY_STEP_REWARD: step_reward,
    }


def is_finished(observation: Mapping[str, Any]) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    return observation[OBS_KEY_POS] in set((0, observation[OBS_KEY_STEPS] - 1))


def create_env_spec(
    steps: int,
    left_end_reward: float = LEFT_REWARD,
    right_end_reward: float = RIGHT_REWARD,
    step_reward: float = STEP_REWARD,
) -> core.EnvSpec:
    """
    Creates an env spec from a config.
    """
    environment = StateRandomWalk(
        steps=steps,
        left_end_reward=left_end_reward,
        right_end_reward=right_end_reward,
        step_reward=step_reward,
    )
    discretizer = StateRandomWalkMdpDiscretizer()
    num_states = steps
    num_actions = len(ACTIONS)
    return core.EnvSpec(
        name=ENV_NAME,
        level=core.encode_env((steps, left_end_reward, right_end_reward, step_reward)),
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
    pos: int = observation[OBS_KEY_POS]
    return pos


def state_representation(observation: Mapping[str, Any]) -> np.ndarray:
    """
    An array view of the state, where the tokens
    sorted are marked with a 1.
    There an extra flag for the starting state.

    E.g. {
        "pos": 2,
        "steps": 5,
        "step_reward": 0,
        ...
    }

    output = [0, 0, 1, 0, 0]
    """
    array = np.zeros(shape=(observation[OBS_KEY_STEPS],), dtype=np.int64)
    array[observation[OBS_KEY_POS]] = 1
    return array
