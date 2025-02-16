import copy
from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from daaf import core
from daaf.core import InitState, MutableEnvTransition, ObsType, RenderType, TimeStep

GRID_WIDTH = 5
GRID_HEIGHT = 5
NUM_ACTIONS = 4
NUM_STATES = GRID_HEIGHT * GRID_WIDTH

CLIFF_COLOR = (25, 50, 75)
PATH_COLOR = (50, 75, 25)
ACTOR_COLOR = (75, 25, 50)
EXIT_COLOR = (255, 204, 0)


class CountEnv(gym.Env[np.ndarray, int]):
    """
    Choose between moving forward or stopping, until we reach 3, starting from zero.
        - States: 0, 1, 2, 3 (terminal)
        - Actions: do nothing, next

    If none: value = value, R = -10
    If next: value + 1, R = -1
    If value == 3, action = next - game over, R = 0.

    Q-Table:
      none next
    0:  -10  -1
    1:  -10  -1
    2:  -10  -1
    3:   0   0
    """

    MAX_VALUE = 3
    ACTION_NOTHING = 0
    ACTION_NEXT = 1
    WRONG_MOVE_REWARD = -10.0
    RIGHT_MOVE_REWARD = -1.0

    def __init__(self):
        super().__init__()
        # env specific
        self._observation: np.ndarray = np.empty(shape=(0,))
        self._seed = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.MAX_VALUE + 1)
        self.transition: MutableEnvTransition = {}

        for state in range(self.MAX_VALUE + 1):
            self.transition[state] = {}
            for action in range(2):
                self.transition[state][action] = []
                if state == CountEnv.MAX_VALUE:
                    reward = 0.0
                elif action == CountEnv.ACTION_NOTHING:
                    reward = CountEnv.WRONG_MOVE_REWARD
                else:
                    reward = CountEnv.RIGHT_MOVE_REWARD

                for next_state in range(self.MAX_VALUE + 1):
                    terminated = (
                        state == CountEnv.MAX_VALUE - 1
                        and next_state == CountEnv.MAX_VALUE
                    )
                    prob = (
                        1.0
                        if (
                            (
                                state == CountEnv.MAX_VALUE
                                and np.array_equal(next_state, state)
                            )
                            or (
                                action == CountEnv.ACTION_NEXT
                                and np.array_equal(next_state, state + 1)
                            )
                            or (
                                action == CountEnv.ACTION_NOTHING
                                and np.array_equal(next_state, state)
                            )
                        )
                        else 0.0
                    )
                    self.transition[state][action].append(
                        (prob, next_state, reward, terminated)
                    )

    def reward(self, state: int, action: int, next_state: int) -> float:
        """
        Given a state s, action a, and next state s' returns the expected reward.

        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.
        Returns
            A transition probability.
        """
        del next_state
        if state == CountEnv.MAX_VALUE:
            return 0.0
        elif action == CountEnv.ACTION_NOTHING:
            return CountEnv.WRONG_MOVE_REWARD
        return CountEnv.RIGHT_MOVE_REWARD

    @property
    def env_desc(self) -> core.EnvDesc:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """
        return core.EnvDesc(num_states=self.MAX_VALUE + 1, num_actions=2)

    def step(self, action: int) -> TimeStep:
        """
        Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.
        """

        if np.size(self._observation) == 0:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )

        if action == self.ACTION_NOTHING:
            new_obs = copy.deepcopy(self._observation)
            reward = self.WRONG_MOVE_REWARD
        elif action == self.ACTION_NEXT:
            new_obs = np.array(
                np.min([self._observation + 1, self.MAX_VALUE]), np.int64
            )
            reward = self.RIGHT_MOVE_REWARD
        else:
            raise ValueError(f"Unknown action {action}")

        # terminal state reward override
        if self._observation == self.MAX_VALUE:
            reward = 0.0

        self._observation = new_obs
        finished = np.array_equal(new_obs, self.MAX_VALUE)
        return copy.deepcopy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del seed
        del options
        self._observation = np.array(0, np.int64)
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """Render env"""
        return super().render()


class SingleStateEnv(gym.Env[np.ndarray, int]):
    """
    An environment that remains in a perpetual state.
    """

    def __init__(self, num_actions: int):
        assert num_actions > 0, "`num_actios` must be positive."
        super().__init__()
        self.num_actions = num_actions

        # env specific
        self._observation: np.ndarray = np.empty(shape=(0,))
        self._seed = None

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.
        """

        # none
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Unknown action {action}")
        return copy.deepcopy(self._observation), 0.0, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del seed, options
        self._observation = np.array(0, np.int64)
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """Render env"""
        return super().render()


class RoundRobinActionsPolicy(core.PyPolicy):
    """
    Chooses a sequence of actions provided in the constructor, forever.
    """

    def __init__(
        self,
        actions: Sequence[Any],
    ):
        super().__init__(emit_log_probability=True)
        self._counter = 0
        self._actions = actions
        self._iterator = iter(actions)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        """
        Takes the current time step (which includes the environment feedback)
        """
        del observation, policy_state, seed
        state, info = (), {"log_probability": np.log(0.5)}

        try:
            action = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._actions)
            action = next(self._iterator)
        return core.PolicyStep(np.array(action, dtype=np.int64), state, info)


def identity(value: Any) -> Any:
    """
    Returns the input.
    """
    return value


def item(value: np.ndarray) -> Any:
    """
    Returns the basic value in an array.
    """
    try:
        return value.item()
    except AttributeError:
        return value


def array(*args: Any):
    """
    Collects a sequence of values into an np.ndarray.
    """
    # We use int64 and float64 for all examples/tests
    sample = next(iter(args))
    if isinstance(sample, float):
        return np.array(args, dtype=np.float64)
    if isinstance(sample, int):
        return np.array(args, dtype=np.int64)
    return np.array(args)


def batch(*args: Any):
    """
    Collects a sequence of values into an np.ndarray.
    """
    # We use int32 and float32 for all examples/tests
    sample = next(iter(args))
    if isinstance(sample, float):
        return np.array(args, dtype=np.float32)
    if isinstance(sample, int):
        return np.array(args, dtype=np.int64)
    return np.array(args)


def policy_info(log_probability: float):
    """
    Creates a policy_step.PolicyInfo instance from a given log_probability.
    """
    return {"log_probability": log_probability}
