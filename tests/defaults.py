import copy
from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from daaf import core
from daaf.core import InitState, ObsType, RenderType, TimeStep


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
    WRONG_MOVE_REWARD = -10.0
    RIGHT_MOVE_REWARD = -1.0

    def __init__(self):
        self.action_space = spaces.Box(low=0, high=1, dtype=np.int64)
        self.observation_space = spaces.Box(low=0, dtype=np.int64)

        # env specific
        self._observation: np.ndarray = np.empty(shape=(0,))
        self._seed: Optional[int] = None

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """
        assert self._observation is not None

        if action == 0:
            new_obs = copy.deepcopy(self._observation)
            reward = self.WRONG_MOVE_REWARD
        elif action == 1:
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
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = np.array(0, np.int64)
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if self.render_mode == "rgb_array":
            return copy.deepcopy(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class SingleStateEnv(gym.Env[Mapping[str, Any], int]):
    """
    An environment that remains in a perpetual state.
    """

    def __init__(self, num_actions: int):
        assert num_actions > 0, "`num_actios` must be positive."
        self.num_actions = num_actions
        self.action_space = spaces.Box(low=0, high=num_actions, dtype=np.int64)
        self.observation_space = spaces.Box(low=0, high=0, dtype=np.int64)

        # env specific
        self._observation: Optional[np.ndarray] = None
        self._seed: Optional[int] = None

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """

        # none
        if not (0 <= action < self.num_actions):
            raise ValueError(f"Unknown action {action}")
        return copy.deepcopy(self._observation), 0.0, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = np.array(0, np.int64)
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if self.render_mode == "rgb_array":
            return copy.deepcopy(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class RoundRobinActionsPolicy(core.PyPolicy):
    """
    Chooses a sequence of actions provided in the constructor, forever.
    """

    def __init__(
        self,
        actions: Sequence[Any],
    ):
        self._counter = 0
        self._actions = actions
        self._iterator = iter(actions)
        self.emit_log_probability = True

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
        Takes the current time step.
        """
        del observation
        if self.emit_log_probability:
            policy_info = {"log_probability": np.array(np.log(1.0), dtype=np.float64)}
        else:
            policy_info = {}

        try:
            action = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._actions)
            action = next(self._iterator)
        return core.PolicyStep(
            action=action,
            state=policy_state,
            info=policy_info,
        )


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
    except ValueError:
        pass
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
