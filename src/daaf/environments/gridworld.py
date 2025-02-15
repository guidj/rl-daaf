"""
The objective of the environment is simple: go from a starting point to a goal state.
There can be cliffs, and if the agent falls into one, the reward is -100,
and they go back to the starting position.
The agent can go up, down, left and right.
If an action takes the agent outside the grid, they stay in the same position.
The reward for every action is -1.
"""

import abc
import contextlib
import copy
import io
import os
import os.path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image as image

from daaf import core
from daaf.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "GridWorld"
CLIFF_PENALTY = -100.0
MOVE_PENALTY = -1.0
TERMINAL_REWARD = 0.0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MOVES = ["L", "R", "U", "D"]
COLOR_SILVER = (192, 192, 192)
PATH_BG = "path-bg.png"
CLIFF_BG = "cliff-bg.png"
ACTOR = "actor.png"


class Layers:
    """
    Layers spec.
    """

    agent = 0
    cliff = 1
    exit = 2


class Strings:
    """
    Env keys.
    """

    size = "size"
    agent = "agent"
    cliffs = "cliffs"
    exits = "exits"
    start = "start"


class GridWorld(gym.Env[Mapping[str, Any], int]):
    """
    GridWorld environment.
    """

    _num_layers = 3

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
        exits: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        validate_dimensions(size=size, start=start, cliffs=cliffs, exits=exits)
        validate_starting_grid(start=start, cliffs=cliffs, exits=exits)
        self.render_mode = render_mode
        self._height, self._width = size
        self._size = size
        self._start = start
        self._cliffs = set(cliffs)
        self._exits = set(exits)

        # left, right, up, down
        self.action_space = spaces.Discrete(len(MOVES))
        self.observation_space = spaces.Dict(
            {
                "start": spaces.Tuple(
                    (spaces.Discrete(self._height), spaces.Discrete(self._width))
                ),
                "agent": spaces.Tuple(
                    (spaces.Discrete(self._height), spaces.Discrete(self._width))
                ),
                "cliffs": spaces.Sequence(
                    spaces.Tuple(
                        (spaces.Discrete(self._height), spaces.Discrete(self._width))
                    )
                ),
                "exits": spaces.Sequence(
                    spaces.Tuple(
                        (spaces.Discrete(self._height), spaces.Discrete(self._width))
                    )
                ),
                "size": spaces.Box(
                    low=np.array([self._height - 1, self._width - 1]),
                    high=np.array([self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
            }
        )
        states_mapping_ = states_mapping(
            size=self._size,
            cliffs=tuple(self._cliffs),
        )
        num_states = len(states_mapping_)
        num_actions = len(MOVES)
        self.__reverse_state_mapping = {
            value: key for key, value in states_mapping_.items()
        }
        self.transition: MutableEnvTransition = {}
        for state in range(num_states):
            self.transition[state] = {}
            state_pos = self.__reverse_state_mapping[state]
            for action in range(num_actions):
                self.transition[state][action] = []
                next_obs, reward = apply_action(
                    create_observation(
                        size=self._size,
                        start=self._start,
                        agent=state_pos,
                        cliffs=tuple(self._cliffs),
                        exits=tuple(self._exits),
                    ),
                    action,
                )
                for next_state in range(num_states):
                    next_state_pos = self.__reverse_state_mapping[next_state]
                    prob = 1.0 if next_obs[Strings.agent] == next_state_pos else 0.0
                    actual_reward = (
                        reward if next_obs[Strings.agent] == next_state_pos else 0.0
                    )
                    # transition to an exit
                    terminated = state != next_state and (next_state_pos in self._exits)
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
        if not self._observation:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        next_observation, reward = apply_action(self._observation, action)
        self._observation = next_observation
        finished = self._observation[Strings.agent] in self._exits
        # Note: obs fields are immutable;
        # so shallow copies suffice to prevent tampering with
        # internal state.
        return copy.copy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        # place agent at the start
        self._observation = create_observation(
            size=self._size,
            start=self._start,
            agent=self._start,
            cliffs=tuple(self._cliffs),
            exits=tuple(self._exits),
        )
        return copy.copy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if not self._observation:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return as_grid(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)
        return self._seed


class GridWorldMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
    ):
        _states_mapping = states_mapping(size=size, cliffs=cliffs)
        self.__state_fn = create_state_id_fn(states=_states_mapping)

    def state(self, observation: Mapping[str, Any]) -> int:
        """
        Maps an observation to a state ID.
        """
        return self.__state_fn(observation)

    def action(self, action: SupportsInt) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        return int(action)


class GridWorldRenderer:
    """
    Class for handling rendering of the environment
    """

    metadata = {"render.modes": ["raw", "rgb_array", "human", "ansi"]}

    def __init__(self, sprites_dir: Optional[str]):
        self.sprites: Optional[Sprites] = (
            BlueMoonSprites(sprites_dir) if sprites_dir is not None else None
        )
        self.viewer: Optional[Any] = None
        raise NotImplementedError("Rendering not supported!")

    def render(
        self,
        mode: str,
        observation: np.ndarray,
        last_move: Optional[Any],
        caption: Optional[str],
        sleep: Optional[float] = 0.05,
    ) -> Any:
        """
        Args:
            mode: the rendering mode
            observation: the grid, as a numpy array of shape (W, H, 3)
            last_move: index of the last move, [0, 4)
            caption: string write on rendered Window
            sleep: time between rendering frames
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f"Unsupported mode: {mode}")
        if sleep is not None and sleep > 0:
            time.sleep(sleep)
        if mode == "raw":
            return observation
        elif mode == "rgb_array":
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")
            return observation_as_image(self.sprites, observation, last_move)
        elif mode == "human":
            if self.viewer is None:
                raise RuntimeError(
                    "ImageViewer is undefined. Likely cause: pyglget import failure."
                )
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")

            self.viewer.imshow(
                observation_as_image(self.sprites, observation, last_move)
            )
            if isinstance(caption, str):
                self.viewer.window.set_caption(caption)
            return self.viewer.isopen
        elif mode == "ansi":
            return observation_as_string(observation, last_move)
        else:
            raise RuntimeError(
                f"Unknown mode: {mode}. Exepcted of one: {self.metadata['render.modes']}"
            )

    def close(self):
        """
        Closes the viewer if initialized.
        """
        if self.viewer:
            self.viewer.close()


class Sprites(abc.ABC):
    """
    Interface for sprites.
    """

    @property
    @abc.abstractmethod
    def cliff_sprite(self):
        """
        Returns sprite for cliff.
        """
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_sprite(self):
        """
        Returns sprite for clear path.
        """
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def actor_sprite(self):
        """
        Returns sprites for actor.
        """
        del self
        raise NotImplementedError


class BlueMoonSprites(Sprites):
    """
    Implements strips with blue moon as main character.
    """

    def __init__(self, assets_dir: str):
        self._cliff_sprite = image_as_array(
            image.open(os.path.join(assets_dir, CLIFF_BG))
        )
        self._path_sprite = image_as_array(
            image.open(os.path.join(assets_dir, PATH_BG))
        )
        self._actor_sprite = image_as_array(image.open(os.path.join(assets_dir, ACTOR)))

    @property
    def cliff_sprite(self):
        return self._cliff_sprite

    @property
    def path_sprite(self):
        return self._path_sprite

    @property
    def actor_sprite(self):
        return self._actor_sprite


def create_envspec_from_grid_text(grid: str) -> core.EnvSpec:
    """
    Parses a grid file and create an environment from
    the parameters.
    """
    size, cliffs, exits, start = parse_grid_from_text(grid.splitlines())
    return create_env_spec(size=size, cliffs=cliffs, exits=exits, start=start)


def parse_grid_from_text(grid: Sequence[str]):
    """
    Parses grid from text files.
    """
    cliffs = []
    exits = []
    start = None
    height, width = 0, 0
    for pos_x, line in enumerate(grid):
        row = line.strip()
        width = max(width, len(row))
        for pos_y, elem in enumerate(row):
            if elem.lower() == "x":
                cliffs.append((pos_x, pos_y))
            elif elem.lower() == "g":
                exits.append((pos_x, pos_y))
            elif elem.lower() == "s":
                start = (pos_x, pos_y)
        height += 1
    return (height, width), cliffs, exits, start


def create_env_spec(
    size: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
) -> core.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = GridWorld(size=size, cliffs=cliffs, exits=exits, start=start)
    discretizer = GridWorldMdpDiscretizer(size=size, cliffs=cliffs)
    height, width = size
    num_states = height * width - len(cliffs)
    num_actions = len(MOVES)
    mdp = core.EnvMdp(
        env_desc=core.EnvDesc(num_states=num_states, num_actions=num_actions),
        transition=environment.transition,
    )
    return core.EnvSpec(
        name=ENV_NAME,
        level=core.encode_env((size, sorted(cliffs), sorted(exits), start)),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
    )


def states_mapping(
    size: Tuple[int, int], cliffs: Sequence[Tuple[int, int]]
) -> Mapping[Tuple[int, int], int]:
    """
    Creates a mapping of state IDs to grid positions.

    Args:
       size: the grid size, width x height.
       cliffs: the position of cliffs on the grid.
    Returns:
        A mapping grid positions to state ID.
    """
    states = []
    height, width = size
    for pos_x in range(height):
        for pos_y in range(width):
            state = (pos_x, pos_y)
            if state not in set(cliffs):
                states.append(state)
    return {value: key for key, value in enumerate(sorted(states))}


def apply_action(observation: Mapping[str, Any], action: int) -> Tuple[Any, float]:
    """
    One step transition of the MDP.

    Args:
        observation: the current state.
        action: the agent's decision.

    Returns:
        The reward for the action and new state.
    """

    next_position = _step(observation, action)
    reward = _step_reward(observation, next_position=next_position)
    if next_position in observation[Strings.cliffs]:
        # send back to the beginning
        next_position = observation[Strings.start]
    next_observation = dict(observation)
    next_observation[Strings.agent] = next_position
    return next_observation, reward


def _step(observation: Mapping[str, Any], action: int) -> Tuple[int, int]:
    # If in exit, stay
    if observation[Strings.agent] in observation[Strings.exits]:
        pos: Tuple[int, int] = observation[Strings.agent]
        return pos
    pos_x, pos_y = observation[Strings.agent]
    height, width = observation[Strings.size]
    if action == LEFT:
        pos_y = max(pos_y - 1, 0)
    elif action == RIGHT:
        pos_y = min(pos_y + 1, width - 1)
    elif action == UP:
        pos_x = max(pos_x - 1, 0)
    elif action == DOWN:
        pos_x = min(pos_x + 1, height - 1)
    return (pos_x, pos_y)


def _step_reward(
    observation: Mapping[str, Any], next_position: Tuple[int, int]
) -> float:
    # terminal state (current pos)
    if observation[Strings.agent] in observation[Strings.exits]:
        return TERMINAL_REWARD
    if next_position in observation[Strings.cliffs]:
        return CLIFF_PENALTY
    return MOVE_PENALTY


def create_observation(
    size: Tuple[int, int],
    start: Tuple[int, int],
    agent: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> Mapping[str, Any]:
    """
    Creates an observation representation - a mapping of the layers of the grid,
    and other information.
    """
    return {
        Strings.start: start,
        Strings.agent: agent,
        Strings.cliffs: cliffs,
        Strings.exits: exits,
        Strings.size: size,
    }


def validate_dimensions(
    size: Tuple[int, int],
    start: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies the coordinates are sensible:
      - There are no overlaps between items in different layers.
    """
    height, width = size
    start_x, start_y = start
    for pos_x, pos_y in cliffs:
        if not (0 <= pos_x <= height):
            raise ValueError(
                f"Cliff has invalid coordinates, ({pos_x}, _), limits [0, {height})"
            )
        if not (0 <= pos_y <= width):
            raise ValueError(
                f"Cliff has invalid coordinates, (_, {pos_y}), limits [0, {width})"
            )

    for pos_x, pos_y in exits:
        if not (0 <= pos_x <= height):
            raise ValueError(
                f"Exit has invalid coordinates, ({pos_x}, _), limits [0, {height})"
            )
        if not (0 <= pos_y <= width):
            raise ValueError(
                f"Exit has invalid coordinates, (_, {pos_y}), limits [0, {width})"
            )

    if not (0 <= start_x <= height):
        raise ValueError(
            f"Starting position has invalid coordinates, ({start_x}, _), limits [0, {height})"
        )
    if not (0 <= start_y <= width):
        raise ValueError(
            f"Starting position has invalid coordinates, (_, {start_y}), limits [0, {width})"
        )


def validate_starting_grid(
    start: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies starting grid is sensible - there is no overlap between elements.
    """
    for cliff in cliffs:
        if cliff == start:
            raise ValueError(f"Starting on a cliff: ({cliff})")

        for exit_ in exits:
            if cliff == exit_:
                raise ValueError(f"Exit and cliff overlap: ({cliff})")

            # this only needs to run once really
            if exit_ == start:
                raise ValueError(f"Starting on an exit: ({exit_})")


def create_state_id_fn(
    states: Mapping[Tuple[int, int], int],
) -> Callable[[Mapping[str, Any]], int]:
    """
    Creates a function that returns an integer state ID for a given observation.

    Args:
        states: A dictionary that maps observations to state IDs.
    Returns:
        A callable that takes an observation and returns a state ID.
    """

    def state_id(observation: Mapping[str, Any]) -> int:
        """
        A function that takes an observation and returns a state ID.

        Args:
            observation: a state observation.
        Returns:
            An integer state ID.
        """
        return states[observation[Strings.agent]]

    return state_id


def create_position_from_state_id_fn(
    states: Mapping[Tuple[int, int], int],
) -> Callable[[int], Tuple[int, int]]:
    """
    Creates a function that maps a state ID to a grid (x, y) coordinates.
    """

    rev_state = {value: key for key, value in states.items()}

    def position_from_state_id(state: int) -> Tuple[int, int]:
        """
        Given a state ID, computes the grid coordinates.

        Args:
            state: an integer state ID.

        Returns:
            Grid (x, y) position.
        """
        return rev_state[state]

    return position_from_state_id


def as_grid(observation: Mapping[str, Any]) -> np.ndarray:
    """
    Creates a 3D array representation of the grid, with 1s where
    there is an item of the layer and 0 otherwise.

    Args:
        observation: the state observation.
    Returns:
        A stack of 2D grids, with binary flags to indicate the presence layer elements.
    """

    agent = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    cliff = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    exit_ = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    # Place agent at the start.
    # There is only one (x, y) pair.
    agent[observation[Strings.agent]] = 1
    for pos_x, pos_y in observation[Strings.cliffs]:
        cliff[pos_x, pos_y] = 1
    for pos_x, pos_y in observation[Strings.exits]:
        exit_[pos_x, pos_y] = 1
    return np.stack([agent, cliff, exit_])


def image_as_array(img: image.Image) -> np.ndarray:
    """
    Creates an array to represent the image.
    """
    array = np.array(img)
    if len(array.shape) == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


def observation_as_image(
    sprites: Sprites,
    observation: np.ndarray,
    last_move: Optional[Any],
) -> np.ndarray:
    """
    Converts an observation to an image (matrix).
    """
    _, height, width = observation.shape
    rows = []
    for pos_x in range(height):
        row = []
        for pos_y in range(width):
            pos = position_as_string(observation, pos_x, pos_y, last_move)
            if pos in set(["[X]", "[x̄]"]):
                row.append(sprites.cliff_sprite)
            elif pos in set(["[S]"] + [f"[{move}]" for move in MOVES]):
                row.append(sprites.actor_sprite)
            # TODO: add sprite for exits
            else:
                row.append(sprites.path_sprite)
            # vertical border; same size as sprite
            if pos_y < width - 1:
                row.append(_vborder(size=row[-1].shape[0]))
        rows.append(np.hstack(row))
        # horizontal border
        rows.append(_hborder(size=sum(sprite.shape[1] for sprite in row)))
    return np.vstack(rows)


def _vborder(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("vertical border size must be positve")
    return np.array([[COLOR_SILVER]] * size, dtype=np.uint8)


def _hborder(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("horizontal border size must be positive")
    return np.array([[COLOR_SILVER] * size], dtype=np.uint8)


def observation_as_string(observation: np.ndarray, last_move: Optional[Any]) -> str:
    """
    Converts an observation to a string.
    """
    _, height, width = observation.shape
    out = io.StringIO()
    for pos_x in range(height):
        for pos_y in range(width):
            out.write(position_as_string(observation, pos_x, pos_y, last_move))
        out.write("\n")

    out.write("\n\n")

    with contextlib.closing(out):
        sys.stdout.write(out.getvalue())
        return out.getvalue()


def position_as_string(
    observation: np.ndarray, pos_x: int, pos_y: int, last_move: Optional[Any]
) -> str:
    """
    Given a position:

     - If the agent is in a starting position -
        returns the representation for starting position - S
     - If the agent is on the position and they made a move to get there,
        returns the representation of the move they made: L, R, U, D
     - If there is no agent, and it's a cliff, returns the X
     - If the agent is on the cliff returns x̄ (x-bar)
     - If there is no agent and it's a safe zone, returns " "
     - If there is no agent on an exit, returns E
     - If the agent is on an exit, returns Ē

    """
    if (
        observation[Layers.agent, pos_x, pos_y]
        and observation[Layers.cliff, pos_x, pos_y]
    ):
        return "[x̄]"
    elif (
        observation[Layers.agent, pos_x, pos_y]
        and observation[Layers.exit, pos_x, pos_y]
    ):
        return "[Ē]"
    elif observation[Layers.cliff, pos_x, pos_y] == 1:
        return "[X]"
    elif observation[Layers.exit, pos_x, pos_y] == 1:
        return "[E]"
    elif observation[Layers.agent, pos_x, pos_y] == 1:
        if last_move is not None:
            return f"[{MOVES[last_move]}]"
        return "[S]"
    return "[ ]"
