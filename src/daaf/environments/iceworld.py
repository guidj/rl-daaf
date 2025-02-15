"""
The objective of the environment to move from a starting point
to a goal.
There can be lakes, and if the agent falls into one, the reward is -2 * W x H,
and the game ends.
The reward for every other action (including reaching the goal) is -1.

The environment incentivizes moving towards the goal.

The agent can go up, down, left and right.
If an action takes the agent outside the grid, they stay in the same position.

Based on Gynamsium's FrozkenLake environment:
https://gymnasium.farama.org/environments/toy_text/frozen_lake/
"""

import contextlib
import copy
import io
import sys
from typing import Any, Callable, Mapping, Optional, Sequence, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image as image

from rlplg import core
from rlplg.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "IceWorld"
LAKE_PENALTY_MULT = -2.0
MOVE_PENALTY = -1.0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MOVES = ["L", "R", "U", "D"]
MOVE_SYMBOLS = ["←", "→", "↑", "↓"]

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


class Layers:
    """
    Layers spec.
    """

    agent = 0
    lake = 1
    goal = 2


class Strings:
    """
    Env keys.
    """

    size = "size"
    agent = "agent"
    lakes = "lakes"
    goals = "goals"
    start = "start"


class IceWorld(gym.Env[Mapping[str, Any], int]):
    """
    IceWorld environment.
    """

    _num_layers = 3

    def __init__(
        self,
        size: Tuple[int, int],
        lakes: Sequence[Tuple[int, int]],
        goals: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        validate_dimensions(size=size, start=start, lakes=lakes, goals=goals)
        validate_starting_grid(start=start, lakes=lakes, goals=goals)
        self.render_mode = render_mode
        self._height, self._width = size
        self._size = size
        self._start = start
        self._lakes = set(lakes)
        self._goals = set(goals)

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
                "lakes": spaces.Sequence(
                    spaces.Tuple(
                        (spaces.Discrete(self._height), spaces.Discrete(self._width))
                    )
                ),
                "goals": spaces.Sequence(
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
        num_states = self._height * self._width
        num_actions = len(MOVES)
        self.transition: MutableEnvTransition = {}
        for state in range(num_states):
            self.transition[state] = {}
            row, col = divmod(state, self._width)
            for action in range(num_actions):
                self.transition[state][action] = []
                next_obs, reward = apply_action(
                    create_observation(
                        size=self._size,
                        start=self._start,
                        agent=(row, col),
                        lakes=tuple(self._lakes),
                        goals=tuple(self._goals),
                    ),
                    action,
                )
                for next_state in range(num_states):
                    next_state_pos = divmod(next_state, self._width)
                    prob = 1.0 if next_obs[Strings.agent] == next_state_pos else 0.0
                    actual_reward = (
                        reward if next_obs[Strings.agent] == next_state_pos else 0.0
                    )
                    # transition to a goal
                    terminated = state != next_state and (
                        next_state_pos in (self._goals | self._lakes)
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
        if not self._observation:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        next_observation, reward = apply_action(self._observation, action)
        self._observation = next_observation
        finished = self._observation[Strings.agent] in (self._goals | self._lakes)
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
            lakes=tuple(self._lakes),
            goals=tuple(self._goals),
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


class IceWorldMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Mapping[str, Any]) -> int:
        """
        Maps an observation to a state ID.
        """
        pos: Tuple[int, int] = observation[Strings.agent]
        size: Tuple[int, int] = observation[Strings.size]
        row, col = pos
        _, cols = size
        return row * cols + col

    def action(self, action: SupportsInt) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        return int(action)


def create_envspec_from_map(
    map: Optional[str] = None, map_name: Optional[str] = None
) -> core.EnvSpec:
    """
    Parses a map file and create an environment from
    the parameters.
    """
    if map and map_name:
        raise ValueError("Both `map` and `map_name` can't be defined.")

    if map:
        spec = map.splitlines()
    elif map_name:
        if map_name not in MAPS:
            raise ValueError(f"Unknown `map_name`: {map_name}")
        spec = MAPS[map_name]
    else:
        raise ValueError("Either `map` or `map_name` must be provided.")
    size, lakes, goals, start = parse_map_from_text(spec)
    return create_env_spec(size=size, lakes=lakes, goals=goals, start=start)


def parse_map_from_text(map: Sequence[str]):
    """
    Parses map from a list of row
    definitions.
    """
    lakes = []
    goals = []
    start = None
    height, width = 0, 0
    for pos_x, line in enumerate(map):
        row = line.strip()
        width = max(width, len(row))
        for pos_y, elem in enumerate(row):
            if elem.lower() == "h":
                lakes.append((pos_x, pos_y))
            elif elem.lower() == "g":
                goals.append((pos_x, pos_y))
            elif elem.lower() == "s":
                start = (pos_x, pos_y)
        height += 1
    return (height, width), lakes, goals, start


def create_env_spec(
    size: Tuple[int, int],
    lakes: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
) -> core.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = IceWorld(size=size, lakes=lakes, goals=goals, start=start)
    discretizer = IceWorldMdpDiscretizer()
    height, width = size
    num_states = height * width
    num_actions = len(MOVES)
    mdp = core.EnvMdp(
        env_desc=core.EnvDesc(num_states=num_states, num_actions=num_actions),
        transition=environment.transition,
    )
    return core.EnvSpec(
        name=ENV_NAME,
        level=core.encode_env((size, sorted(lakes), sorted(goals), start)),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
    )


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
    next_observation = dict(observation)
    next_observation[Strings.agent] = next_position
    return next_observation, reward


def _step(observation: Mapping[str, Any], action: int) -> Tuple[int, int]:
    # If in goal or lake, stay
    if (observation[Strings.agent] in observation[Strings.goals]) or (
        observation[Strings.agent] in observation[Strings.lakes]
    ):
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
    if (observation[Strings.agent] in observation[Strings.goals]) or (
        observation[Strings.agent] in observation[Strings.lakes]
    ):
        return 0.0
    elif next_position in observation[Strings.lakes]:
        height, width = observation[Strings.size]
        return LAKE_PENALTY_MULT * height * width  # type: ignore
    return MOVE_PENALTY


def create_observation(
    size: Tuple[int, int],
    start: Tuple[int, int],
    agent: Tuple[int, int],
    lakes: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
) -> Mapping[str, Any]:
    """
    Creates an observation representation - a mapping of the layers of the grid,
    and other information.
    """
    return {
        Strings.start: start,
        Strings.agent: agent,
        Strings.lakes: lakes,
        Strings.goals: goals,
        Strings.size: size,
    }


def validate_dimensions(
    size: Tuple[int, int],
    start: Tuple[int, int],
    lakes: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies the coordinates are sensible:
      - There are no overlaps between items in different layers.
    """
    height, width = size
    start_x, start_y = start
    for pos_x, pos_y in lakes:
        if not (0 <= pos_x <= height):
            raise ValueError(
                f"Cliff has invalid coordinates, ({pos_x}, _), limits [0, {height})"
            )
        if not (0 <= pos_y <= width):
            raise ValueError(
                f"Cliff has invalid coordinates, (_, {pos_y}), limits [0, {width})"
            )

    for pos_x, pos_y in goals:
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
    lakes: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies starting grid is sensible - there is no overlap between elements.
    """
    for lake in lakes:
        if lake == start:
            raise ValueError(f"Starting on a lake: ({lake})")

        for goal in goals:
            if lake == goal:
                raise ValueError(f"Goal and lake overlap: ({lake})")

            # this only needs to run once really
            if goal == start:
                raise ValueError(f"Starting on an goal: ({goal})")


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
    lake = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    goal = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    # Place agent at the start.
    # There is only one (x, y) pair.
    agent[observation[Strings.agent]] = 1
    for pos_x, pos_y in observation[Strings.lakes]:
        lake[pos_x, pos_y] = 1
    for pos_x, pos_y in observation[Strings.goals]:
        goal[pos_x, pos_y] = 1
    return np.stack([agent, lake, goal])


def image_as_array(img: image.Image) -> np.ndarray:
    """
    Creates an array to represent the image.
    """
    array = np.array(img)
    if len(array.shape) == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


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
     - If there is no agent, and it's a lake, returns the X
     - If the agent is on the lake returns x̄ (x-bar)
     - If there is no agent and it's a safe zone, returns " "
     - If there is no agent on a goal, returns E
     - If the agent is on a goal, returns Ē

    """
    if (
        observation[Layers.agent, pos_x, pos_y]
        and observation[Layers.lake, pos_x, pos_y]
    ):
        return "[Ħ]"
    elif (
        observation[Layers.agent, pos_x, pos_y]
        and observation[Layers.goal, pos_x, pos_y]
    ):
        return "[Ğ]"
    elif observation[Layers.lake, pos_x, pos_y] == 1:
        return "[H]"
    elif observation[Layers.goal, pos_x, pos_y] == 1:
        return "[G]"
    elif observation[Layers.agent, pos_x, pos_y] == 1:
        if last_move is not None:
            return f"[{MOVES[last_move]}]"
        return "[S]"
    return "[ ]"
