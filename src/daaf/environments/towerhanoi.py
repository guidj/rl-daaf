"""
Tower of Hanoi.
https://en.wikipedia.org/wiki/Tower_of_Hanoi

There are three pegs, and a given number of disks - three or four.
The game starts with the disks stacked on the first peg,
sorted by size with the smallest disk on top.
The game ends when all disks are stacked on the last peg,
sorted by size with the smallest disk on top.

Rules:
1. Only one disk can be moved at a time
2. A legal move consitutes moving the top most disk
from its current peg onto another
3. No disk can be placed on top of a smaller disk

Since there are three pegs, there are only six possible
moves: moving the top most disk from peg 1 to either
2 or 3; from peg 2 to either 1 or 3, and so on.

The number of moves required to solve the puzzle is
is 2^{n} - 1, where n is the number of disks.

Code based on https://github.com/xadahiya/toh-gym.
"""

import collections
import copy
import functools
from typing import Any, Callable, List, Mapping, Optional, Sequence, SupportsInt, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from daaf import combinatorics, core
from daaf.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "TowerOfHanoi"
ACTIONS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
NUM_PEGS = 3
OBS_KEY_STATE = "state"
OBS_KEY_NUM_PEGS = "num_pegs"
MIN_DISKS = 1
MAX_DISKS = 9


class TowerOfHanoi(gym.Env[Mapping[str, Any], int]):
    """
    An environment where an agent is meant to follow a pre-defined
    sequence of actions.
    """

    def __init__(self, num_disks: int, render_mode: str = "rgb_array"):
        """
        Args:
            disks: number of disks in the game.
        """
        super().__init__()
        if num_disks < MIN_DISKS or num_disks > MAX_DISKS:
            raise ValueError(f"Disks must be between [1, 9]. Got: {num_disks}")

        self.num_disks = num_disks
        self.num_pegs = NUM_PEGS
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict(
            {
                OBS_KEY_NUM_PEGS: spaces.Box(
                    low=self.num_pegs, high=self.num_pegs, dtype=np.int64
                ),
                OBS_KEY_STATE: spaces.Tuple(
                    [spaces.Discrete(self.num_pegs) for _ in range(self.num_disks)]
                ),
            }
        )

        self.transition: MutableEnvTransition = {}
        num_states = self.num_pegs**self.num_disks
        terminal_state = num_states - 1
        state_id_to_state = create_state_id_to_state_fn(
            num_pegs=self.num_pegs, num_disks=self.num_disks
        )
        for state_id in range(num_states):
            self.transition[state_id] = {}
            # generate state
            state_obs = state_id_to_state(state_id)
            for action in range(len(ACTIONS)):
                self.transition[state_id][action] = []
                actual_next_state_obs, actual_reward = apply_action(
                    observation=state_obs, action=action
                )
                actual_next_state_id = get_state_id(actual_next_state_obs)
                for next_state_id in range(num_states):
                    prob = 1.0 if next_state_id == actual_next_state_id else 0.0
                    reward = (
                        actual_reward if next_state_id == actual_next_state_id else 0.0
                    )
                    terminated = (
                        state_id != next_state_id and next_state_id == terminal_state
                    )
                    self.transition[state_id][action].append(
                        (prob, next_state_id, reward, terminated)
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
        self._observation = state_observation(
            num_pegs=self.num_pegs, state=(0,) * self.num_disks
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
            return np.array(self._observation[OBS_KEY_STATE])
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)
        return self._seed


class TowerOfHanoiMdpDiscretizer(core.MdpDiscretizer):
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
        disks: 3
        starting_state: (0, 0, 0)
        possible_states: (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)... (2, 2, 2)

        S = (0, 0, 0)
            Try to move the second disk to the last peg.
            Nothing changes, since there are disks on top of it.
            reward = -1
        S = (0, 0, 0)
            Move the top disk from the first peg to the second.
            reward = -1
        S = (1, 0, 0)
            Move the top disk from the first peg to the third.
            reward = -1
        S = (1, 2, 0)
        ....
        S = (1, 2, 2)
            Move the top disk from the second peg to the third.
            Terminal state
            reward = -1
        S = (2, 2, 2)
            Try to move the top disk from the last peg to the first.
            Nothing changes.
            reward = 0
    """
    state = observation[OBS_KEY_STATE]
    new_observation = dict(observation)
    source, dest = ACTIONS[action]
    pegs: Mapping[int, List[int]] = collections.defaultdict(list)
    for idx, peg in enumerate(state):
        pegs[peg].append(idx)

    # In terminal state already, nothing changes
    if is_terminal_state(observation):
        reward = 0.0

    # No disk to move or dest has smaller disk on top
    elif len(pegs[source]) == 0 or (pegs[dest] and pegs[source][0] > pegs[dest][0]):
        reward = -1.0

    # Ok to move
    else:
        reward = -1.0
        # move top disk from source to dest
        new_observation[OBS_KEY_STATE] = (
            state[: pegs[source][0]] + (dest,) + state[pegs[source][0] + 1 :]
        )
    return new_observation, reward


def is_terminal_state(observation: Mapping[str, Any]) -> bool:
    """
    Determines if the given state is terminal.
    """
    for peg in observation[OBS_KEY_STATE]:
        if peg != observation[OBS_KEY_NUM_PEGS] - 1:
            return False
    return True


def create_env_spec(num_disks: int) -> core.EnvSpec:
    """
    Creates an env spec from a config.
    """
    environment = TowerOfHanoi(num_disks=num_disks)
    discretizer = TowerOfHanoiMdpDiscretizer()
    num_states = environment.num_pegs**num_disks
    num_actions = len(ACTIONS)
    return core.EnvSpec(
        name=ENV_NAME,
        level=str(num_disks),
        environment=environment,
        discretizer=discretizer,
        mdp=core.EnvMdp(
            env_desc=core.EnvDesc(num_states=num_states, num_actions=num_actions),
            transition=environment.transition,
        ),
    )


def get_state_id(observation: Mapping[str, Any]) -> int:
    """
    Computes an integer Id that represents that state.
    """
    return combinatorics.sequence_to_integer(
        space_size=observation[OBS_KEY_NUM_PEGS], sequence=observation[OBS_KEY_STATE]
    )


def create_state_id_to_state_fn(
    num_pegs: int, num_disks: int, cache_size: int = 2**7
) -> Callable[[int], Mapping[str, Any]]:
    @functools.lru_cache(maxsize=cache_size)
    def state_id_to_state(state_id: int) -> Mapping[str, Any]:
        """
        Generates a unique Id for a given state.
        """
        return state_observation(
            num_pegs=num_pegs,
            state=combinatorics.interger_to_sequence(
                space_size=num_pegs, sequence_length=num_disks, index=state_id
            ),
        )

    return state_id_to_state


def state_observation(num_pegs: int, state: Sequence[int]) -> Mapping[str, Any]:
    """
    Generates a state observation, given an interger Id and the number
    of disks.

    Args:
        num_peges: the number of pegs in the game.
        state: a sequence defining the placement of disks.

    Returns:
        An observation of the environment.
    """
    return {
        OBS_KEY_NUM_PEGS: num_pegs,
        OBS_KEY_STATE: state,
    }
