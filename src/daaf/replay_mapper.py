"""
Defines mappers for trajectory events.

These mappers alters the sequence of events that was observed, e.g. including the rewards.
They can process batched transitions, but emit them as single events.
"""


import abc
import copy
import dataclasses
import logging
from typing import Any, Callable, Iterator, List, Optional, Set, Tuple

import numpy as np
from rlplg import core

from daaf import math_ops


class TrajMapper(abc.ABC):
    """
    Base class that provides an interface for modifying trajectory steps.
    Modifies a trajectory.
    """

    @abc.abstractmethod
    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        del self, trajectory
        raise NotImplementedError


class IdentifyMapper(TrajMapper):
    """
    Makes no changes to the a trajectory.
    """

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        for traj_step in trajectory:
            yield traj_step


class AverageRewardMapper(TrajMapper):
    """
    Simulates a trajectory of periodic cumulative rewards:
      - For a set of actions, K, we take their reward, add it up, and divide it equally.
      - Each K action is emitted as is.
    """

    def __init__(self, reward_period: int):
        """
        Args:
            reward_period: the interval for cumulative rewards.
        """
        if reward_period < 1:
            raise ValueError(f"Reward period must be positive. Got {reward_period}.")
        self.reward_period = reward_period

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """

        buffer: List[core.TrajectoryStep] = []
        reward_sum = 0.0
        for step, traj_step in enumerate(trajectory):
            buffer.append(traj_step)
            reward_sum += traj_step.reward
            if (step + 1) % self.reward_period == 0:
                average_reward = reward_sum / self.reward_period
                for buffer_traj_step in buffer:
                    yield dataclasses.replace(buffer_traj_step, reward=average_reward)
                # reset
                buffer.clear()
                reward_sum = 0.0


class ImputeMissingRewardMapper(TrajMapper):
    """
    Simulates a trajectory of periodic cumulative rewards:
      - For a set of actions, K, we take their reward, and sum them up.
      - The last event gets the sum of the rewards
      - The others get an imputed value.
    """

    def __init__(self, reward_period: int, impute_value: float):
        """
        Args:
            reward_period: the interval for cumulative rewards.
            impute_value: the reward value to apply in steps where there
                cumulative reward isn't generated.
        """
        if reward_period < 1:
            raise ValueError(f"Reward period must be positive. Got {reward_period}.")
        if np.isnan(impute_value) or np.isinf(impute_value):
            raise ValueError(f"Impute value must be a float. Got {impute_value}.")

        self.reward_period = reward_period
        self.impute_value = impute_value

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        reward_sum = 0.0

        for step, traj_step in enumerate(trajectory):
            reward_sum += traj_step.reward
            if (step + 1) % self.reward_period == 0:
                reward, reward_sum = reward_sum, 0.0
            else:
                reward = self.impute_value
            yield dataclasses.replace(traj_step, reward=reward)


class LeastSquaresAttributionMapper(TrajMapper):
    """
    Simulates a trajectory of periodic cumulative rewards:
      - It accumulates transitions until it reaches a size M
      - Given M transitions, it estimates R(s, a) using the Least Squares method.

    This means that updates can be delayed up until M transitions are observed.
    Because we use Least-Squares, a problem can have multiple solutions.
    And since the reward is only constrained on meeting the cumulative reward conditions,
    the estimated values might not even correspond to the true rewards.

    We use a decaying learning rate to merge consecutive reward estimates.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_period: int,
        state_id_fn: Callable[[Any], int],
        action_id_fn: Callable[[Any], int],
        init_rtable: np.ndarray,
        buffer_size: int = 2**9,
    ):
        """
        Args:
            num_states: The number of finite states in the MDP.
            num_actions: The number of finite actions in the MDP.
            reward_period: The interval at which cumulative reward is obsered.
            state_id_fn: A function that maps observations from trajectories into a state ID (int).
            action_id_fn: A function that maps actions from the trajectories into an action ID (int).
            init_rtable: A table shaped [num_states, num_actions], encoding prior beliefs about the rewards for each (S, A) pair.
            buffer_size: The maximum number of trajectories to keep in the buffer - each one should contain `reward_period` steps.

        Note: decay isn't used when summing up the rewards for K steps.
        """
        if reward_period < 2:
            raise ValueError(
                f"Reward period must be greater than 1. Got {reward_period}"
            )

        if init_rtable.shape != (num_states, num_actions):
            raise ValueError(
                f"Tensor initial_rtable must have shape [{num_states},{num_actions}]. Got [{init_rtable}]."
            )

        num_factors = num_states * num_actions
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_period = reward_period
        self.state_id_fn = state_id_fn
        self.action_id_fn = action_id_fn
        self.buffer_size = buffer_size
        self.num_updates = 0
        self._estimation_buffer = AbQueueBuffer(
            self.buffer_size, num_factors=num_factors
        )
        self.rtable = copy.deepcopy(init_rtable)

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        state_action_mask = np.zeros(
            shape=(self.num_states, self.num_actions), dtype=np.float32
        )
        rewards = 0.0

        for step, traj_step in enumerate(trajectory):
            state_id = self.state_id_fn(traj_step.observation)
            action_id = self.action_id_fn(traj_step.action)
            state_action_mask[state_id, action_id] += 1
            rewards += traj_step.reward
            # snapshot rtable for the current traj
            rtable_snapshot = copy.deepcopy(self.rtable)

            if self.num_updates == 0 and (step + 1) % self.reward_period == 0:
                # Grab info about event for reward estimation
                matrix_entry = np.reshape(state_action_mask, newshape=[-1])
                self._estimation_buffer.add(matrix_entry, rhs=rewards)
                # reset
                state_action_mask = np.zeros(
                    shape=(self.num_states, self.num_actions), dtype=np.float32
                )
                rewards = 0.0

            # Run estimation at the first possible moment,
            if (
                self.num_updates == 0
                and not self._estimation_buffer.empty
                and math_ops.meets_least_squares_sufficient_conditions(
                    self._estimation_buffer.matrix
                )
            ):
                logging.debug("Estimating rewards with Least-Squares.")
                try:
                    new_rtable = math_ops.solve_least_squares(
                        matrix=self._estimation_buffer.matrix,
                        rhs=self._estimation_buffer.rhs,
                    )
                    new_rtable = np.reshape(
                        new_rtable, newshape=(self.num_states, self.num_actions)
                    )
                    # update the reward estimates by a fraction of the delta
                    # between the currente estimate and the latest.
                    self.rtable = new_rtable
                    self.num_updates += 1

                except ValueError as err:
                    # the computation failed, likely due to the matix being unsuitable (no solution).
                    logging.debug("Reward estimation failed: %s", err)

            yield dataclasses.replace(
                traj_step, reward=rtable_snapshot[state_id, action_id]
            )


class MdpWithOptionsMapper(TrajMapper):
    """
    Simulates a trajectory with an options policy.
    The trajectory generated is that of the parent policy only.

    Trajectory must have been generated by an options policy.
    Policy info in `TrajectoryStep` indicates when an option
    ends.
    """

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        # Use policy info to determine if option has ended.
        reward_sum = 0.0
        current_traj_step: Optional[core.TrajectoryStep] = None
        options_traj_steps = []
        for traj_step in trajectory:
            if current_traj_step is None:
                current_traj_step = traj_step
            reward_sum += traj_step.reward
            if traj_step.policy_info["option_terminated"]:
                reward, reward_sum = reward_sum, 0.0
                options_traj_steps.append(
                    dataclasses.replace(
                        current_traj_step,
                        reward=reward,
                        action=current_traj_step.policy_info["option_id"],
                        policy_info={},
                    )
                )
                current_traj_step = None

        # options termination did not coincide with end of episode
        if options_traj_steps and current_traj_step is not None:
            options_traj_steps[-1] = dataclasses.replace(
                options_traj_steps[-1], truncated=True
            )
        for traj_step in options_traj_steps:
            yield traj_step


class AbQueueBuffer:
    """
    A buffer to store entries for a matrix A and vector b.

    Note: this class pre-allocates the buffer.
    """

    def __init__(self, buffer_size: int, num_factors: int):
        """
        Args:
            buffer_size: the size of the memory buffer for events.
            num_factors: the size of arrays stored in the buffer.
        """
        if buffer_size < num_factors:
            raise ValueError(
                f"Buffer size is too small for Least Squares estimate: {buffer_size}, should be >= {num_factors}"
            )

        self.buffer_size = buffer_size
        self.num_factors = num_factors
        # pre-allocate arrays
        self._rows = np.zeros(shape=(buffer_size, num_factors), dtype=np.float32)
        self._b = np.zeros(shape=(buffer_size,), dtype=np.float32)
        self._next_pos = 0
        self._additions = 0
        self._factors_tracker: Set[Tuple] = set()

    def add(self, row: np.ndarray, rhs: np.ndarray) -> None:
        """
        Note: Only adds rows if they are independent.

        Args:
            row: an array to store in the buffer.
            rhs: the rhs of the of Ax=b to be stored in the buffer.
        """
        if row.shape[0] != self.num_factors:
            raise ValueError(
                f"Expects row of dimension {self.num_factors}, received {len(row)}"
            )

        mask = (row > 0).astype(np.int32)
        row_key = tuple(mask.tolist())
        if row_key not in self._factors_tracker:
            self._factors_tracker.add(row_key)
            self._rows[self._next_pos] = row
            self._b[self._next_pos] = rhs
            # cycle least recent
            self._next_pos = (self._next_pos + 1) % self.buffer_size
            self._additions += 1

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns:
            The buffer as a numpy array. If the buffered isn't filled,
            it returns the values available - which can be an empty array.
        """
        if self._additions >= self.buffer_size:
            return self._rows
        return self._rows[: self._next_pos]

    @property
    def rhs(self) -> np.ndarray:
        """
        Returns:
            The rhs of the Ax=b stored in the buffer. If the buffered isn't filled,
            it returns the values available - which can be an empty array.
        """
        if self._additions >= self.buffer_size:
            return self._b
        return self._b[: self._next_pos]

    @property
    def empty(self) -> bool:
        """
        Returns:
            True if the buffer is empty.
        """
        return self._additions == 0


class Counter:
    """
    A basic counter.
    """

    def __init__(self):
        """
        Initializes the counter.
        Reset has to be called for the counter to increment.
        """
        self._value: Optional[int] = None

    def inc(self) -> None:
        """
        Increments the counter by one.
        """
        if self._value is not None:
            self._value += 1

    @property
    def value(self) -> Optional[int]:
        """
        Returns:
            The value of the counter.
        """
        return self._value

    def reset(self) -> None:
        """
        Resets the counter.
        """
        self._value = 0
