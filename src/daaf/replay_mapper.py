"""
Defines mappers for trajectory events.

These mappers alters the sequence of events that was observed, e.g. including the rewards.
They can process batched transitions, but emit them as single events.
"""

import abc
import copy
import dataclasses
import logging
from typing import Any, Callable, Iterator, List, Optional, Sequence, Set, Tuple

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


class IdentityMapper(TrajMapper):
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
        yield from trajectory


class DaafImputeMissingRewardMapper(TrajMapper):
    """
    Simulates a trajectory of aggregate anonymous feedback:
      - For a set of actions, K, we take their reward, and sum them up.
      - The last event gets the sum of the rewards
      - The others get an imputed value.
    """

    def __init__(self, reward_period: int, impute_value: float):
        """
        Args:
            reward_period: the interval for aggregate rewards.
            impute_value: the reward value to apply in steps where there
                aggregate reward isn't generated.
        """
        if reward_period < 1:
            raise ValueError(f"Reward period must be positive. Got {reward_period}.")
        if np.isnan(impute_value) or np.isinf(impute_value):
            raise ValueError(f"Impute value must be a float. Got {impute_value}.")
        super().__init__()
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
                reward, reward_sum, imputed = reward_sum, 0.0, False
            else:
                reward, imputed = self.impute_value, True
            yield dataclasses.replace(
                traj_step, reward=reward, info={**traj_step.info, "imputed": imputed}
            )


class DaafLsqRewardAttributionMapper(TrajMapper):
    """
    Simulates a trajectory of aggregate anonymous feedback:
      - It accumulates transitions until it reaches a size M
      - Given M transitions, it estimates R(s, a) using the Least Squares method.

    This means that updates can be delayed up until M transitions are observed.
    Because we use Least-Squares, a problem can have multiple solutions.
    And since the reward is only constrained on meeting the aggregate reward conditions,
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
        impute_value: float = 0.0,
    ):
        """
        Args:
            num_states: The number of finite states in the MDP.
            num_actions: The number of finite actions in the MDP.
            reward_period: The interval at which aggregate reward is obsered.
            state_id_fn: A function that maps observations from trajectories
                into a state ID (int).
            action_id_fn: A function that maps actions from the trajectories
                into an action ID (int).
            init_rtable: A table shaped [num_states, num_actions],
                encoding prior beliefs about the rewards for each (S, A) pair.
            buffer_size: The maximum number of trajectories to keep
                in the buffer - each one should contain `reward_period` steps.
            impute_value: Value to use when rewards are missing.

        Note: decay isn't used when summing up the rewards for K steps.
        """
        if reward_period < 2:
            raise ValueError(
                f"Reward period must be greater than 1. Got {reward_period}"
            )

        if init_rtable.shape != (num_states, num_actions):
            raise ValueError(
                f"""Tensor initial_rtable must have shape[{num_states},{num_actions}].
                Got [{init_rtable}]."""
            )
        super().__init__()

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
        self.impute_value = impute_value
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
        reward_sum = 0.0

        for step, traj_step in enumerate(trajectory):
            state_id = self.state_id_fn(traj_step.observation)
            action_id = self.action_id_fn(traj_step.action)
            if self.num_updates == 0:
                state_action_mask[state_id, action_id] += 1
                reward_sum += traj_step.reward
                if (step + 1) % self.reward_period == 0:
                    # reward is the aggretate reward
                    reward = reward_sum
                    # Grab info about event for reward estimation
                    matrix_entry = np.reshape(state_action_mask, newshape=[-1])
                    self._estimation_buffer.add(matrix_entry, rhs=reward_sum)
                    # reset
                    state_action_mask = np.zeros(
                        shape=(self.num_states, self.num_actions), dtype=np.float32
                    )
                    reward_sum = 0.0
                    # Run estimation at the first possible moment,
                    if self._estimation_buffer.is_full_rank:
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
                            # the computation failed, likely due to the
                            # matix being unsuitable (no solution).
                            logging.debug("Reward estimation failed: %s", err)
                else:
                    # Use impute value before estimation
                    reward = self.impute_value
            else:
                reward = float(self.rtable[state_id, action_id])

            yield dataclasses.replace(traj_step, reward=reward)


class DaafMdpWithOptionsMapper(TrajMapper):
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
        for traj_step in trajectory:
            if current_traj_step is None:
                current_traj_step = traj_step
            reward_sum += traj_step.reward
            if traj_step.policy_info["option_terminated"]:
                reward, reward_sum = reward_sum, 0.0
                yield dataclasses.replace(
                    current_traj_step,
                    reward=reward,
                    action=current_traj_step.policy_info["option_id"],
                    policy_info={},
                )
                current_traj_step = None

        # options termination did not coincide with end of episode
        if current_traj_step is not None:
            yield dataclasses.replace(
                current_traj_step,
                reward=0.0,
                action=current_traj_step.policy_info["option_id"],
                truncated=True,
                policy_info={},
            )


class DaafNStepTdUpdateMarkMapper(TrajMapper):
    """
    Marks which steps an n-step TD learning
    algorithm can update based on the availability
    of aggregate feedback.

    n-step TD uses the next n steps to sum
    up rewards and then value of the final step.

    In a DAAF setting, that means that only
    steps that lie n-t+1 behind the step
    where feedback is observed can be updated
    with an accurate value - provided the in-between
    steps are imputed with zero.
    """

    def __init__(self, reward_period: int, impute_value: float = 0.0):
        """
        Args:
            reward_period: the interval for cumulative rewards.
        """
        if reward_period < 1:
            raise ValueError(f"Reward period must be positive. Got {reward_period}.")
        super().__init__()
        self.reward_period = reward_period
        self.nstep = reward_period
        self.impute_value = impute_value

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        reward_sum = 0.0
        traj_steps: List[core.TrajectoryStep] = {}
        tau = 0

        for step, traj_step in enumerate(trajectory):
            reward_sum += traj_step.reward
            tau = step - self.nstep + 1
            if tau >= 0 and (step + 1) % self.reward_period == 0:
                traj_steps[tau].info["ok_nstep_tau"] = True
                reward, reward_sum, imputed = reward_sum, 0.0, False
            else:
                reward, imputed = self.impute_value, True

            traj_steps[step] = dataclasses.replace(
                traj_step,
                reward=reward,
                info={**traj_step.info, "imputed": imputed, "ok_nstep_tau": False},
            )
            if tau >= 0:
                yield traj_steps[tau]
                # clear emitted step
                del traj_steps[tau]
        for idx in range(tau + 1, tau + 1 + len(traj_steps)):
            yield traj_steps[idx]


class DaafDropEpisodeWithTruncatedFeedbackMapper(TrajMapper):
    """
    In DAAF, the ending of an episode can coincide
    with feedback.

    When that's not the case, this mapper
    drops the episode, i.e. it emits no steps.

    Note: this mapper does not modify the trajectory.
    """

    def __init__(self, reward_period: int):
        """
        Args:
            reward_period: the interval for aggregate rewards.
        """
        if reward_period < 1:
            raise ValueError(f"Reward period must be positive. Got {reward_period}.")
        super().__init__()
        self.reward_period = reward_period

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        traj_steps = list(trajectory)
        if len(traj_steps) % self.reward_period == 0:
            yield from traj_steps


class CollectReturnsMapper(TrajMapper):
    """
    Tracks trajectory returns internally.
    """

    def __init__(self):
        super().__init__()
        self.__traj_returns = []

    def apply(
        self, trajectory: Iterator[core.TrajectoryStep]
    ) -> Iterator[core.TrajectoryStep]:
        """
        Args:
            trajectory: A iterator of trajectory steps.
        """
        returns = 0.0
        for traj_step in trajectory:
            returns += traj_step.reward
            yield traj_step
        self.__traj_returns.append(returns)

    @property
    def traj_returns(self) -> Sequence[float]:
        return self.__traj_returns[:]


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
                f"""Buffer size is too small for Least Squares estimate.
                 Got {buffer_size}, should be >= {num_factors}"""
            )

        self.buffer_size = buffer_size
        self.num_factors = num_factors
        # pre-allocate arrays
        self._rows = np.zeros(shape=(buffer_size, num_factors), dtype=np.float32)
        self._b = np.zeros(shape=(buffer_size,), dtype=np.float32)
        self._next_pos = 0
        self._additions = 0
        self._factors_tracker: Set[Tuple] = set()
        self._rank_flag = np.zeros(shape=self.num_factors, dtype=np.float32)

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
            current_row_key = tuple((self._rows[self._next_pos] > 0).astype(np.int64))
            if current_row_key in self._factors_tracker:
                self._factors_tracker.remove(current_row_key)
                self._rank_flag -= self._rows[self._next_pos]
            self._factors_tracker.add(row_key)
            self._rank_flag += mask

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
    def is_empty(self) -> bool:
        """
        Returns:
            True if the buffer is empty.
        """
        return self._additions == 0

    @property
    def is_full_rank(self) -> bool:
        return (
            self._additions >= self.num_factors
            and np.sum(self._rank_flag > 0) == self.num_factors
        )


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
