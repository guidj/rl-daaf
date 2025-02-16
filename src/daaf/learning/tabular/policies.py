"""
This module contains implemenation for certain discrete arm
"""

import copy
import dataclasses
import functools
import random
from typing import Any, Callable, Iterable, Optional, Protocol

import numpy as np

from daaf import combinatorics, core
from daaf.core import ObsType


class SupportsStateActionProbability(Protocol):
    """
    An interface for policies that can emit the probability for state action pair.
    """

    def state_action_prob(self, state: Any, action: Any) -> float:
        """
        Given a state and action, it returns a probability
        choosing the action in that state.
        """


class PyRandomPolicy(core.PyPolicy, SupportsStateActionProbability):
    """
    A policy that chooses actions with equal probability.
    """

    def __init__(
        self,
        num_actions: int,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(emit_log_probability=emit_log_probability, seed=seed)
        self._num_actions = num_actions
        self._arms = tuple(range(self._num_actions))
        self._uniform_chance = np.array(1.0) / np.array(num_actions, dtype=np.float32)
        self._rng = random.Random(seed)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        del observation
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        action = self._rng.choice(self._arms)
        if self.emit_log_probability:
            policy_info = {
                "log_probability": np.array(
                    np.log(self._uniform_chance), dtype=np.float32
                )
            }
        else:
            policy_info = {}

        return core.PolicyStep(
            action=action,
            state=policy_state,
            info=policy_info,
        )

    def state_action_prob(self, state, action) -> float:
        """
        Returns the probability of choosing an arm.
        """
        del state
        del action
        return self._uniform_chance.item()  # type: ignore


class PyQGreedyPolicy(core.PyPolicy):
    """
    A Q policy for tabular problems, i.e. finite states and finite actions.
    """

    def __init__(
        self,
        state_id_fn: Callable[[Any], int],
        action_values: np.ndarray,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        """
        The following initializes to base class defaults:
            - policy_state_spec: Any = (),
            - info_spec: Any = (),
            - observation_and_action_constraint_splitter: Optional[types.Splitter] = None
        """

        super().__init__(emit_log_probability=emit_log_probability, seed=seed)

        self._state_id_fn = state_id_fn
        self._state_action_value_table = copy.deepcopy(action_values)
        self._rng = random.Random(seed)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")

        state_id = self._state_id_fn(observation)
        candidate_actions = np.flatnonzero(
            self._state_action_value_table[state_id]
            == np.max(self._state_action_value_table[state_id])
        )
        action = self._rng.choice(candidate_actions)
        if self.emit_log_probability:
            policy_info = {
                "log_probability": np.array(
                    np.log(1.0 / len(candidate_actions)), dtype=np.float32
                )
            }
        else:
            policy_info = {}

        return core.PolicyStep(
            action=action,
            state=policy_state,
            info=policy_info,
        )

    def set_action_values(self, action_values: np.ndarray) -> None:
        """
        Overrides q-table.
        """
        self._state_action_value_table = action_values


class PyEpsilonGreedyPolicy(core.PyPolicy):
    """
    A e-greedy, which randomly chooses actions with e probability,
    and the chooses teh best action otherwise.
    """

    def __init__(
        self,
        policy: PyQGreedyPolicy,
        num_actions: int,
        epsilon: float,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"Epsilon must be between [0, 1]: {epsilon}")
        if emit_log_probability and not hasattr(policy, "emit_log_probability"):
            raise ValueError("Policy has no property `emit_log_probability`")
        if emit_log_probability != getattr(policy, "emit_log_probability"):
            raise ValueError(
                f"""emit_log_probability differs between given policy and constructor argument:
                policy.emit_log_probability={getattr(policy, "emit_log_probability")},
                emit_log_probability={emit_log_probability}""",
            )

        super().__init__(emit_log_probability=emit_log_probability, seed=seed)

        self._num_actions = num_actions
        self.exploit_policy = policy
        self.explore_policy = PyRandomPolicy(
            num_actions=num_actions,
            emit_log_probability=emit_log_probability,
            seed=seed,
        )
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._probs = {
            "explore": self.epsilon / self._num_actions,
            "exploit": self.epsilon / self._num_actions + (1.0 - self.epsilon),
        }

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        # greedy move, find out the greedy arm
        explore = self._rng.random() <= self.epsilon
        policy_: core.PyPolicy = self.explore_policy if explore else self.exploit_policy
        prob = self._probs["explore"] if explore else self._probs["exploit"]
        policy_step_ = policy_.action(observation, policy_state)
        # Update log-prob in _policy_step
        if self.emit_log_probability:
            policy_info = {
                "log_probability": np.array(
                    np.log(prob),
                    np.float32,
                )
            }
            return dataclasses.replace(policy_step_, info=policy_info)
        return policy_step_


class UniformlyRandomCompositeActionPolicy(
    core.PyPolicy, SupportsStateActionProbability
):
    """
    A stateful composition action options policy.
    """

    def __init__(
        self,
        primitive_actions: Iterable[Any],
        options_duration: int,
        emit_log_probability: bool = False,
    ):
        super().__init__(emit_log_probability=emit_log_probability)
        self.primitive_actions = tuple(primitive_actions)
        self.options_duration = options_duration
        self._num_options: int = len(self.primitive_actions) ** options_duration
        self._action_prob: float = 1.0 / self._num_options

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """
        del batch_size
        return {"option_id": None, "option_step": -1}

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        """Implementation of `action`.

        Args:
          observation: An observation.
          policy_state: An Array, or a nested dict, list or tuple of Arrays
            representing the previous policy state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A `PolicyStep` named tuple containing:
            `action`: The policy's chosen action.
            `state`: A policy state to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """
        del observation
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        if policy_state and (
            policy_state["option_step"] + 1 == self.options_duration
            or policy_state["option_id"] is None
        ):
            # Random policy chooses a new option
            option_id = random.randint(0, self._num_options - 1)
            option_step = 0
        else:
            option_id = policy_state["option_id"]
            option_step = policy_state["option_step"] + 1

        option = self._get_option(option_id)
        action = self.primitive_actions[option[option_step]]
        policy_info = {
            "option_id": option_id,
            "option_terminated": option_step == self.options_duration - 1,
        }

        if self.emit_log_probability:
            policy_info["log_probability"] = np.array(  # type: ignore
                np.log(self._action_prob),
                np.float64,
            )
        return core.PolicyStep(
            action=action,
            state={
                "option_id": option_id,
                "option_step": option_step,
            },
            info=policy_info,
        )

    def state_action_prob(self, state, action) -> float:
        """
        Returns the probability of choosing an arm.
        """
        del state
        del action
        return self._action_prob

    @functools.lru_cache(maxsize=64)
    def _get_option(self, option_id: int):
        """
        This method is here to avoid re-generating an option
        from an Id on every call.

        Alternatively, we could have added the option to the state.
        However, we then expose two coupled factors the API of
        this class: option_id and option.
        A caller than has the ability to pass incorrect values.
        Rather than check for that, or simply apply it without
        verifying, we keep the logic of mapping an
        `option_id` to an option internal, and cache the
        computations.
        """
        return combinatorics.interger_to_sequence(
            space_size=len(self.primitive_actions),
            sequence_length=self.options_duration,
            index=option_id,
        )


class OptionsQGreedyPolicy(PyEpsilonGreedyPolicy):
    def __init__(
        self,
        policy: PyQGreedyPolicy,
        primitive_actions: Iterable[Any],
        options_duration: int,
        epsilon: float,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        self.primitive_actions = tuple(primitive_actions)
        self.options_duration = options_duration
        self._num_options = len(self.primitive_actions) ** options_duration

        super().__init__(
            policy,
            num_actions=self._num_options,
            epsilon=epsilon,
            emit_log_probability=emit_log_probability,
            seed=seed,
        )
        self.options_duration = self.options_duration
        self._option_prob = -1.0

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """
        del batch_size
        return {"option_id": None, "option_step": -1}

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        new_option = policy_state and (
            policy_state["option_step"] + 1 == self.options_duration
            or policy_state["option_id"] is None
        )
        if new_option:
            # Random policy chooses a new option
            explore = self._rng.random() <= self.epsilon
            policy_: core.PyPolicy = (
                self.explore_policy if explore else self.exploit_policy
            )
            self._option_prob = (
                self._probs["explore"] if explore else self._probs["exploit"]
            )
            policy_step_ = policy_.action(observation, policy_state)

            option_id = policy_step_.action
            option_step = 0
        else:
            option_id = policy_state["option_id"]
            option_step = policy_state["option_step"] + 1

        option = self._get_option(option_id)
        action = self.primitive_actions[option[option_step]]
        policy_info = {
            "option_id": option_id,
            "option_terminated": option_step == self.options_duration - 1,
        }
        if self.emit_log_probability:
            policy_info["log_probability"] = np.array(
                np.log(self._option_prob),
                np.float64,
            )
        return core.PolicyStep(
            action=action,
            state={
                "option_id": option_id,
                "option_step": option_step,
            },
            info=policy_info,
        )

    @functools.lru_cache(maxsize=64)
    def _get_option(self, option_id: int):
        """
        This method is here to avoid re-generating an option
        from an Id on every call.
        """
        return combinatorics.interger_to_sequence(
            space_size=len(self.primitive_actions),
            sequence_length=self.options_duration,
            index=option_id,
        )
