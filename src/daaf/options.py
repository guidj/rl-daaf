"""
This module implements components for
MDP with Options.
"""

import functools
import random
from typing import Any, Iterable, Optional

from rlplg import combinatorics, core
from rlplg.core import ObsType
from rlplg.learning.tabular import policies


class UniformlyRandomCompositeActionPolicy(
    core.PyPolicy, policies.SupportsStateActionProbability
):
    """
    A stateful composition action options policy.
    """

    def __init__(
        self,
        actions: Iterable[Any],
        options_duration: int,
        emit_log_probability: bool = False,
    ):
        super().__init__(emit_log_probability=emit_log_probability)
        self.primitive_actions = tuple(actions)
        self.options_duration = options_duration
        self._num_options = len(self.primitive_actions) ** options_duration

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
        del seed
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
        return core.PolicyStep(
            action=action,
            state={
                "option_id": option_id,
                "option_step": option_step,
            },
            info={
                "option_id": option_id,
                "option_terminated": option_step == self.options_duration - 1,
            },
        )

    def state_action_prob(self, state, action) -> float:
        """
        Returns the probability of choosing an arm.
        """
        del state
        del action
        return 1.0 / self._num_options

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
