"""
This module implements components for
MDP with Options.
"""

import itertools
import random
from typing import Any, Optional

from rlplg import core
from rlplg.core import ObsType


class UniformlyRandomCompositeActionPolicy(core.PyPolicy):
    """
    A stateful composition action options policy.
    """

    def __init__(
        self,
        num_actions: int,
        options_duration: int,
        emit_log_probability: bool = False,
    ):
        super().__init__(emit_log_probability=emit_log_probability)
        self.options_duration = options_duration
        self._options = {}

        for idx, option in enumerate(
            itertools.product(range(num_actions), repeat=options_duration)
        ):
            self._options[idx] = option

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """
        del batch_size
        return {"current_step": 0, "current_option": None}

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

        if policy_state and policy_state["current_step"] % self.options_duration == 0:
            # Random policy chooses at random
            option = self._options[random.randint(0, len(self._options))]
            current_step = 0
        else:
            option = policy_state["current_option"]
            current_step = policy_state["current_step"]
        action = option[current_step]
        return core.PolicyStep(
            action=action,
            state={
                "current_step": current_step + 1,
                "current_option": option,
            },
            info={},
        )
