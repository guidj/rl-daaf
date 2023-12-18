"""
This module implements components for
MDP with Options.
"""

import itertools
import random
from typing import Any, Iterable, Optional

from rlplg import core
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
        self.options_duration = options_duration
        self._options = {}

        for idx, option in enumerate(
            itertools.product(actions, repeat=options_duration)
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
            # Random policy chooses at random
            option_id = random.randint(0, len(self._options) - 1)
            option_step = 0
        else:
            option_id = policy_state["option_id"]
            option_step = policy_state["option_step"] + 1
        action = self._options[option_id][option_step]
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
        return 1.0 / len(self._options)
