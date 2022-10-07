from copy import deepcopy
import copy
import dataclasses
from typing import Any, Mapping, Optional

from daaf.periodic_reward import constants


@dataclasses.dataclass(frozen=True)
class ControlArgs:
    """
    Class holds experiment arguments.
    """

    epsilon: float
    alpha: float
    gamma: float


@dataclasses.dataclass(frozen=True)
class CPRArgs:
    """
    Class holds arguments for cumulative periodic rewards (CPR) experiments.

    Args:
        reward_period: the period for generating cumulative rewards.
        cu_step_mapper: the method to handle cumulative rewards - generally estimating rewards.
        buffer_size: how many steps to keep in memory for reward estimation, if applicable.
        buffer_size_multiplier: if provided, gets multiplied by (num_states x num_actions) to
            determine the `buffer_size` for the reward estimation, if applicable.

    Raises:

    """

    reward_period: int
    cu_step_mapper: str
    buffer_size: Optional[int]
    buffer_size_multiplier: Optional[int]

    def __post_init__(self):
        if (
            self.cu_step_mapper
            and self.cu_step_mapper not in constants.CU_MAPPER_METHODS
        ):
            raise ValueError(
                f"cu_step_mapper value `{self.cu_step_mapper}` is unknown. Should one of: {constants.CU_MAPPER_METHODS}"
            )

        if self.buffer_size is not None and self.buffer_size_multiplier is not None:
            raise ValueError(
                "Either buffer_size or buffer_size_multiplier can be defined, never both:\n"
                + f"\tbuffer_size={self.buffer_size}, buffer_size_multiplier={self.buffer_size_multiplier}"
            )


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Experiment arguments.
    """

    # TODO: replace unitiated fields with the grouper classes
    # TODO: create constructor that uses the flattend fields
    # TODO: replace use of this library in rngexp

    run_id: str
    problem: str
    output_dir: str
    num_episodes: int
    control_args: ControlArgs
    log_steps: int
    mdp_stats_path: str
    mdp_stats_num_episodes: int
    cpr_args: CPRArgs
    problem_args: Mapping[str, Any]


def parse_args(args: Mapping[str, Any]):
    """
    Parse task arguments.
    """
    mutable_args = dict(**copy.deepcopy(args))
    control_args = ControlArgs(
        epsilon=mutable_args.pop("control_epsilon"),
        alpha=mutable_args.pop("control_alpha"),
        gamma=mutable_args.pop("control_gamma"),
    )
    cpr_args = CPRArgs(
        reward_period=mutable_args.pop("reward_period"),
        cu_step_mapper=mutable_args.pop("cu_step_mapper"),
        buffer_size=mutable_args.pop("buffer_size"),
        buffer_size_multiplier=mutable_args.pop("buffer_size_multiplier"),
    )
    return Args(**mutable_args, control_args=control_args, cpr_args=cpr_args)
