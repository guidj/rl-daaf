"""
Program arguments.
"""


import copy
import dataclasses
from typing import Any, Mapping, Optional

from daaf import constants, utils


@dataclasses.dataclass(frozen=True)
class ControlArgs:
    """
    Class holds experiment arguments.
    """

    epsilon: float
    alpha: float
    gamma: float


@dataclasses.dataclass(frozen=True)
class DaafArgs:
    """
    Class holds arguments for delayed, aggregated, anonymous feedback
     (DAAF) experiments experiments.

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
class ExperimentArgs:
    """
    Experiment arguments.
    """

    run_id: str
    env_name: str
    env_args: Mapping[str, Any]
    control_args: ControlArgs
    daaf_args: DaafArgs
    output_dir: str
    num_episodes: int
    algorithm: str
    log_episode_frequency: int

    @staticmethod
    def from_flat_dict(args: Mapping[str, Any]) -> "ExperimentArgs":
        """
        Parse task arguments.
        """
        mutable_args = dict(**copy.deepcopy(args))
        control_args = ControlArgs(
            epsilon=mutable_args.pop("control_epsilon"),
            alpha=mutable_args.pop("control_alpha"),
            gamma=mutable_args.pop("control_gamma"),
        )
        daaf_args = DaafArgs(
            reward_period=mutable_args.pop("reward_period"),
            cu_step_mapper=mutable_args.pop("cu_step_mapper"),
            buffer_size=mutable_args.pop("buffer_size"),
            buffer_size_multiplier=mutable_args.pop("buffer_size_multiplier"),
        )
        return ExperimentArgs(
            **mutable_args, control_args=control_args, daaf_args=daaf_args
        )


@dataclasses.dataclass(frozen=True)
class ComputingSpec:
    """
    Computing execution mode specification.
    """

    concurrency: Optional[int]


@dataclasses.dataclass(frozen=True)
class ExperimentRunConfig:
    """
    Experiments definition.
    """

    num_runs: int
    args: ExperimentArgs

    def as_dict(self) -> Mapping[str, Any]:
        """
        Converts the class into dictionary with basic types.
        """
        mapping = copy.deepcopy(self.__dict__)
        mapping["args"] = mapping["args"].__dict__
        mapping["args"]["daaf_args"] = mapping["args"]["daaf_args"].__dict__
        mapping["args"]["control_args"] = mapping["args"]["control_args"].__dict__
        return mapping

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExperimentRunConfig":
        """
        Generates an instance from a serialized mapping generated using `as_dict`.
        """
        # we pop the fields that aren't a part of common.Args first
        _data = dict(**copy.deepcopy(data))
        nested_dataclasses = (
            ("daaf_args", DaafArgs),
            ("control_args", ControlArgs),
        )
        for field_name, clazz in nested_dataclasses:
            _data["args"][field_name] = utils.dataclass_from_dict(
                clazz, _data["args"].pop(field_name)
            )
        _data["args"] = utils.dataclass_from_dict(ExperimentArgs, _data["args"])
        instance: ExperimentRunConfig = utils.dataclass_from_dict(
            ExperimentRunConfig, _data
        )
        return instance
