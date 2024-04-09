"""
Configuration to generate experiments.
"""

import dataclasses
import json
import os
import os.path
from typing import Any, Iterator, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from daaf import utils


@dataclasses.dataclass(frozen=True)
class LearningArgs:
    """
    Class holds experiment arguments.
    """

    epsilon: float
    learning_rate: float
    discount_factor: float


@dataclasses.dataclass(frozen=True)
class DaafConfig:
    """
    Configuration for cumulative periodic reward experiments.
    """

    policy_type: str
    traj_mapping_method: str
    algorithm: str
    algorithm_args: Mapping[str, Any]
    reward_period: int
    drop_truncated_feedback_episodes: bool


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """
    Configuration parameters for an experiment.
    """

    name: str
    args: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """
    Configuration for experiment run.
    """

    num_episodes: int
    log_episode_frequency: int
    metrics_last_k_episodes: int
    output_dir: str


@dataclasses.dataclass(frozen=True)
class Experiment:
    """
    Experiments definition.
    """

    env_config: EnvConfig
    daaf_config: DaafConfig
    learning_args: LearningArgs


@dataclasses.dataclass(frozen=True)
class ExperimentTask:
    """
    A single experiment task.
    """

    exp_id: str
    run_id: int
    experiment: Experiment
    run_config: RunConfig
    context: Mapping[str, Any]


def parse_environments(envs_path: str) -> Sequence[EnvConfig]:
    """
    Parse environments from a file.

    Yields:
        A set of environments.
    """
    with open(envs_path, "r", encoding="UTF-8") as readable:
        envs = json.load(readable)

    return [EnvConfig(name=entry["name"], args=entry["args"]) for entry in envs]


def parse_experiment_configs(
    config_path: str,
) -> Sequence[Tuple[DaafConfig, LearningArgs]]:
    """
    Generates experiment configurations for a problem file.

    Yields:
        A set of experiments
    """
    with open(config_path, "r", encoding="UTF-8") as readable:
        df_config = pd.read_csv(
            readable,
            dtype={
                "policy": str,
                "traj_mapper": str,
                "algorithm": str,
                "reward_period": np.int64,
                "drop_truncated_feedback_episodes": np.bool_,
                "discount_factor": np.float64,
                "learning_rate": np.float64,
            },
            converters={"algorithm_args": json.loads},
        )
    configs = []
    for entry in df_config.to_dict(orient="records"):
        # epsilon has a default value - no exploration
        learning_args = LearningArgs(
            epsilon=entry.pop("epsilon", 0.0),
            learning_rate=entry.pop("learning_rate"),
            discount_factor=entry.pop("discount_factor"),
        )
        daaf_config = DaafConfig(**entry)  # type: ignore
        configs.append((daaf_config, learning_args))
    return tuple(configs)


def create_experiments(
    envs_configs: Sequence[EnvConfig],
    experiment_configs: Sequence[Tuple[DaafConfig, LearningArgs]],
) -> Iterator[Experiment]:
    """
    Generates experiments for a problem given the parameters (configs).
    Yields:
        Instances of `record.Experiment`.
    """
    for env_config in envs_configs:
        for daaf_config, learning_args in experiment_configs:
            yield Experiment(
                env_config=env_config,
                daaf_config=daaf_config,
                learning_args=learning_args,
            )


def generate_tasks_from_experiments_context_and_run_config(
    run_config: RunConfig,
    experiments_and_context: Sequence[Tuple[Experiment, Mapping[str, Any]]],
    num_runs: int,
    task_prefix: str,
) -> Iterator[ExperimentTask]:
    """
    Given a sequence of experiments, expands them
    to tasks.
    E.g.
    Input
    A, num_runs=2, key1=value1
    b, num_runs=1, key2=value2

    Output"
    A, key1=value1
    A, key1=value1
    B, key1=value1
    """

    for experiment, context in experiments_and_context:
        exp_id = "-".join(
            [
                utils.create_task_id(task_prefix),
                experiment.env_config.name,
            ]
        )
        for idx in range(num_runs):
            yield ExperimentTask(
                exp_id=exp_id,
                run_id=idx,
                experiment=experiment,
                run_config=dataclasses.replace(
                    run_config,
                    # replace run output with run specific values
                    output_dir=os.path.join(
                        run_config.output_dir,
                        exp_id,
                        f"run{idx}",
                        experiment.daaf_config.traj_mapping_method,
                        f"p{experiment.daaf_config.reward_period}",
                    ),
                ),
                context=context,
            )
