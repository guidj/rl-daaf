"""
Configuration to generate experiments.
"""

import copy
import dataclasses
import json
import os
import os.path
import time
from typing import Any, Generator, Mapping, Optional, Sequence

from daaf import progargs, utils


@dataclasses.dataclass(frozen=True)
class DaafConfig:
    """
    Configuration for cumulative periodic reward experiments.
    """

    reward_periods: Sequence[int]
    cu_step_mapper: str


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration parameters for an experiment.
    """

    env_name: str
    level: str
    env_args: Mapping[str, Any]
    mdp_stats_path: str
    mdp_stats_num_episodes: int
    daaf_config: DaafConfig
    tags: Sequence[str]


def parse_experiments_config(config_path: str) -> Sequence[ExperimentConfig]:
    """
    Generates experiment configurations for a problem file.

    Yields:
        A set of experiments
    """
    with open(config_path, "r", encoding="UTF-8") as readable:
        configs = json.load(readable)

    experiment_configs = []
    for config in configs:
        for experiment in config["experiments"]:
            for cu_mapping_method in experiment["daaf_config"]["methods"]:
                exp_config_args = copy.deepcopy(experiment)
                daaf_config = DaafConfig(
                    reward_periods=experiment["daaf_config"]["reward_periods"],
                    cu_step_mapper=cu_mapping_method,
                )
                exp_config_args["daaf_config"] = daaf_config
                experiment_configs.append(
                    ExperimentConfig(
                        **exp_config_args,
                        env_name=config["env_name"],
                        tags=config["tags"],
                    )
                )
    return experiment_configs


def create_experiment_runs_from_configs(
    experiment_configs: Sequence[ExperimentConfig],
    num_runs: int,
    num_episodes: int,
    algorithm: str,
    output_dir: str,
    timestamp: Optional[int] = None,
) -> Generator[progargs.ExperimentRunConfig, None, None]:
    """
    Generates experiments for a problem given the parameters (configs).
    Yields:
        Instances of `record.Experiment`.
    """
    now = timestamp or int(time.time())

    for config in experiment_configs:
        subdir = os.path.join(*config.tags)
        task_name = "-".join(config.tags)
        for reward_period in config.daaf_config.reward_periods:
            exp_id = utils.create_task_id(now)
            task_id = f"{task_name}-L{config.level}-P{reward_period}"
            run_id = f"{task_id}-{exp_id}-{config.daaf_config.cu_step_mapper}"

            daaf_args = progargs.DaafArgs(
                reward_period=reward_period,
                cu_step_mapper=config.daaf_config.cu_step_mapper,
                buffer_size=None,
                buffer_size_multiplier=None,
            )
            control_args = progargs.ControlArgs(
                epsilon=0.0,
                alpha=0.1,
                gamma=1.0,
            )
            exp_args = progargs.ExperimentArgs(
                run_id=run_id,
                env_name=config.env_name,
                output_dir=os.path.join(
                    output_dir,
                    subdir,
                    str(now),
                    config.daaf_config.cu_step_mapper,
                    f"L{config.level}-P{reward_period}",
                    exp_id,
                ),
                num_episodes=num_episodes,
                algorithm=algorithm,
                log_episode_frequency=10,
                mdp_stats_path=config.mdp_stats_path,
                mdp_stats_num_episodes=config.mdp_stats_num_episodes,
                daaf_args=daaf_args,
                control_args=control_args,
                env_args=config.env_args,
            )
            yield progargs.ExperimentRunConfig(
                num_runs=num_runs,
                args=exp_args,
            )


def generate_experiments_per_run_configs(
    experiment_run_configs: Sequence[progargs.ExperimentRunConfig],
) -> Generator[progargs.ExperimentArgs, None, None]:
    """
    Given a sequence of experiments, expands them
    based on the number of runs per experiments.
    E.g.
    Input
    A, num_runs=2, key1=value1
    b, num_runs=1, key2=value2

    Output"
    A, key1=value1
    A, key1=value1
    B, key1=value1
    """

    def replace_run_and_output_with_local_values(
        exp_args: progargs.ExperimentArgs, idx: int
    ) -> progargs.ExperimentArgs:
        return dataclasses.replace(
            exp_args,
            run_id=f"{exp_args.run_id}-{str(idx)}",
            output_dir=os.path.join(exp_args.output_dir, str(idx)),
        )

    for experiment in experiment_run_configs:
        runs_params_list = [
            replace_run_and_output_with_local_values(experiment.args, idx)
            for idx in range(experiment.num_runs)
        ]
        # Send a list of jobs for each worker once.
        for runs_split in runs_params_list:
            yield runs_split
