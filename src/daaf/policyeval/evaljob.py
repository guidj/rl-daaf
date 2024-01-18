"""
Module contains job to run policy evaluation with replay mappers.
"""


import argparse
import dataclasses
import logging
import random
from typing import Any, Mapping, Optional, Sequence, Tuple

import ray

import daaf
from daaf import expconfig, task, utils
from daaf.policyeval import evaluation


@dataclasses.dataclass(frozen=True)
class EvalPipelineArgs:
    """
    Program arguments.
    """

    # problem args
    envs_path: str
    config_path: str
    num_runs: int
    num_episodes: int
    assets_dir: int
    output_dir: str
    log_episode_frequency: int
    task_prefix: str
    # ray args
    cluster_uri: Optional[str]
    num_tasks: int


def main(args: EvalPipelineArgs):
    """
    Program entry point.
    """

    ray_env = {
        "py_modules": [daaf],
    }
    logging.info("Ray environment: %s", ray_env)
    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        tasks_futures = create_tasks(
            envs_path=args.envs_path,
            config_path=args.config_path,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            assets_dir=args.assets_dir,
            output_dir=args.output_dir,
            task_prefix=args.task_prefix,
            log_episode_frequency=args.log_episode_frequency,
            num_tasks=args.num_tasks,
        )

        # since ray tracks objectref items
        # we swap the key:value
        futures = [future for _, (future, _) in tasks_futures.items()]
        futures_experiments = {
            future: experiments for _, (future, experiments) in tasks_futures.items()
        }
        unfinished_tasks = futures
        while True:
            finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
            log_completion(finished_tasks, futures_experiments)
            for finished_task in finished_tasks:
                logging.info(
                    "Completed task %s, %d left out of %d.",
                    ray.get(finished_task),
                    len(unfinished_tasks),
                    len(futures),
                )

            if len(unfinished_tasks) == 0:
                break


def create_tasks(
    envs_path: str,
    config_path: str,
    num_runs: int,
    num_episodes: int,
    assets_dir: str,
    output_dir: str,
    task_prefix: str,
    log_episode_frequency: int,
    num_tasks: int,
) -> Mapping[int, Tuple[ray.ObjectRef, Sequence[Any]]]:
    """
    Runs numerical experiments on policy evaluation.
    """
    envs_configs = expconfig.parse_environments(envs_path=envs_path)
    experiment_configs = expconfig.parse_experiment_configs(
        config_path=config_path,
    )
    experiments = tuple(
        expconfig.create_experiments(
            envs_configs=envs_configs,
            experiment_configs=experiment_configs,
        )
    )
    experiments_and_context = add_experiment_context(experiments, assets_dir=assets_dir)
    experiment_tasks = tuple(
        expconfig.generate_tasks_from_experiments_context_and_run_config(
            run_config=expconfig.RunConfig(
                num_episodes=num_episodes,
                log_episode_frequency=log_episode_frequency,
                output_dir=output_dir,
            ),
            experiments_and_context=experiments_and_context,
            num_runs=num_runs,
            task_prefix=task_prefix,
        )
    )
    # shuffle tasks to balance workload
    experiment_tasks = random.sample(experiment_tasks, len(experiment_tasks))
    worker_split_tasks = utils.partition(
        items=experiment_tasks, num_partitions=num_tasks
    )

    logging.info(
        "Parsed %d DAAF configs and %d environments into %d tasks, split into %d groups",
        len(experiment_configs),
        len(envs_configs),
        len(experiment_tasks),
        len(worker_split_tasks),
    )
    futures = {}
    for group_id, split_tasks in enumerate(worker_split_tasks):
        future = evaluate.remote(group_id, split_tasks)
        futures[group_id] = (future, split_tasks)
    return futures


def add_experiment_context(
    experiments: Sequence[expconfig.Experiment], assets_dir: str
) -> Sequence[Tuple[expconfig.Experiment, Mapping[str, Any]]]:
    """
    Enriches expeirment config with context.
    """
    dyna_prog_specs = []
    for experiment in experiments:
        env_spec = task.create_env_spec(
            problem=experiment.env_config.name, env_args=experiment.env_config.args
        )
        dyna_prog_specs.append(
            (
                env_spec.name,
                env_spec.level,
                experiment.learning_args.discount_factor,
                env_spec.mdp,
            )
        )

    dyna_prog_index = utils.DynaProgStateValueIndex.build_index(
        specs=dyna_prog_specs, path=assets_dir
    )

    experiments_and_context = []
    for experiment, (name, level, gamma, _) in zip(experiments, dyna_prog_specs):
        experiments_and_context.append(
            (
                experiment,
                {
                    "dyna_prog_state_values": dyna_prog_index.get(
                        name, level, gamma
                    ).tolist(),  # so it can be serialized
                },
            )
        )
    return experiments_and_context


@ray.remote
def evaluate(group_id: int, experiment_tasks: Sequence[Any]) -> int:
    """
    Runs evaluation.
    """
    logging.info("Group %d starting", group_id)
    for idx, experiment_task in enumerate(experiment_tasks):
        logging.info(
            "Task %d starting work item %d out of %d",
            group_id,
            idx + 1,
            len(experiment_tasks),
        )
        evaluation.run_fn(experiment_task)
    return group_id


def log_completion(
    finished_tasks: Sequence[ray.ObjectRef],
    tasks_experiments: Mapping[ray.ObjectRef, Sequence[Any]],
):
    """
    Logs completed tasks's configuration, for tracing.
    """
    for finished_task in finished_tasks:
        for experiment in tasks_experiments[finished_task]:
            logging.info("Completed experiment: %s", getattr(experiment, "run_id"))


def parse_args() -> EvalPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--envs-path", type=str, required=True)
    arg_parser.add_argument("--config-path", type=str, required=True)
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--assets-dir", type=str, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--log-episode-frequency", type=int, required=True)
    arg_parser.add_argument("--task-prefix", type=str, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    arg_parser.add_argument("--num-tasks", type=int, default=1)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return EvalPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
