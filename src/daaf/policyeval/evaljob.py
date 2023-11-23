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
from daaf import expconfig, utils
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
    output_dir: str
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
            output_dir=args.output_dir,
            num_tasks=args.num_tasks,
        )

        # since ray tracks objectref items
        # we swap the key:value
        futures = [future for _, (future, _) in tasks_futures.items()]
        futures_experiments = {
            future: experiments for _, (future, experiments) in tasks_futures.items()
        }
        finished, unfinished = ray.wait(futures)
        log_completion(finished, futures_experiments)
        for task in finished:
            logging.info(
                "Completed task %s, %d left out of %d.",
                ray.get(task),
                len(unfinished),
                len(futures),
            )

        while len(unfinished) > 0:
            finished, unfinished = ray.wait(unfinished)
            log_completion(finished, futures_experiments)
            for task in finished:
                logging.info(
                    "Completed task %s, %d left out of %d.",
                    ray.get(task),
                    len(unfinished),
                    len(futures),
                )


def create_tasks(
    envs_path: str,
    config_path: str,
    num_runs: int,
    num_episodes: int,
    output_dir: str,
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
            envs_configs=envs_configs, experiment_configs=experiment_configs
        )
    )

    experiment_tasks = tuple(
        expconfig.generate_tasks_from_experiments_and_run_config(
            run_config=expconfig.RunConfig(
                num_episodes=num_episodes,
                log_episode_frequency=10,
                output_dir=output_dir,
            ),
            experiments=experiments,
            num_runs=num_runs,
        )
    )
    # shuffle tasks to balance workload
    experiment_tasks = random.sample(experiment_tasks, len(experiment_tasks))
    worker_split_tasks = utils.split(items=experiment_tasks, num_partitions=num_tasks)

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
        evaluation.main(experiment_task)
    return group_id


def log_completion(
    finished_tasks: Sequence[ray.ObjectRef],
    tasks_experiments: Mapping[ray.ObjectRef, Sequence[Any]],
):
    """
    Logs completed tasks's configuration, for tracing.
    """
    for task in finished_tasks:
        for experiment in tasks_experiments[task]:
            logging.info("Completed experiment: %s", vars(experiment))


def parse_args() -> EvalPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--envs-path", type=str, required=True)
    arg_parser.add_argument("--config-path", type=str, required=True)
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    arg_parser.add_argument("--num-tasks", type=int, default=1)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return EvalPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
