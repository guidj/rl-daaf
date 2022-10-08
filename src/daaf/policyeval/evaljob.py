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
    config_path: str, num_runs: int, num_episodes: int, output_dir: str, num_tasks: int
) -> Mapping[int, Tuple[ray.ObjectRef, Sequence[Any]]]:
    """
    Runs numerical experiments on policy evaluation.
    """
    experiment_configs = expconfig.parse_experiments_config(
        config_path=config_path,
    )
    # expand each run into ift's own task
    experiment_run_configs = tuple(
        expconfig.create_experiment_runs_from_configs(
            experiment_configs=experiment_configs,
            num_runs=num_runs,
            num_episodes=num_episodes,
            output_dir=output_dir,
        )
    )
    experiments = tuple(
        expconfig.generate_experiments_per_run_configs(experiment_run_configs)
    )
    # shuffle tasks to balance workload
    experiments = random.sample(experiments, len(experiments))
    worker_experiments = utils.split(items=experiments, num_partitions=num_tasks)

    logging.info(
        "Parsed %d experiment run configs into %d experiments, split into %d tasks",
        len(experiment_run_configs),
        len(experiments),
        len(worker_experiments),
    )
    futures = {}
    for task_id, experiments in enumerate(worker_experiments):
        future = evaluate.remote(task_id, experiments)
        futures[task_id] = (future, experiments)
    return futures


@ray.remote
def evaluate(task_id: int, run_args: Sequence[Any]) -> int:
    """
    Runs evaluation.
    """
    logging.info("Task %d starting", task_id)
    for idx, experiment_run_config in enumerate(run_args):
        logging.info(
            "Task %d starting work item %d out of %d", task_id, idx + 1, len(run_args)
        )
        evaluation.main(experiment_run_config)
    return task_id


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
