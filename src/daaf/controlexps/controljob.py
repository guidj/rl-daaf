"""
Module contains job to run policy evaluation with replay mappers.
"""

import argparse
import dataclasses
import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray

from daaf import constants, expconfig, task, utils
from daaf.controlexps import control


@dataclasses.dataclass(frozen=True)
class ControlPipelineArgs:
    """
    Program arguments.
    """

    # problem args
    envs_path: str
    config_path: str
    num_runs: int
    num_episodes: int
    assets_dir: str
    output_dir: str
    log_episode_frequency: int
    metrics_last_k_episodes: int
    task_prefix: str
    # ray args
    cluster_uri: Optional[str]


def main(args: ControlPipelineArgs):
    """
    Program entry point.
    """

    ray_env: Dict[str, Any] = {}
    logging.info("Ray environment: %s", ray_env)
    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        tasks_results_refs = create_tasks(
            envs_path=args.envs_path,
            config_path=args.config_path,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            assets_dir=args.assets_dir,
            output_dir=args.output_dir,
            task_prefix=args.task_prefix,
            log_episode_frequency=args.log_episode_frequency,
            metrics_last_k_episodes=args.metrics_last_k_episodes,
        )

        # since ray tracks objectref items
        # we swap the key:value
        results_refs = [result_ref for _, result_ref in tasks_results_refs]
        unfinished_tasks = results_refs
        while True:
            finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
            for finished_task in finished_tasks:
                logging.info(
                    "Completed task %s, %d left out of %d.",
                    ray.get(finished_task),
                    len(unfinished_tasks),
                    len(results_refs),
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
    metrics_last_k_episodes: int,
) -> Sequence[Tuple[ray.ObjectRef, expconfig.ExperimentTask]]:
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
                metrics_last_k_episodes=metrics_last_k_episodes,
                output_dir=output_dir,
            ),
            experiments_and_context=experiments_and_context,
            num_runs=num_runs,
            task_prefix=task_prefix,
        )
    )
    # shuffle tasks to balance workload
    experiment_tasks = random.sample(
        experiment_tasks,
        len(experiment_tasks),  # type: ignore
    )
    experiment_batches = utils.bundle(
        experiment_tasks, bundle_size=constants.DEFAULT_BATCH_SIZE
    )
    logging.info(
        "Parsed %d DAAF configs and %d environments into %d tasks",
        len(experiment_configs),
        len(envs_configs),
        len(experiment_batches),
    )
    results_refs = []
    for batch in experiment_batches:
        result_ref = evaluate.remote(batch)
        results_refs.append((batch, result_ref))
    return results_refs


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
def evaluate(experiments_batch: Sequence[expconfig.ExperimentTask]) -> Sequence[str]:
    """
    Runs evaluation.
    """
    ids: List[str] = []
    for experiment_task in experiments_batch:
        task_id = f"{experiment_task.exp_id}/{experiment_task.run_id}"
        logging.debug(
            "Experiment %s starting: %s",
            task_id,
            experiment_task,
        )
        control.run_fn(experiment_task)
        ids.append(task_id)
        logging.debug("Experiment %s finished", task_id)
    return ids


def parse_args() -> ControlPipelineArgs:
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
    arg_parser.add_argument("--metrics-last-k-episodes", type=int, required=True)
    arg_parser.add_argument("--task-prefix", type=str, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return ControlPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
