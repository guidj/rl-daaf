"""
Module contains job to run policy evaluation with replay mappers.
"""

import argparse
import dataclasses
import logging
import uuid
from typing import Any, Mapping, Optional, Sequence, Tuple

import ray
import ray.data

import daaf
from daaf.rewardest import estimation

ENV_SPECS = [
    {"name": "ABCSeq", "args": {"length": 7}},
    {"name": "ABCSeq", "args": {"length": 10}},
    {"name": "FrozenLake-v1", "args": {"is_slippery": False, "map_name": "4x4"}},
    {
        "name": "GridWorld",
        "args": {"grid": "oooooooooooo\noooooooooooo\noooooooooooo\nsxxxxxxxxxxg"},
    },
    {
        "name": "RedGreenSeq",
        "args": {
            "cure": ["red", "green", "wait", "green", "red", "red", "green", "wait"]
        },
    },
    {"name": "IceWorld", "args": {"map_name": "4x4"}},
    {"name": "IceWorld", "args": {"map_name": "8x8"}},
    {"name": "TowerOfHanoi", "args": {"num_disks": 4}},
    {"name": "TowerOfHanoi", "args": {"num_disks": 6}},
    {"name": "TowerOfHanoi", "args": {"num_disks": 9}},
]

AGG_REWARD_PERIODS = [2, 4, 6, 8]

EST_ACCURACY = 1e-8


@dataclasses.dataclass(frozen=True)
class EstimationPipelineArgs:
    """
    Program arguments.
    """

    # problem args
    num_runs: int
    max_episodes: int
    output_dir: str
    log_episode_frequency: int
    # ray args
    cluster_uri: Optional[str]


@dataclasses.dataclass(frozen=True)
class EstimationTask:
    uid: str
    spec: Mapping[str, Any]
    run_id: int
    reward_period: int
    accuracy: float
    max_episodes: int
    log_episode_frequency: int


def main(args: EstimationPipelineArgs):
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
            env_specs=ENV_SPECS,
            agg_reward_periods=AGG_REWARD_PERIODS,
            num_runs=args.num_runs,
            max_episodes=args.max_episodes,
            log_episode_frequency=args.log_episode_frequency,
            accuracy=EST_ACCURACY,
        )

        # since ray tracks objectref items
        # we swap the key:value
        futures = [future for _, future in tasks_futures]
        results = []
        unfinished_tasks = futures
        while True:
            finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
            for finished_task in finished_tasks:
                result = ray.get(finished_task)
                results.append(result)

                logging.info(
                    "Tasks left: %d out of %d.",
                    len(unfinished_tasks),
                    len(futures),
                )

            if len(unfinished_tasks) == 0:
                break

        ray.data.from_items(results).write_json(args.output_dir)


def create_tasks(
    env_specs: Sequence[Mapping[str, Any]],
    agg_reward_periods: Sequence[int],
    num_runs: int,
    max_episodes: int,
    log_episode_frequency: int,
    accuracy: float,
) -> Sequence[Tuple[ray.ObjectRef, EstimationTask]]:
    futures = []
    for spec in env_specs:
        for reward_period in agg_reward_periods:
            for run_id in range(num_runs):
                task = EstimationTask(
                    uid=str(uuid.uuid4()),
                    spec=spec,
                    reward_period=reward_period,
                    run_id=run_id,
                    accuracy=accuracy,
                    max_episodes=max_episodes,
                    log_episode_frequency=log_episode_frequency,
                )
                futures.append((task, estimate.remote(task)))
    return futures


@ray.remote
def estimate(task: EstimationTask) -> Mapping[str, Any]:
    """
    Runs evaluation.
    """
    logging.info(
        "Task %s for %s/%d (%s) starting",
        task.uid,
        task.spec["name"],
        task.run_id,
        task.spec["args"],
    )
    result = estimation.estimate_reward(
        spec=task.spec,
        reward_period=task.reward_period,
        accuracy=task.accuracy,
        max_episodes=task.max_episodes,
        logging_steps=task.log_episode_frequency,
    )
    logging.info(
        "Task %s for %s/%d (%s) finished",
        task.uid,
        task.spec["name"],
        task.run_id,
        task.spec["args"],
    )
    return result


def parse_args() -> EstimationPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--max-episodes", type=int, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--log-episode-frequency", type=int, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return EstimationPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
