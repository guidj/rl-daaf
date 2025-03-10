"""
Module contains job to run policy evaluation with replay mappers.
"""

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import pathlib
import random
import time
import uuid
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import ray
import ray.data

from daaf import constants, utils
from daaf.rewardest import estimation

ENV_SPECS = [
    {"name": "ABCSeq", "args": {"length": 2, "distance_penalty": False}},
    {"name": "ABCSeq", "args": {"length": 3, "distance_penalty": False}},
    {"name": "ABCSeq", "args": {"length": 7, "distance_penalty": False}},
    {"name": "ABCSeq", "args": {"length": 10, "distance_penalty": False}},
    {"name": "FrozenLake-v1", "args": {"is_slippery": False, "map_name": "4x4"}},
    {
        "name": "GridWorld",
        "args": {"grid": "ooooo\nooxoo\noxooo\nsxxxg"},
    },
    {
        "name": "GridWorld",
        "args": {"grid": "oooooooooooo\noooooooooooo\noooooooooooo\nsxxxxxxxxxxg"},
    },
    {"name": "RedGreenSeq", "args": {"cure": ["red", "green"]}},
    {
        "name": "RedGreenSeq",
        "args": {
            "cure": ["red", "green", "wait", "green", "red", "red", "green", "wait"]
        },
    },
    {"name": "IceWorld", "args": {"map_name": "4x4"}},
    {"name": "IceWorld", "args": {"map_name": "8x8"}},
    {"name": "TowerOfHanoi", "args": {"num_disks": 4}},
]

EST_PLAIN = "plain"
EST_FACTOR_TS = "factor-ts"
EST_PREFILL_BUFFER = "prefill-buffer"
AGG_REWARD_PERIODS = [2, 3, 4, 5, 6, 7, 8, 15]

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
class EstimationRun:
    uid: str
    env_spec: Mapping[str, Any]
    run_id: int
    reward_period: int
    accuracy: float
    max_episodes: int
    log_episode_frequency: int
    method: str


def main(args: EstimationPipelineArgs):
    """
    Program entry point.
    """

    ray_env: Mapping[str, Any] = {}
    logging.info("Ray environment: %s", ray_env)
    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        tasks_and_result_refs = create_tasks(
            env_specs=ENV_SPECS,
            agg_reward_periods=AGG_REWARD_PERIODS,
            num_runs=args.num_runs,
            max_episodes=args.max_episodes,
            log_episode_frequency=args.log_episode_frequency,
            accuracy=EST_ACCURACY,
        )

        # since ray tracks objectref items
        # we swap the key:value
        task_ref_to_tasks = {
            result_ref: tasks for tasks, result_ref in tasks_and_result_refs
        }
        datasets = []
        unfinished_task_refs = list(task_ref_to_tasks.keys())
        while True:
            finished_task_refs, unfinished_task_refs = ray.wait(unfinished_task_refs)
            for finished_task_ref in finished_task_refs:
                datasets.append(ray.get(finished_task_ref))

                logging.info(
                    "Tasks left: %d out of %d.",
                    len(unfinished_task_refs),
                    len(task_ref_to_tasks),
                )

            if len(unfinished_task_refs) == 0:
                break

        if len(datasets) > 0:
            if len(datasets) > 1:
                ds_head, ds_tail = datasets[0], datasets[1:]
                ds_result: ray.data.Dataset = ds_head.union(*ds_tail)
            else:
                ds_result: ray.data.Dataset = datasets[0]  # type: ignore
            ds_output = ds_result.map(serialize)
            ds_output.write_parquet(args.output_dir)


def create_tasks(
    env_specs: Sequence[Mapping[str, Any]],
    agg_reward_periods: Sequence[int],
    num_runs: int,
    max_episodes: int,
    log_episode_frequency: int,
    accuracy: float,
) -> Sequence[Tuple[Sequence[EstimationRun], ray.ObjectRef]]:
    estimation_runs = []
    futures = []
    methods = (EST_PLAIN, EST_FACTOR_TS, EST_PREFILL_BUFFER)
    for env_spec, reward_period, method in itertools.product(
        env_specs, agg_reward_periods, methods
    ):
        uid = str(uuid.uuid4())
        estimation_runs.extend(
            [
                EstimationRun(
                    uid=uid,
                    env_spec=env_spec,
                    reward_period=reward_period,
                    run_id=run_id,
                    accuracy=accuracy,
                    max_episodes=max_episodes,
                    log_episode_frequency=log_episode_frequency,
                    method=method,
                )
                for run_id in range(num_runs)
            ]
        )

    # shuffle to workload
    random.shuffle(estimation_runs)
    # batch tasks
    estimation_run_batches = utils.bundle(
        estimation_runs, bundle_size=constants.DEFAULT_BATCH_SIZE
    )
    for batch in estimation_run_batches:
        futures.append((batch, run_fn.remote(batch)))
    return futures


@ray.remote
def run_fn(estimation_runs: Sequence[EstimationRun]) -> ray.data.Dataset:
    results = []
    for experiment_run in estimation_runs:
        estimation_run_dict = dataclasses.asdict(experiment_run)
        result = estimate(experiment_run)
        result = {"result": result}
        entry = {**result, **estimation_run_dict}
        results.append(entry)
    return ray.data.from_items(results)


def estimate(task: EstimationRun) -> Mapping[str, Any]:
    """
    Reward estimation.
    """
    logging.info(
        "Task %s for %s/%d (%s) starting",
        task.uid,
        task.env_spec["name"],
        task.run_id,
        task.env_spec["args"],
    )
    if task.method == EST_PLAIN:
        factor_terminal_states = False
        prefill_buffer = False
    elif task.method == EST_FACTOR_TS:
        factor_terminal_states = True
        prefill_buffer = False
    elif task.method == EST_PREFILL_BUFFER:
        factor_terminal_states = False
        prefill_buffer = True
    else:
        raise ValueError(f"""
        Unknown method: {task.method}.
        Use one of: {EST_PLAIN}, {EST_FACTOR_TS}, {EST_PREFILL_BUFFER}""")

    result = estimation.estimate_reward(
        spec=task.env_spec,
        reward_period=task.reward_period,
        accuracy=task.accuracy,
        max_episodes=task.max_episodes,
        logging_steps=task.log_episode_frequency,
        factor_terminal_states=factor_terminal_states,
        prefill_buffer=prefill_buffer,
    )
    logging.info(
        "Task %s for %s/%d (%s) finished",
        task.uid,
        task.env_spec["name"],
        task.run_id,
        task.env_spec["args"],
    )
    return result


def serialize(example: Mapping[str, Any]) -> Mapping[str, Any]:
    def go(key: str, element: Any):
        if key == "args" and isinstance(element, Mapping):
            return json.dumps(element)
        elif isinstance(element, Mapping):
            return {skey: go(skey, svalue) for skey, svalue in element.items()}
        elif isinstance(element, np.ndarray):
            return element.flatten()
        return copy.deepcopy(element)

    return {key: go(key, value) for key, value in example.items()}


def parse_args() -> EstimationPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, default=3)
    arg_parser.add_argument("--max-episodes", type=int, default=2500)
    arg_parser.add_argument(
        "--output-dir",
        type=str,
        default=pathlib.Path.home()
        / "fs/daaf/exp/reward-estjob/logs"
        / str(int(time.time())),
    )
    arg_parser.add_argument("--log-episode-frequency", type=int, default=1)

    # arg_parser.add_argument("--num-runs", type=int, required=True)
    # arg_parser.add_argument("--max-episodes", type=int, required=True)
    # arg_parser.add_argument("--output-dir", type=str, required=True)
    # arg_parser.add_argument("--log-episode-frequency", type=int, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return EstimationPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
