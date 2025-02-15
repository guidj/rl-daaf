import argparse
import dataclasses
import logging
import time
from typing import Tuple

import numpy as np
import ray
from daaf import core, envsuite
from daaf.learning.tabular import dynamicprog, policies

from daaf import utils

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
    {"name": "TowerOfHanoi", "args": {"num_disks": 5}},
]

DISCOUNTS = [1.0, 0.99, 0.9]

EST_ACCURACY = 1e-8


@dataclasses.dataclass(frozen=True)
class Args:
    assets_dir: str


@dataclasses.dataclass(frozen=True)
class Task:
    env_spec: core.EnvSpec
    discount: float


def main(args: Args):
    """
    Run dynamic programming estimation.
    """
    futures = []
    for spec in ENV_SPECS:
        for discount in DISCOUNTS:
            env_spec = envsuite.load(spec["name"], **spec["args"])
            future = run_dynaprog.remote(
                Task(
                    env_spec=env_spec,
                    discount=discount,
                )
            )
            futures.append(future)

    results = {}
    unfinished_tasks = futures
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        for finished_task in finished_tasks:
            result: Tuple[Task, np.ndarray] = ray.get(finished_task)
            task, state_values = result
            key = (task.env_spec.name, task.env_spec.level, task.discount)
            results[key] = state_values

            logging.info(
                "Tasks left: %d out of %d.",
                len(unfinished_tasks),
                len(futures),
            )

        if len(unfinished_tasks) == 0:
            break

    utils.DynaProgStateValueIndex._export_index(
        path=args.assets_dir, state_value_mapping=results
    )


@ray.remote
def run_dynaprog(task: Task) -> Tuple[Task, np.ndarray]:
    logging.info(
        "Running dynamic programming for %s/%s, with gamma=%f",
        task.env_spec.name,
        task.env_spec.level,
        task.discount,
    )
    start = time.time()
    policy = policies.PyRandomPolicy(num_actions=task.env_spec.mdp.env_desc.num_actions)
    state_values = dynamicprog.iterative_policy_evaluation(
        mdp=task.env_spec.mdp, policy=policy, gamma=task.discount, accuracy=EST_ACCURACY
    )
    end = time.time()
    logging.info(
        "Completed dynamic programming for %s/%s, with gamma=%f in %ds",
        task.env_spec.name,
        task.env_spec.level,
        task.discount,
        int(end - start),
    )
    return (task, state_values)


def parse_args() -> Args:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--assets-dir", type=str, required=True)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return Args(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
