"""
Utilities, helpers.
"""


import contextlib
import dataclasses
import json
import logging
import math
import os.path
import types
import uuid
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from rlplg import core
from rlplg.learning.tabular import dynamicprog, policies

STATE_VALUE_FN_FILENAME = "state_action_value_index.json"


class ExperimentLogger(contextlib.AbstractContextManager):
    """
    Logs info for an experiment for given episodes.
    """

    LOG_FILE_NAME = "experiment-logs.jsonl"
    PARAM_FILE_NAME = "experiment-params.json"

    def __init__(
        self,
        log_dir: str,
        exp_id: str,
        run_id: int,
        params: Mapping[str, Union[int, float, str]],
    ):
        self.log_file = os.path.join(log_dir, self.LOG_FILE_NAME)
        self.param_file = os.path.join(log_dir, self.PARAM_FILE_NAME)
        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)

        with tf.io.gfile.GFile(self.param_file, "w") as writer:
            writer.write(json.dumps(dict(params, exp_id=exp_id, run_id=run_id)))

        self._writer: Optional[tf.io.gfile.GFile] = None

    def open(self) -> None:
        """
        Opens the log file for writing.
        """
        self._writer = tf.io.gfile.GFile(self.log_file, "w")

    def close(self) -> None:
        """
        Closes the log file.
        """
        if self._writer is None:
            raise RuntimeError("File is not opened")
        self._writer.close()

    def __enter__(self) -> "ExperimentLogger":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self.close()
        super().__exit__(exc_type, exc_value, traceback)

    def log(
        self,
        episode: int,
        steps: int,
        returns: float,
        info: Optional[Mapping[str, Any]] = None,
    ):
        """
        Logs an experiment entry for an episode.
        """
        entry = {
            "episode": episode,
            "steps": steps,
            "returns": returns,
            "info": info if info is not None else {},
        }

        if self._writer is None:
            raise RuntimeError("File is not opened")
        self._writer.write(f"{json.dumps(entry)}\n")


class DynaProgStateValueIndex:
    """
    Index for state value functions.
    """

    def __init__(
        self, state_value_mapping: Mapping[Tuple[str, str, float], np.ndarray]
    ):
        self.state_value_mapping = state_value_mapping

    def get(self, name: str, level: str, gamma: float) -> np.ndarray:
        """
        Get value estimates.
        """
        key = (name, level, gamma)
        return self.state_value_mapping[key]

    @classmethod
    def build_index(
        cls, specs: Sequence[Tuple[str, str, float, core.Mdp]], path: str
    ) -> "DynaProgStateValueIndex":
        """
        Builds the cache for given configurations.
        This function does not override existing
        entries.
        """
        state_value_mapping: Mapping[Tuple[str, str, float], np.ndarray] = {}
        state_value_mapping = DynaProgStateValueIndex._parse_index(path)
        for name, level, gamma, mdp in specs:
            key = (name, level, gamma)
            if key not in state_value_mapping:
                logging.info(
                    "Solving dynamic programming for %s/%s, discount=%f",
                    name,
                    level,
                    gamma,
                )
                state_value_mapping[key], _ = dynamic_prog_estimation(
                    mdp=mdp, gamma=gamma
                )
        # overrides initial index, if it existed
        cls._export_index(path=path, state_value_mapping=state_value_mapping)
        logging.info("Dynamic programming index updated at %s", path)
        return DynaProgStateValueIndex(state_value_mapping)

    @classmethod
    def load_index_from_file(
        cls,
        path: str,
    ) -> "DynaProgStateValueIndex":
        """
        Loads the index from a file.
        """
        state_values = cls._parse_index(path)
        return DynaProgStateValueIndex(state_values)

    @staticmethod
    def _parse_index(path: str):
        file_path = os.path.join(path, STATE_VALUE_FN_FILENAME)
        logging.info("Loading dynamic programming index from %s", file_path)
        state_values: Mapping[Tuple[str, str, float], np.ndarray] = {}
        with tf.io.gfile.GFile(file_path, "r") as readable:
            for line in readable:
                row = json.loads(line)
                key = (row["env_name"], row["level"], float(row["gamma"]))
                state_values[key] = np.array(row["state_values"], dtype=np.float64)
        return state_values

    @staticmethod
    def _export_index(
        path: str, state_value_mapping: Mapping[Tuple[str, str, float], np.ndarray]
    ) -> None:
        """
        Export index to a file.
        """
        if not tf.io.gfile.exists(path):
            tf.io.gfile.makedirs(path)
        file_path = os.path.join(path, STATE_VALUE_FN_FILENAME)
        with tf.io.gfile.GFile(file_path, "w") as writable:
            for (env_name, level, gamma), state_values in state_value_mapping.items():
                row = {
                    "env_name": env_name,
                    "level": level,
                    "gamma": gamma,
                    "state_values": state_values.tolist(),
                }
                writable.write("".join([json.dumps(row), "\n"]))


def dynamic_prog_estimation(
    mdp: core.Mdp, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs dynamic programming on an MDP to generate state-value and action-value
    functions.

    Args:
        env_spec: environment specification.
        mdp: Markov Decison Process dynamics
        gamma: the discount factor.

    Returns:
        A tuple of state-value and action-value estimates.
    """
    observable_random_policy = policies.PyRandomPolicy(
        num_actions=mdp.env_desc.num_actions,
    )
    state_values = dynamicprog.iterative_policy_evaluation(
        mdp=mdp,
        policy=observable_random_policy,
        gamma=gamma,
    )
    logging.debug("State value V(s):\n%s", state_values)
    action_values = dynamicprog.action_values_from_state_values(
        mdp=mdp, state_values=state_values, gamma=gamma
    )
    logging.debug("Action value Q(s,a):\n%s", action_values)
    return state_values, action_values


def create_task_id(task_prefix: str):
    """
    Creates a task id using a given prefix
    and a generated partial uuid.
    """
    _uuid = next(iter(str(uuid.uuid4()).split("-")))
    return f"{task_prefix}-{_uuid}"


def partition(items: Sequence[Any], num_partitions: int) -> Sequence[Sequence[Any]]:
    """
    Attempts to partition a list of items into sublists of equal size.

    If the numbers of items is not divisible by the number of
    partition sizes, the first partitions will have more items.
    If the number of partitions is higher than the number of items,
    only non-empty partitions are returned.
    """
    if num_partitions < 1:
        raise ValueError("`num_partitions` must be positive.")
    partition_size = math.ceil(len(items) / num_partitions)
    splits = []
    for idx in range(0, num_partitions - 1):
        splits.append(items[idx * partition_size : (idx + 1) * partition_size])
    splits.append(items[(num_partitions - 1) * partition_size :])
    return [partition for partition in splits if partition]


def bundle(items: Sequence[Any], bundle_size: int) -> Sequence[Sequence[Any]]:
    """
    Bundles items into groups of size `bundle_size`, if possible.
    The last bundle may have fewer items.
    """
    if bundle_size < 1:
        raise ValueError("`bundle_size` must be positive.")

    bundles = []
    bundle_ = []
    for idx, item in enumerate(items):
        if idx > 0 and (idx % bundle_size) == 0:
            if bundle_:
                bundles.append(bundle_)
            bundle_ = []
        bundle_.append(item)
    if bundle_:
        bundles.append(bundle_)
    return bundles


def dataclass_from_dict(clazz: Callable, data: Mapping[str, Any]):  # type: ignore [arg-type]
    """
    Creates an instance of a dataclass from a dictionary.
    """
    fields = list(dataclasses.fields(clazz))
    return clazz(**{field.name: data[field.name] for field in fields})
