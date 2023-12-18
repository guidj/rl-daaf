"""
Utilities, helpers.
"""


import contextlib
import dataclasses
import json
import math
import os.path
import types
import uuid
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import tensorflow as tf


class ExperimentLogger(contextlib.AbstractContextManager):
    """
    Logs info for an experiment for given episodes.
    """

    LOG_FILE_NAME = "experiment-logs.jsonl"
    PARAM_FILE_NAME = "experiment-params.json"

    def __init__(
        self, log_dir: str, name: str, params: Mapping[str, Union[int, float, str]]
    ):
        self.log_file = os.path.join(log_dir, self.LOG_FILE_NAME)
        self.param_file = os.path.join(log_dir, self.PARAM_FILE_NAME)
        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)

        with tf.io.gfile.GFile(self.param_file, "w") as writer:
            writer.write(json.dumps(dict(params, name=name)))

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


def create_task_id(timestamp: int):
    """
    Creates a task id using a given timestamp (epoch)
    and a partial uuid.
    """
    _uuid = next(iter(str(uuid.uuid4()).split("-")))
    return f"{timestamp}-{_uuid}"


def partition(items: Sequence[Any], num_partitions: int) -> Sequence[Sequence[Any]]:
    """
    Attempts to partition a list of items into sublists of equal size.

    If the numbers of items is not divisible by the number of
    partition sizes, the first partitions will have more items.
    If the number of partitions is higher than the number of items,
    only non-empty partitions are returned.
    """
    partition_size = math.ceil(len(items) / num_partitions)
    splits = []
    for idx in range(0, num_partitions - 1):
        splits.append(items[idx * partition_size : (idx + 1) * partition_size])
    splits.append(items[(num_partitions - 1) * partition_size :])
    return [partition for partition in splits if partition]


def dataclass_from_dict(clazz: Callable, data: Mapping[str, Any]):  # type: ignore [arg-type]
    """
    Creates an instance of a dataclass from a dictionary.
    """
    fields = list(dataclasses.fields(clazz))
    return clazz(**{field.name: data[field.name] for field in fields})
