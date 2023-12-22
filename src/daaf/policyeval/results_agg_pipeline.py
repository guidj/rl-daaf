"""
Data aggregation pipeline to compute.
Aggregates results from multiple
runs from each experiment.
"""


import argparse
import copy
import dataclasses
import json
import logging
import os.path
import time
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import ray
import ray.data
import ray.data.datasource
import tensorflow as tf
from ray.data import aggregate

import daaf


@dataclasses.dataclass(frozen=True)
class PipelineArgs:
    """
    Pipeline arguments.
    """

    input_dir: str
    output_dir: str


class StateValueAggretator(aggregate.AggregateFn):
    """
    Aggregates state-values.
    """

    AggType = Mapping[Tuple[str, int], Tuple[Any, Sequence[np.ndarray]]]
    Row = Mapping[str, Any]

    def __init__(self, name: str = "StateValueAggretator()"):
        super().__init__(
            init=self._init,
            merge=self._merge,
            accumulate_row=self._accumulate_row,
            finalize=self._finalize,
            name=name,
        )

    def _init(self, key: Any) -> AggType:
        del key
        return {}

    def _accumulate_row(self, acc: AggType, row: Row) -> AggType:
        """
        Add a single row to aggregation.
        """
        new_acc = copy.deepcopy(acc)
        if row["experiment_id"] not in acc:
            new_acc[row["experiment_id"]] = {
                "meta": row["meta"],
                "state_values": [],
            }
        new_acc[row["experiment_id"]]["state_values"].append(
            row["info"]["state_values"]
        )
        return new_acc

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        """
        Combine two accumulators.
        """
        acc = copy.deepcopy(acc_left)
        for key, value in acc_right.items():
            if key not in acc:
                acc[key] = copy.deepcopy(acc_right[key])
            acc[key]["state_values"].extend(value["state_values"])
        return acc

    def _finalize(self, acc: AggType) -> Any:
        """
        Project final output.
        """
        return acc


def main():
    """
    Entry point
    """
    args = parse_args()
    paths = tf.io.gfile.glob(os.path.join(args.input_dir, "**/**/**/**"))
    logging.info("Found the following input paths: %s", paths)
    ray_env = {
        "py_modules": [daaf],
    }
    logging.info("Ray environment: %s", ray_env)
    with ray.init(runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())
        datasets = ray.get([parse_experiment_logs.remote(path) for path in paths])
        if len(datasets) > 1:
            ds_logs = (
                datasets[0].union(*datasets[1:]) if len(datasets) > 1 else datasets[0]
            )

        output: ray.data.Dataset = ray.get(pipeline.remote(ds_logs))
        now = int(time.time())
        # TODO: save as parquet
        output.write_json(os.path.join(args.output_dir, str(now)))


@ray.remote
def parse_experiment_logs(path: str) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """
    metadata_file = os.path.join(path, "experiment-params.json")
    logs_file = os.path.join(path, "experiment-logs.jsonl")
    with tf.io.gfile.GFile(metadata_file, mode="r") as readable:
        meta = json.load(readable)
        ds_exp_logs = ray.data.read_json(
            logs_file,
            partition_filter=ray.data.datasource.FileExtensionFilter(
                file_extensions=["jsonl"]
            ),
        )
        tokens = meta["name"].split("-")
        experiment_id = "-".join(tokens[:-1])
        run_id = tokens[-1]
        return (
            ds_exp_logs.add_column(col="experiment_id", fn=lambda _: experiment_id)
            .add_column(col="run_id", fn=lambda _: run_id)
            .add_column(col="meta", fn=lambda df: [meta] * len(df))
        )


@ray.remote
def pipeline(ds_logs: ray.data.Dataset) -> ray.data.Dataset:
    """
    This pipeline aggregates the output of
    multiple runs from each experiment.
    """
    return (
        ds_logs.groupby("episode")
        .aggregate(StateValueAggretator(name="state_value_agg"))
        .flat_map(
            lambda row: [
                {"episode": row["episode"], "experiment_id": experiment_id, **data}
                for experiment_id, data in row["state_value_agg"].items()
            ]
        )
    )


def parse_args() -> PipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input-dir", type=str)
    arg_parser.add_argument("--output-dir", type=str)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return PipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main()
