"""
Data aggregation pipeline to compute
metrics from experiments.

Metrics
Accuracy of V(s) @ K (episodes)
Monotonic correlation of V(s) @ K
Accuracy of R(s,a)
"""


import json
import logging
import os.path
import time
from typing import Sequence

import ray
import ray.data
import ray.data.datasource
import tensorflow as tf

import daaf


def main():
    """
    Entry point
    """
    paths = tf.io.gfile.glob("/tmp/daaf/exp/logs/1701280078/**/**/**/**")
    logging.info("Found the following input paths: %s", paths)
    ray_env = {
        "py_modules": [daaf],
    }
    logging.info("Ray environment: %s", ray_env)
    with ray.init(runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())
        output = pipeline.remote(
            ray.data.from_items(paths).flat_map(
                fn=lambda path: ray.get(parse_experiment_logs.remote(path))
                .iterator()
                .iter_rows()
            )
        )
        ray.get(output).write_json(f"/tmp/output/{int(time.time())}")


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
            .add_column(col="meta", fn=lambda _: meta)
        )


@ray.remote
def pipeline(ds_logs: ray.data.Dataset) -> ray.data.Dataset:
    """ """
    ds_logs.groupby(key="experiment_id")

    return ds_logs.random_sample(0.01)


if __name__ == "__main__":
    main()
