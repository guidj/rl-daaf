"""
Data aggregation pipeline to compute
metrics from experiments.

Metrics
Accuracy of V(s) @ K (episodes)
Monotonic correlation of V(s) @ K
Accuracy of R(s,a)
"""


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
class Stat:
    """ """

    mean: float
    stddev: float
    p95_ci: float


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

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        acc = copy.deepcopy(acc_left)
        for key, value in acc_right.items():
            if key in acc:
                acc[key]["state_values"].extend(value["state_values"])
            else:
                acc[key] = copy.deepcopy(acc_right[key])
        return acc

    def _accumulate_row(self, acc: AggType, row: Row) -> AggType:
        key = (row["experiment_id"], row["episode"])
        new_acc = copy.deepcopy(acc)
        if key not in acc:
            new_acc[key] = {
                "meta": row["meta"],
                "experiment_id": row["experiment_id"],
                "state_values": [],
            }
        new_acc[key]["state_values"].append(row["info"]["state_values"])
        return new_acc

    def _finalize(self, acc: AggType) -> Any:
        return acc


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
        # create io placement group
        # to limit resources on data reading
        # pg_io = placement_group([{"CPU": 2}])
        # ray.get(pg_io.ready())
        # output = pipeline.remote(
        #     ray.data.from_items(paths).flat_map(
        #         fn=lambda row: ray.get(
        #             parse_experiment_logs.remote(row["item"])
        #         ).iter_rows(),
        #     )
        # )
        # ray.get(output).write_json(f"/tmp/output/{int(time.time())}")
        datasets = ray.get([parse_experiment_logs.remote(path) for path in paths])
        if len(datasets) > 1:
            ds_logs = (
                datasets[0].union(*datasets[1:]) if len(datasets) > 1 else datasets[0]
            )

        output = pipeline.remote(ds_logs)
        # output = pipeline.remote()
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
            .add_column(col="meta", fn=lambda df: [meta] * len(df))
        )


@ray.remote
def pipeline(ds_logs: ray.data.Dataset) -> ray.data.Dataset:
    """
    This pipeline.
    Just report results (e.g. RMSE) from each run;
    Leave statistics for notebooks.
    """
    # jsonl: episode, steps, returns, info (state_values)
    # meta: {
    # "policy_type": "single-step",
    # "traj_mapping_method": "identity-mapper",
    # "algorithm": "one-step-td",
    # "reward_period": 1,
    # "drop_truncated_episodes": false,
    # "epsilon": 0.0,
    # "learning_rate": 0.1,
    # "discount_factor": 1.0,`  11`
    # "name": "1701280078-d0c40572-StateRandomWalk-lvl1-run4"
    # }
    # groupby dataset by experiment_id
    # aggregate each metric
    # output: (ekey, episode, meta, metrics)

    # result = ds_logs.aggregate(StateValueAggretator(name="state_value_agg"))
    # return result["state_value_agg"]
    return ds_logs.groupby("steps").count()


def calculate_statistics(sample: Sequence[float]) -> Stat:
    # https://www.scribbr.com/statistics/confidence-interval/
    # assumption of normality and t-distribution
    pass


if __name__ == "__main__":
    main()
