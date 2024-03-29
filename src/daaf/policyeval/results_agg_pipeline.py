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
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
import ray.data
import ray.data.datasource
import tensorflow as tf
from ray.data import aggregate
from rlplg import envsuite
from rlplg.learning.tabular import dynamicprog

from daaf import evalmetrics


@dataclasses.dataclass(frozen=True)
class PipelineArgs:
    """
    Pipeline arguments.
    """

    input_dir: str
    output_dir: str


@dataclasses.dataclass(frozen=True)
class StatTest:
    """
    Outputs of a statistical test.
    """

    statistic: float
    pvalue: float


@dataclasses.dataclass(frozen=True)
class MetricStat:
    """
    Statics over a metric.
    """

    mean: float
    stddev: float
    stderr: Optional[float]
    normal_test: Optional[StatTest]


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
        if row["exp_id"] not in acc:
            meta = copy.deepcopy(row["meta"])
            # Path is a random pick
            # from one of the runs
            del meta["path"]
            new_acc[row["exp_id"]] = {
                "meta": meta,
                "state_values": [],
            }
        new_acc[row["exp_id"]]["state_values"].append(row["info"]["state_values"])
        return new_acc

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        """
        Combine two accumulators.
        """
        acc = copy.deepcopy(acc_left)
        for key, value in acc_right.items():
            if key not in acc:
                # copy meta and values
                acc[key] = copy.deepcopy(value)
            else:
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
    # The trailing slash at the end is for gcs paths
    paths = tf.io.gfile.glob(os.path.join(args.input_dir, "**/**/**/**/"))
    ray_env = {}

    logging.info("Running with args: %s", vars(args))
    logging.info("Found a total of %d paths", len(paths))
    logging.info("Ray environment: %s", ray_env)

    with ray.init(runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        ds_metadata = parse_experiment_metadata(paths)
        ds_logs = parse_experiment_logs(paths)

        logging.info("Metadata size: %fMB", ds_metadata.size_bytes() / 1024 / 1024)
        logging.info("Datalogs size: %fMB", ds_logs.size_bytes() / 1024 / 1024)

        metadata = convert_metadata_to_mapping(ds_metadata.to_pandas())
        logging.info("Metadata # keys: %d", len(metadata))
        ds_logs_and_metadata = join_logs_and_metadata(ds_logs, metadata)

        results: Mapping[str, ray.data.Dataset] = ray.get(
            pipeline.remote(ds_logs_and_metadata)
        )
        for key, ds in results.items():
            output_path = os.path.join(args.output_dir, key)
            logging.info("Writing %s to %s", key, output_path)
            ds.write_parquet(output_path)


def parse_experiment_metadata(paths: Sequence[str]) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """
    metadata_files = [os.path.join(path, "experiment-params.json") for path in paths]
    ds_metadata = ray.data.read_json(metadata_files, include_paths=True)
    return ds_metadata


def parse_experiment_logs(paths: Sequence[str]) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """
    logs_files = [os.path.join(path, "experiment-logs.jsonl") for path in paths]
    ds_logs = ray.data.read_json(
        logs_files,
        include_paths=True,
        partition_filter=ray.data.datasource.FileExtensionFilter(
            file_extensions=["jsonl"]
        ),
    )
    return ds_logs


def join_logs_and_metadata(
    ds_logs: ray.data.Dataset, metadata: Mapping[str, Any]
) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """

    def get_exp_id(df: pd.DataFrame) -> pd.Series:
        return df["meta"].apply(lambda meta: meta["exp_id"])

    def get_run_id(df: pd.DataFrame) -> pd.Series:
        return df["meta"].apply(lambda meta: meta["run_id"])

    def get_metadata(df: pd.DataFrame) -> pd.Series:
        paths = df["path"].apply(parse_path_from_filename)
        return paths.apply(lambda path: metadata[path])

    return (
        ds_logs.add_column("meta", get_metadata)
        .add_column("exp_id", get_exp_id)
        .add_column("run_id", get_run_id)
    )


def convert_metadata_to_mapping(df_metadata: pd.DataFrame) -> Mapping[str, Any]:
    """
    Converts the metadata into a mapping.
    """
    mapping = {}
    for row in df_metadata.to_dict(orient="records"):
        path = parse_path_from_filename(row["path"])
        mapping[path] = row
    return mapping


def parse_path_from_filename(file_name: str) -> str:
    """
    Returns the path.
    """
    dir_name, _ = os.path.split(file_name)
    return dir_name


@ray.remote
def pipeline(ds_logs: ray.data.Dataset) -> Mapping[str, ray.data.Dataset]:
    """
    This pipeline aggregates the output of
    multiple runs from each experiment.
    """
    ds_logs = (
        ds_logs.groupby("episode")
        .aggregate(StateValueAggretator(name="state_value_agg"))
        .flat_map(
            lambda row: [
                {"episode": row["episode"], "exp_id": exp_id, **data}
                for exp_id, data in row["state_value_agg"].items()
            ]
        )
    )
    ds_metrics = calculate_metrics(ds_logs)
    return {"logs": ds_logs, "metrics": ds_metrics}


def calculate_metrics(ds: ray.data.Dataset) -> ray.data.Dataset:
    """
    Calculates metrics across runs for each experiment
    entry.
    """

    def calc_metrics(y_preds, y_true, axis):
        mae = evalmetrics.mean_absolute_error(y_preds, y_true, axis=axis)
        rmse = evalmetrics.rmse(y_preds, y_true, axis=axis)
        return {
            "mae": {"mean": np.mean(mae), "std": np.std(mae)},
            "rmse": {"mean": np.mean(rmse), "std": np.std(rmse)},
        }

    def calc_policy_metrics(env_def, gamma, y_preds, y_true):
        env_spec = envsuite.load(env_def["name"], **json.loads(env_def["args"]))
        # just need to do it once for the solution
        dyna_action_values = dynamicprog.action_values_from_state_values(
            mdp=env_spec.mdp, state_values=y_true[0], gamma=gamma
        )
        dyna_best_actions = np.argmax(dyna_action_values, axis=1)
        results = []
        for idx in range(y_preds.shape[0]):
            action_values = dynamicprog.action_values_from_state_values(
                mdp=env_spec.mdp, state_values=y_preds[idx], gamma=gamma
            )
            best_actions = np.argmax(action_values, axis=1)
            results.append(np.mean(best_actions == dyna_best_actions))
        return {"pi_equi": {"mean": np.mean(results), "std": np.std(results)}}

    def apply(row):
        y_preds = row["state_values"]
        y_true = np.tile(row["meta"]["dyna_prog_state_values"], reps=(len(y_preds), 1))
        over_runs_then_states = calc_metrics(y_preds=y_preds, y_true=y_true, axis=0)
        over_states_then_runs = calc_metrics(y_preds=y_preds, y_true=y_true, axis=1)
        policy_equivalence = calc_policy_metrics(
            row["meta"]["env"],
            gamma=row["meta"]["discount_factor"],
            y_preds=y_preds,
            y_true=y_true,
        )
        return {
            **row,
            "over_states_then_runs": over_states_then_runs,
            "over_runs_then_states": over_runs_then_states,
            "policy_metrics": policy_equivalence,
        }

    return ds.map(apply)


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
