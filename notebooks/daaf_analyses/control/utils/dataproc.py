import collections
import copy
import importlib
import os
import pathlib
import uuid
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
import ray
import scipy
import seaborn as sns
import tensorflow as tf
from daaf import estimator_metrics
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats import proportion

ENVS_MAPPING = {
    (
        "IceWorld",
        "4KE3ASUFQGGUPERSDDRQAZAMA46CI2CMCJHGWJ7MRNI64JMEBETNDXFFPYWTQJF46S5BJ4NXXCHNMJSLII3ROYXI76DFOC3VAABGNVA=",
    ): {"args": '{"map_name": "4x4"}', "name": "4x4"},
    ("ABCSeq", "10"): {
        "args": '{"length": 10, "distance_penalty": false}',
        "name": "n=10",
    },
    (
        "RedGreenSeq",
        "NNLHYJFTC5ENMMDZWRNQ37B6VVDXQ7WHB5EJOPXYZFLMJEZOYLTSLB4ID4WHQG57XQPNUHGZCFDCWHYGXWSBW7FBWYRZGAGBW4J7MEQ=",
    ): {
        "args": '{"cure": ["red", "green", "wait", "green", "red", "red", "green", "wait"]}',
        "name": "n=9",
    },
    (
        "FrozenLake-v1",
        "U75ZLQLLXYRFQE5KOJJGNVQZGQ65U5RVVN3ZV5F4UNYQVK6NGTAAU62O2DKMOEGACNNUQOSWGYYOV7LQHK7GAWG2CL3U3RZJFIEIB5I=",
    ): {"args": '{"is_slippery": false, "map_name": "4x4"}', "name": "4x4"},
    ("TowerOfHanoi", "4"): {"args": '{"num_disks": 4}', "name": "disks=4"},
    ("ABCSeq", "7"): {
        "args": '{"length": 7, "distance_penalty": false}',
        "name": "n=7",
    },
    (
        "IceWorld",
        "JKNDNWGM45FELU53ZLLVJEPY2SFZBCX54PSACOQOFMTDUAK5VNQ4KE45QZINGYFU5GR6D7F3GJMW7EC4TAY5PHCYRN5GPGP7YNACHEI=",
    ): {"args": '{"map_name": "8x8"}', "name": "8x8"},
    (
        "GridWorld",
        "P3VJZBIJ7PNUOFG2SCF532NH5AQ6NOBZEZ6UZNZ7D3AU3GQZSLKURMS2SRPEUF6O65F3ETJXEFNTR3UYS73TUCIIU3YIONXHAR6WE5A=",
    ): {
        "args": '{"grid": "oooooooooooo\\noooooooooooo\\noooooooooooo\\nsxxxxxxxxxxg"}',
        "name": "4x12",
    },
}


MAPPERS_NAMES = {
    "identity-mapper": "FR",
    "daaf-trajectory-mapper": "DMR",
    "daaf-impute-missing-reward-mapper": "IMR",
    "daaf-lsq-reward-attribution-mapper": "LEAST",
    "daaf-nstep-td-update-mark-mapper": "nTD-SU",
    "daaf-mdp-with-options-mapper": "OT",
}

POLICY_NAMES = {"options": "OP", "single-step": "PP"}


def process_data(df_raw, envs_mapping):
    def get_method(meta: Mapping[str, Any]):
        return "/".join([meta["policy_type"], meta["traj_mapping_method"]])

    def simplify_meta(meta):
        new_meta = copy.deepcopy(meta)
        name = new_meta["env"]["name"]
        level = new_meta["env"]["level"]
        spec = envs_mapping.get((name, level), {"name": level})
        new_meta["env"]["_level"] = level
        new_meta["env"]["level"] = spec["name"]
        new_meta["traj_mapping_method"] = MAPPERS_NAMES[new_meta["traj_mapping_method"]]
        new_meta["policy_type"] = POLICY_NAMES[new_meta["policy_type"]]
        return new_meta

    df_proc = copy.deepcopy(df_raw)
    df_proc["meta"] = df_proc["meta"].apply(simplify_meta)
    df_proc["method"] = df_proc["meta"].apply(get_method)
    return df_proc


def read_data(files):
    ds_metrics = ray.data.read_parquet(files)
    df_metrics = ds_metrics.to_pandas()
    return process_data(df_metrics, envs_mapping=ENVS_MAPPING)


def wide_metrics(df_metrics):
    df_raw = df_metrics.drop(["metrics"], axis=1, inplace=False)
    return df_raw.explode("returns")


def get_distinct_envs(df_data: pd.DataFrame):
    envs = {}
    for row in pd.DataFrame(df_data["meta"]).to_dict("records"):
        env = row["meta"]["env"]
        key = (env["name"], env["level"])
        envs[key] = env["args"]
    return envs


def drop_duplicate_sets(df_data: pd.DataFrame, keys):
    col = str(uuid.uuid4())
    rows = []
    for row in df_data.to_dict("records"):
        col_set = sorted([row[key] for key in keys])
        new_row = copy.deepcopy(row)
        new_row[col] = col_set
        rows.append(new_row)
    df_raw = pd.DataFrame(rows)
    return df_raw.drop_duplicates(col).drop([col], axis=1)
