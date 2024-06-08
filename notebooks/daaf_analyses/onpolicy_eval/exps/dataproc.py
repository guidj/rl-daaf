import copy
from typing import Any, Mapping

import pandas as pd
import ray

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


def read_data(files):
    with ray.init():
        ds_metrics = ray.data.read_parquet(files)
        df_metrics = ds_metrics.to_pandas()
    del ds_metrics
    return process_data(df_metrics, envs_mapping=ENVS_MAPPING)


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


def get_distinct_envs(df_data: pd.DataFrame):
    envs = {}
    for row in pd.DataFrame(df_data["meta"]).to_dict("records"):
        env = row["meta"]["env"]
        key = (env["name"], env["level"])
        envs[key] = env["args"]
    return envs


def create_eval_result_data(
    df_data: pd.DataFrame,
):
    df_result = copy.deepcopy(df_data)
    for key in (
        "algorithm",
        "algorithm_args",
        "reward_period",
        "discount_factor",
        "drop_truncated_feedback_episodes",
    ):
        df_result[key] = df_result["meta"].apply(lambda meta: meta[key])
    for key in ("level", "name"):
        df_result[key] = df_result["meta"].apply(lambda meta: meta["env"][key])

    del df_result["meta"]

    df_baseline = df_result[df_result["reward_period"] == 1]
    df_result = df_result[df_result["reward_period"] != 1]
    baseline_dfs = []
    distinct_algo_configs = df_result[
        ["algorithm", "algorithm_args", "reward_period"]
    ].drop_duplicates()
    for row in distinct_algo_configs.to_dict("records"):
        df_baseline_rp = copy.deepcopy(df_baseline)
        # reward period is one
        mask = (df_baseline_rp["algorithm"] == row["algorithm"]) & (
            df_baseline_rp["algorithm_args"] == row["algorithm_args"]
        )
        df_baseline_rp = df_baseline_rp[mask]
        df_baseline_rp["reward_period"] = row["reward_period"]
        baseline_dfs.append(df_baseline_rp)
    df_result = pd.concat(baseline_dfs + [df_result], axis=0)
    df_result = df_result.sort_values(
        [
            "method",
            "algorithm",
            "reward_period",
            "algorithm_args",
            "drop_truncated_feedback_episodes",
        ]
    )

    # agg returns
    # df_result["mean_returns"] = df_result["returns"].apply(lambda returns: np.mean(returns))
    dfs = {}
    algo_types = {
        "one-step": set(["one-step-td", "first-visit-mc"]),
        "n-step": set(["nstep-td"]),
    }
    for algo_type, algorithms in algo_types.items():
        df_algo = df_result[df_result["algorithm"].apply(lambda met: met in algorithms)]
        dfs[algo_type] = df_algo

    return dfs
