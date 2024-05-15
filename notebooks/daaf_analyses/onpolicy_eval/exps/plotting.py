import copy
import json
import os
import os.path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

DASHES = {
    "PP/FR": (1, 0),
    "PP/IMR": (1, 1),
    "PP/LEAST": (2, 1),
    "PP/nTD-SU": (3, 1),
    "OP/OT": (4, 2),
    "PP/DMR": (6, 2),
}


METHODS_PALETTES = {
    key: palette
    for key, palette in zip(
        ["PP/FR", "PP/IMR", "PP/LEAST", "PP/nTD-SU", "OP/OT", "PP/DMR"],
        sns.color_palette(),
    )
}

SHORT_NAMES = {"TowerOfHanoi": "ToH", "RedGreenSeq": "RGS"}


def plot_eval_result(
    env: str,
    level: str,
    discount_factor: float,
    drop_truncated_feedback_episodes: bool,
    df_data: pd.DataFrame,
    suffix: str,
    metric_family: str,
    metric_col: str,
    max_episode: int,
    max_reward_period: int,
    output_dir: str,
):
    df_data = df_data[suffix]
    filter_mask = (
        (df_data["name"] == env)
        & (df_data["level"] == level)
        & (df_data["discount_factor"] == discount_factor)
        & (
            df_data["drop_truncated_feedback_episodes"]
            == drop_truncated_feedback_episodes
        )
        & (df_data["episode"] <= max_episode)
        & (df_data["reward_period"] <= max_reward_period)
    )
    df_data = df_data[filter_mask]
    metric_values = df_data[metric_family].apply(lambda mp: mp[metric_col])
    df_result = copy.deepcopy(
        df_data.drop(
            columns=[
                "state_values",
                "over_states_then_runs",
                "over_runs_then_states",
                "policy_metrics",
            ]
        )
    )
    del df_data
    df_result[metric_col] = metric_values
    df_result = df_result.explode(metric_col)

    def rename_env(env: str):
        try:
            return SHORT_NAMES[env]
        except KeyError:
            return env

    df_result["name"] = df_result["name"].apply(rename_env)
    df_result = df_result.rename(
        {
            "reward_period": "P",
            "episode": "Episode",
            "algorithm": "A",
            metric_col: metric_col.upper(),
            "method": "Method",
        },
        axis=1,
    )
    # split to give flexibility in plotting
    algorithms = sorted(df_result["A"].unique())
    for algorithm in algorithms:
        print(algorithm)
        name_prefix = "_".join(
            [str(token) for token in ["rc", algorithm, env, level, discount_factor]]
        )
        df_plot = df_result[df_result["A"] == algorithm]
        facet_grid = sns.relplot(
            data=df_plot,
            x="Episode",
            y=metric_col.upper(),
            col="P",
            row="A",
            hue="Method",
            style="Method",
            kind="line",
            palette=METHODS_PALETTES,
            errorbar="sd",
            height=2,
            dashes=DASHES,
        )
        title_template = ", ".join([f"{rename_env(env)}({level})", "P = {col_name}"])
        facet_grid.set_titles(title_template)
        export_figure(
            facet_grid.figure, os.path.join(output_dir, f"{name_prefix}_{suffix}")
        )
        plt.show()


def export_figure(
    figure, name: str, format: str = "pdf", dpi=300, transparent: bool = True
):
    # to be able to save, we must plot
    base_dir = os.path.dirname(name)
    if not os.path.exists(base_dir):
        tf.io.gfile.makedirs(base_dir)
    figure.savefig(f"{name}.{format}", dpi=dpi, format=format, transparent=transparent)
