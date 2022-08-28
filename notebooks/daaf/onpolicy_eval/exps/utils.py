import os
import os.path
import json
from typing import Dict, List, Optional, Sequence, Callable, Any
import copy
import math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf


METHOD_ID_TO_NAME = {
    "identity-mapper": "FR",
    "reward-imputation-mapper": "ZI-M",
    "reward-estimation-ls-mapper": "LEAST",
    "cumulative-reward-mapper": "S-M"
}


def read_file(path: str):
    with tf.io.gfile.GFile(path, "r") as reader:
        return reader.read()


def read_json(path: str):
    with tf.io.gfile.GFile(path, "r") as reader:
        return json.load(reader)


def qtable(logs: pd.DataFrame, episode: int) -> np.ndarray:
    return logs.loc[episode, "metadata"]["qtable"]


def metadata(logs: pd.DataFrame, key: str, dtype=None) -> pd.Series:
    return pd.Series(logs["metadata"].apply(lambda x: x[key]), name=key, dtype=dtype)


def read_logs(path: str) -> pd.DataFrame:
    with tf.io.gfile.GFile(path, "r") as reader:
        return pd.read_json(reader, lines=True)


def plot_stages(df_logs: pd.DataFrame, episodes, size=4):
    _, axes = plt.subplots(ncols=len(episodes), figsize=(size*len(episodes), size))
    for episode, ax in zip(episodes, axes):
        sns.heatmap(qtable(df_logs, episode), annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"After Episode {episode}")
        
        
def replace_method_id_with_name(df: pd.DataFrame):
    _df = copy.deepcopy(df)
    _df["method"] = _df["method"].apply(lambda x: METHOD_ID_TO_NAME[x])
    return _df


def read_experiment_results(pattern: str, post_proc_fn: Callable[[pd.DataFrame], pd.DataFrame] = replace_method_id_with_name):
    files = tf.io.gfile.glob(pattern)
    dfs = []
    for filename in files:
        dfs.append(read_logs(filename))
    _df = pd.concat(dfs, axis=0)
    return post_proc_fn(_df)
    


def incomplete_or_missing_results(df, num_expected_alternatives: Optional[int] = 2, episode_lower_bound: int = 9990):
    """
    There is no reason to remove the baseline,
    since other configurations of reward internval
    can exist.

    But if a baseline is missing, there is no point including
    the other configs
    """

    # did not run to the end
    _df = copy.deepcopy(df)
    df_slice = _df[["config", "method", "episode"]]
    df_group = df_slice.groupby(["config", "method"]).max().reset_index()
    incomplete = df_group[df_group["episode"] < episode_lower_bound]
    incomplete_configs = incomplete["config"].unique().tolist()

    # is missing some methods
    # anything P1 should just be one
    # anything > P1 should be 2
    df_slice = _df[["config", "method"]]
    df_slice = df_slice.drop_duplicates()
    config_counts = df_slice["config"].value_counts()
    missing_status = []
    for config, count in config_counts.items():
        is_baseline = config.endswith("P1") and count == 1
        is_alternative = not config.endswith("P1") and (num_expected_alternatives is None or count == num_expected_alternatives)
        missing_status.append({
            "config": config,
            "ok_count": (is_baseline or is_alternative)
        })

    df_missing = pd.DataFrame(missing_status)
    missing_configs = df_missing[df_missing["ok_count"] == False]["config"].unique().tolist()
    return set(incomplete_configs + missing_configs)


def filter_configs_from_results(df, configs: Sequence[str]):
    _df = copy.deepcopy(df)
    mask = _df["config"].apply(lambda x: x not in configs)
    return _df[mask]


def get_experiment_configs(df):
    groups = groupby(df, key="L", value="P")
    result = {}
    for key, value in groups.items():
        result[key] = sorted(np.unique(value).tolist())
    return result


def groupby(df, key: str, value: str):
    acc = {}
    for _key, df_group in df[[key, value]].groupby([key]):
        acc[_key] = df_group[value].values
    return acc
    

def slice_config(df, config: str, method: Optional[str] = None):
    _df = copy.deepcopy(df)
    if method is not None:
        mask = (_df["config"] == config) & (_df["method"] == method)
    else:
        mask = _df["config"] == config
    return _df[mask].sort_values(by="episode").reset_index()


def slice_method(df, method: str):
    _df = copy.deepcopy(df)
    mask =_df["method"] == method
    return _df[mask].sort_values(by="episode").reset_index()


def config_baseline(df_config_slice, df):
    _df = copy.deepcopy(df)
    configs = df_config_slice["config"].apply(lambda x: x.split("-")[0] + "-P1").unique().tolist()
    assert len(configs) == 1
    config = next(iter(configs))
    return df[df["config"] == config].sort_values(by="episode").reset_index()


def plot_config_comparison(df, config, metric="rmse", ax=None, log_scale_x:bool = False, log_scale_y:bool = False, colors: Sequence[str] = None):
    df_config = slice_config(df, config)
    df_baseline = config_baseline(df_config, df)
    _df = pd.concat([df_config, df_baseline], axis=0)
    level, period = config.split("-")
    level, period = level[1:], period[1:]
    title = f"Level={level},P={period}"
    return _plot_config_comparison(_df, title=title, metric=metric, ax=ax, log_scale_x=log_scale_x, log_scale_y=log_scale_y, colors=colors)


def _plot_config_comparison(df, title, metric, ax=None, log_scale_x:bool = False, log_scale_y:bool = False, colors: Sequence[str] = None):
    methods = sorted(df["method"].unique())
    if ax is None:
        fig, axes = plt.subplots(1, figsize=(8, 6))
    else:
        axes = ax
    axes = [axes]*len(methods)
    handles = []
    if colors is None:
        colors = [None] * len(methods)
    for method, ax, color in zip(methods, axes, colors):
        # x̄ ± (1.96 × SE)
        df_method = df[df["method"] == method]
        episode = df_method["episode"]
        mean = df_method[metric].apply(lambda x: x["mean"])
        std = df_method[metric].apply(lambda x: x["std"])
        # nom = df_method[metric].apply(lambda x: x["std"])
        # denom = df_method[metric].apply(lambda x: np.sqrt(x["sample_size"]))
        # std = (nom / denom) * 1.96
        yerr = [mean - std, mean + std]
        handle, = ax.plot(episode, mean, axes=ax, label=method, color=color)
        handles.append(handle)
        ax.fill_between(episode, mean-std, mean+std, alpha=0.3, color=handle.get_color())
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric)
        if log_scale_x:
            ax.set_xscale("log")
        if log_scale_y:
            ax.set_yscale("log")

    axes[0].set_title(title)
    plt.legend(handles=handles, labels=methods)
    return axes[0]


def expand_for_lookup(df):
    _df = copy.deepcopy(df)
    # skip first char
    _df["L"] = df["config"].apply(lambda x: x.split("-")[0][1:])
    _df["P"] = df["config"].apply(lambda x: int(x.split("-")[1][1:]))
    return _df


def table_config_comparison(df, config, metric="rmse"):
    df_config = slice_config(df, config)
    df_baseline = config_baseline(df_config, df)
    _df = pd.concat([df_config, df_baseline], axis=0)
    _df = _df[_df["episode"] == _df["episode"].max()]

    _df[f"{metric} ± std"] = _df[metric].apply(lambda x: str(np.around(x["mean"], 3)) + "±" + str(np.around(x["std"], 3)))
    _df[f"{metric} (mean)"] = _df[metric].apply(lambda x: str(np.around(x["mean"], 3)))
    _df[f"{metric} (std)"] = _df[metric].apply(lambda x: str(np.around(x["std"], 3)))

    _df["config"] = config
    _df = _df[["config", "method", "episode", f"{metric} ± std", f"{metric} (mean)", f"{metric} (std)"]]
    return _df.sort_values(by=["config", "method"])


def combine_table_config_comparisons(df, configs, metric="rmse"):
    dfs = []
    for config in configs:
        dfs.append(table_config_comparison(df, config, metric=metric))
    return pd.concat(dfs, axis=0)


def plot_multiple_configs(df, configs, nrows, ncols, metric="rmse", figsize=(24,16), log_scale_x:bool = False, 
                          log_scale_y:bool = False, colors: Sequence[str] = None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for config, axes in zip(configs, axes.flatten()):
        plot_config_comparison(df, config, metric=metric, ax=axes, log_scale_x=log_scale_x, log_scale_y=log_scale_y, colors=colors)
    return axes


def _plot_config_method(df, title, metric, ax=None):
    configs = sorted(df["config"].unique())
    if ax is None:
        fig, axes = plt.subplots(1, figsize=(8, 6))
    else:
        axes = ax
    axes = [axes]*len(configs)
    handles = []
    for config, ax in zip(configs, axes):
        df_config = df[df["config"] == config]
        episode = df_config["episode"]
        mean = df_config[metric].apply(lambda x: x["mean"])
        std = df_config[metric].apply(lambda x: x["std"])

        yerr = [mean - std, mean + std]
        handle, = ax.plot(episode, mean, axes=ax, label=config)
        handles.append(handle)
        ax.fill_between(episode, mean-std, mean+std, alpha=0.3, color=handle.get_color())

    axes[0].set_title(title)
    plt.legend(handles=handles, labels=configs)
    return axes[0]



def plot_config_method(df, config, method, metric="rmse", ax=None):
    return _plot_config_method(df_config, title=config, metric=metric, ax=ax)


def plot_multiple_configs_method(df, configs, method, metric, title=None, figsize=(24,16)):
    fig, axes = plt.subplots(figsize=figsize)
    axes = [axes]*len(configs)
    handles = []
    df_method = slice_method(df, method)
    for config, ax in zip(configs, axes):
        _df = df_method[df_method["config"] == config]
        episode = _df["episode"]
        mean = _df[metric].apply(lambda x: x["mean"])
        std = _df[metric].apply(lambda x: x["std"])

        yerr = [mean - std, mean + std]
        handle, = ax.plot(episode, mean, axes=ax, label=config)
        handles.append(handle)
        ax.fill_between(episode, mean-std, mean+std, alpha=0.3, color=handle.get_color())

    axes[0].set_title(title)
    plt.legend(handles=handles, labels=configs)
    return axes[0]


# def error_plot(df, x_col: str, split_col: str, figsize=(6, 6)):
#     fig, axes = plt.subplots(figsize=figsize)
#     splits = df[split_col].unique()
#     for split in splits:
#         mask = df[split_col] == split
#         _df_split = df[mask]
#         x = _df_split[x_col]
#         y = _df_split["rmse (mean)"].astype(float)
#         e = _df_split["rmse (std)"].astype(float)    
#         plt.xticks(rotation=45)
#         axes.errorbar(x, y, yerr=e)
#     plt.legend(labels=splits)
#     return axes


def get_configs(df, level: str, exclude_baseline: bool = True):
    mask = df["L"] == level
    _df = df[mask]
    configs = _df.sort_values(by=["P"])["config"].unique().tolist()
    if exclude_baseline:
        configs = [config for config in configs if not config.endswith("P1")]
    return configs


def export_figure(ax, name: str, format: str = "pdf", dpi=300, transparent: bool = True):
    # to be able to save, we must plot
    _ = ax.plot()
    base_dir = os.path.dirname(name)
    if not os.path.exists(base_dir):
        tf.io.gfile.makedirs(base_dir)
    plt.savefig(f"{name}.{format}", dpi=dpi, format=format, transparent=transparent)
    
    
    
def df_final_print(df, metric):
    _df = copy.deepcopy(df)
    config = _df["config"]
    level = config.apply(lambda x: x.split("-")[0][1:])
    period = config.apply(lambda x: int(x.split("-")[1][1:]))
    _df["Method"] = _df["method"]
    _df["Level"] = level
    _df["P"] = period
    return _df[["Level", "P", "Method", f"{metric} ± std"]]
    
    
def final_episode_logs(df):
    _df = copy.deepcopy(df)
    max_episode = _df["episode"].max()
    mask = _df["episode"] == max_episode
    return _df[mask]
    

def final_episode_metrics(df, metric):
    df_final_episode = final_episode_logs(df)
    df_metric = copy.deepcopy(df_final_episode[["config", "method", "L", "P"]])
    df_metric[f"{metric} (mean)"] = df_final_episode[metric].apply(lambda x: x["mean"])
    df_metric[f"{metric} (std)"] = df_final_episode[metric].apply(lambda x: x["std"])
    return df_metric
    

def boxplot_corr(df, metric, title: str, x_label: str = None, y_label: str = None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    mask = df["P"] != 1
    sns.boxplot(x="method", y=metric,
                hue="P",
                data=df[mask],
                ax=ax)
    # plt.xticks(rotation = 25)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([0.0, 1.0])
    plt.title(title)
    return ax
    

def metric_results_table(df, metric, experiments, level_transform_fn: Callable[[str], Any]):
    _configs = []
    for level, periods in experiments.items():
        _configs.extend([f"L{level}-P{period}" for period in periods if period > 1])

    _df_configs = combine_table_config_comparisons(df, configs=_configs, metric=metric).reset_index().drop(labels=["index"], axis=1)
    _df_pivot = _df_configs.pivot(index=["config"], columns="method", values=f"{metric} ± std").reset_index()
    _df_pivot["Level/Map"] = _df_pivot["config"].apply(lambda x: level_transform_fn(x.split("-")[0].lstrip("L")))
    _df_pivot["P"] = _df_pivot["config"].apply(lambda x: int(x.split("-")[1].lstrip("P")))
    _df_pivot = _df_pivot[["Level/Map", "P","FR", "S-M", "ZI-M", "LEAST"]]
    _df_pivot = _df_pivot.sort_values(by=["Level/Map", "P"])
    # select columns to order them
    return _df_pivot

    
def qtable_sample_heatmaps(df, method, figsize=(16, 16)):
    _dfq = final_episode_metrics(df, metric="qtable")
    _dfq = _dfq[_dfq["method"] == method]
    _dfq_sample = _dfq.sample(n=4)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    for idx, ax in enumerate(axes.flatten()):
        sns.heatmap(_dfq_sample.iloc[idx]["qtable (mean)"], ax=ax)
        ax.set_title(f"{method}, {_dfq_sample.iloc[idx]['config']}")
    return axes


def highlight_min_result(df, cols):
    def highlight(text):
        return "".join(["\textbf{", text, "}"])
    _df = copy.deepcopy(df)
    _df_cols = _df[cols]
    mask = np.argmin(_df_cols.applymap(lambda x: float(x.split("±")[0])).values, axis=1)
    new_df = copy.deepcopy(_df_cols)
    rows = list(range(len(df)))
    assert len(rows) == len(mask)
    new_df.values[rows, mask] = list(map(highlight, new_df.values[rows, mask]))
    _df[cols] = new_df[cols]
    return _df


def gsize(cols: int, rows: int, size: int = 10):
    return (cols*size), (rows*size)