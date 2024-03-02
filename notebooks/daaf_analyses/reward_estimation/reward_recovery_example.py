import concurrent.futures
import logging
import os.path
import pathlib
import time
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from rlplg import envplay, envsuite
from rlplg.learning.tabular import policies

from daaf import math_ops, replay_mapper

ENV_SPECS = [
    {"name": "ABCSeq", "args": {"length": 7}},
    {"name": "FrozenLake-v1", "args": {"is_slippery": False, "map_name": "4x4"}},
    {
        "name": "GridWorld",
        "args": {"grid": "oooooooooooo\noooooooooooo\noooooooooooo\nsxxxxxxxxxxg"},
    },
    {
        "name": "RedGreenSeq",
        "args": {
            "cure": ["red", "green", "wait", "green", "red", "red", "green", "wait"]
        },
    },
    {"name": "IceWorld", "args": {"map_name": "4x4"}},
    {"name": "TowerOfHanoi", "args": {"num_disks": 4}},
]

BUFFER_MULT = 2**10


def lstsq_reward_estimation(
    obs_matrix: np.ndarray, agg_rewards: np.ndarray
) -> np.ndarray:
    return math_ops.solve_least_squares(
        matrix=obs_matrix,
        rhs=agg_rewards,
    )


def ols_em_reward_estimation(
    obs_matrix: np.ndarray,
    agg_rewards: np.ndarray,
    accuracy: float = 1e-8,
    max_iters: int = 1_000_000,
    stop_check_interval: int = 1000,
) -> Tuple[np.ndarray, int]:
    iteration = 1
    yhat_rewards = np.random.rand(obs_matrix.shape[1])
    # multiply the cumulative reward by visits of each state action
    # dim: (num obs, num states x num actions)
    nomin = np.expand_dims(agg_rewards, axis=-1) * obs_matrix
    qs = np.sum(obs_matrix, axis=0)
    while True:
        delta = np.zeros_like(yhat_rewards)
        # multiply reward guess by row and sum each row's entry
        # dim: num obs
        denom = np.sum(yhat_rewards * obs_matrix, axis=1)
        factor = np.sum(nomin / np.expand_dims(denom, 1), axis=0)
        new_yhat_rewards = yhat_rewards * (factor / qs)
        delta = np.maximum(delta, np.abs(yhat_rewards - new_yhat_rewards))
        if (
            iteration % stop_check_interval == 0
            and np.sum(np.isnan(new_yhat_rewards)) > 0
        ):
            logging.info(
                "Stopping at iteration %d/%d. `nan` values: %s",
                iteration,
                max_iters,
                new_yhat_rewards,
            )
            break
        if np.alltrue(delta < accuracy) or iteration >= max_iters:
            logging.info(
                "Stopping at iteration %d/%d. Max error: %f",
                iteration,
                max_iters,
                np.max(delta),
            )
            break
        yhat_rewards = new_yhat_rewards
        iteration += 1
    return yhat_rewards, iteration


def estimate_reward(
    spec: Mapping[str, Any],
    accuracy: float = 1e-8,
    max_episodes: int = 7500,
    logging_steps: int = 100,
) -> Mapping[str, np.ndarray]:
    env_spec = envsuite.load(spec["name"], **spec["args"])
    init_rtable = np.zeros(
        shape=(env_spec.mdp.env_desc.num_states, env_spec.mdp.env_desc.num_actions),
        dtype=np.float64,
    )
    mapper = replay_mapper.DaafLsqRewardAttributionMapper(
        num_states=env_spec.mdp.env_desc.num_states,
        num_actions=env_spec.mdp.env_desc.num_actions,
        reward_period=4,
        state_id_fn=env_spec.discretizer.state,
        action_id_fn=env_spec.discretizer.action,
        init_rtable=init_rtable,
        buffer_size=env_spec.mdp.env_desc.num_states
        * env_spec.mdp.env_desc.num_actions
        * BUFFER_MULT,
    )
    policy = policies.PyRandomPolicy(num_actions=env_spec.mdp.env_desc.num_actions)
    # collect data
    logging.info("Collecting data for %s", spec["name"])
    episode = 1
    while True:
        traj = envplay.generate_episodes(
            env_spec.environment, policy=policy, num_episodes=1
        )
        for _ in mapper.apply(traj):
            pass

        if (
            not mapper._estimation_buffer.is_empty
            and mapper._estimation_buffer.is_full_rank
        ):
            break

        if episode % logging_steps == 0:
            logging.info("Data collection for %s at %d episodes", spec["name"], episode)
        if episode >= max_episodes:
            break
        episode += 1

    # estimate rewards
    if mapper._estimation_buffer.is_full_rank:
        logging.info(
            "Estimating rewards for %s, after %d episodes. Matrix shape: %s",
            spec["name"],
            episode,
            mapper._estimation_buffer.matrix.shape,
        )
        yhat_ols_em, iters = ols_em_reward_estimation(
            obs_matrix=mapper._estimation_buffer.matrix,
            agg_rewards=mapper._estimation_buffer.rhs,
            accuracy=accuracy,
        )
        yhat_ols_em = np.reshape(
            yhat_ols_em,
            newshape=(
                env_spec.mdp.env_desc.num_states,
                env_spec.mdp.env_desc.num_actions,
            ),
        )
        logging.info("OLS ran in %d iterations for %s", iters, spec["name"])
        yhat_lstsq = np.reshape(
            lstsq_reward_estimation(
                obs_matrix=mapper._estimation_buffer.matrix,
                agg_rewards=mapper._estimation_buffer.rhs,
            ),
            newshape=(
                env_spec.mdp.env_desc.num_states,
                env_spec.mdp.env_desc.num_actions,
            ),
        )
        return {"least": yhat_lstsq, "ols-em": yhat_ols_em}
    logging.info(
        "Matrix is ill defined. Skipping reward estimation for %s: %s",
        spec["name"],
        spec["args"],
    )
    return {"least": None, "ols-em": None}


def estimation_experiment(env_specs: Sequence[Mapping[str, Any]]):
    rows = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_spec = {
            executor.submit(estimate_reward, spec): spec for spec in env_specs
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            output = future.result()
            rows.append({"spec": spec, **output})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    now = int(time.time())
    df_results = estimation_experiment(env_specs=ENV_SPECS)
    df_results.to_json(
        os.path.join(str(pathlib.Path.home()), f"fs/exp/reward-recovery/{now}-report.json"), orient="records"
    )
