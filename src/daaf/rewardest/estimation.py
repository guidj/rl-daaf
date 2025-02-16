import collections
import itertools
import logging
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import numpy as np
from daaf import core, envplay, envsuite
from daaf.learning.tabular import policies

from daaf import replay_mapper
from daaf.learning import opt

BUFFER_MULT = 2**10


def estimate_reward(
    spec: Mapping[str, Any],
    reward_period: int,
    accuracy: float = 1e-8,
    max_episodes: int = 7500,
    logging_steps: int = 100,
    factor_terminal_states: bool = False,
    prefill_buffer: bool = False,
) -> Mapping[str, Any]:
    env_spec = envsuite.load(spec["name"], **spec["args"])
    terminal_states = core.infer_env_terminal_states(env_spec.mdp.transition)
    init_rtable = np.zeros(
        shape=(env_spec.mdp.env_desc.num_states, env_spec.mdp.env_desc.num_actions),
        dtype=np.float64,
    )
    mapper = replay_mapper.DaafLsqRewardAttributionMapper(
        num_states=env_spec.mdp.env_desc.num_states,
        num_actions=env_spec.mdp.env_desc.num_actions,
        reward_period=reward_period,
        state_id_fn=env_spec.discretizer.state,
        action_id_fn=env_spec.discretizer.action,
        init_rtable=init_rtable,
        buffer_size=env_spec.mdp.env_desc.num_states
        * env_spec.mdp.env_desc.num_actions
        * BUFFER_MULT,
        terminal_states=frozenset(terminal_states),
        factor_terminal_states=factor_terminal_states,
        prefill_buffer=prefill_buffer,
    )
    policy = policies.PyRandomPolicy(num_actions=env_spec.mdp.env_desc.num_actions)
    # collect data
    logging.info("Collecting data for %s/%s", spec["name"], spec["args"])
    episode = 1
    steps = 0
    yhat_lstsq: Optional[np.ndarray] = None
    yhat_ols_em: Optional[np.ndarray] = None
    meta: Dict[str, Any] = {
        "max_episodes": max_episodes,
        "est_accuracy": accuracy,
        "ols_iters": None,
    }
    num_visited_states_dist: Dict[int, int] = collections.defaultdict(int)

    while True:
        traj = envplay.generate_episode(env_spec.environment, policy=policy)
        episode_visited_states = set()
        for traj_step, step in zip(mapper.apply(traj), itertools.count()):
            episode_visited_states.add(
                env_spec.discretizer.state(traj_step.observation)
            )
        num_visited_states_dist[len(episode_visited_states)] += 1

        if (
            not mapper._estimation_buffer.is_empty
            and mapper._estimation_buffer.is_full_rank
        ):
            break

        if episode % logging_steps == 0:
            logging.info(
                "Data collection for %s/%s at %d episodes",
                spec["name"],
                spec["args"],
                episode,
            )
        if episode >= max_episodes:
            break
        episode += 1
        steps += step + 1

    # estimate rewards
    if mapper._estimation_buffer.is_full_rank:
        logging.info(
            "Estimating rewards for %s/%s, after %d episodes (%d steps). Matrix shape: %s",
            spec["name"],
            spec["args"],
            episode,
            steps,
            mapper._estimation_buffer.matrix.shape,
        )
        yhat_ols_em, iters = ols_em_reward_estimation(
            obs_matrix=mapper._estimation_buffer.matrix,
            agg_rewards=mapper._estimation_buffer.rhs,
            accuracy=accuracy,
        )
        logging.info(
            "OLS ran in %d iterations for %s/%s", iters, spec["name"], spec["args"]
        )
        yhat_lstsq = lstsq_reward_estimation(
            obs_matrix=mapper._estimation_buffer.matrix,
            agg_rewards=mapper._estimation_buffer.rhs,
        )

        if factor_terminal_states:
            yhat_lstsq = expand_reward_with_terminal_action_values(
                yhat_lstsq,
                num_states=env_spec.mdp.env_desc.num_states,
                num_actions=env_spec.mdp.env_desc.num_actions,
                terminal_states=terminal_states,
            )
            yhat_ols_em = expand_reward_with_terminal_action_values(
                yhat_ols_em,
                num_states=env_spec.mdp.env_desc.num_states,
                num_actions=env_spec.mdp.env_desc.num_actions,
                terminal_states=terminal_states,
            )
        meta["ols_iters"] = iters
    else:
        logging.info(
            "Matrix is ill defined. Skipping reward estimation for %s: %s",
            spec["name"],
            spec["args"],
        )
    return {
        "least": yhat_lstsq,
        "ols_em": yhat_ols_em,
        "episodes": episode,
        "steps": steps,
        "full_rank": mapper._estimation_buffer.is_full_rank,
        "samples": mapper._estimation_buffer.matrix.shape[0],
        "data": {
            "lhs": mapper._estimation_buffer.matrix,
            "rhs": mapper._estimation_buffer.rhs,
        },
        "buffer_size": mapper._estimation_buffer.buffer_size,
        "episode_visited_states_count": {
            "num_unique_states": list(num_visited_states_dist.keys()),
            "num_episodes": list(num_visited_states_dist.values()),
        },
        "meta": meta,
    }


def lstsq_reward_estimation(
    obs_matrix: np.ndarray, agg_rewards: np.ndarray
) -> np.ndarray:
    return opt.solve_least_squares(
        matrix=obs_matrix,
        rhs=agg_rewards,
    )


def ols_em_reward_estimation(
    obs_matrix: np.ndarray,
    agg_rewards: np.ndarray,
    accuracy: float = 1e-8,
    max_iters: int = 1_000_000,
    stop_check_interval: int = 100,
) -> Tuple[np.ndarray, int]:
    iteration = 1
    yhat_rewards = np.random.rand(obs_matrix.shape[1]).astype(np.float64)
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
        if np.all(delta < accuracy) or iteration >= max_iters:
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


def expand_reward_with_terminal_action_values(
    estimated_rewards: np.ndarray,
    num_states: int,
    num_actions: int,
    terminal_states: Set[int],
):
    pos = 0
    est_rewards_ext = np.zeros(
        num_states * num_actions,
        dtype=np.float64,
    )
    terminal_state_action_mask = np.zeros(
        shape=(
            num_states,
            num_actions,
        ),
        dtype=np.float64,
    )
    # factor in terminal states
    for state in terminal_states:
        for action in range(num_actions):
            terminal_state_action_mask[state, action] = 1
    ignore_factors_mask = np.reshape(terminal_state_action_mask, newshape=[-1])
    for pos, ignore_factor in enumerate(ignore_factors_mask):
        if ignore_factor == 1:
            est_rewards_ext[pos] = 0.0
        else:
            est_rewards_ext[pos] = estimated_rewards[pos]
            pos += 1
    return est_rewards_ext
