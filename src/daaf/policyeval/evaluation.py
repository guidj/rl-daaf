"""
In this module, we can do on-policy evaluation with
delayed aggregate feedback - for tabular problems.
"""


import dataclasses
import logging
from typing import Callable, Generator, Iterator, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from rlplg import core
from rlplg.learning.opt import schedules
from rlplg.learning.tabular.evaluation import onpolicy

from daaf import constants, expconfig, task, utils


def run_fn(experiment_task: expconfig.ExperimentTask):
    """
    Entry point running on-policy evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec = task.create_env_spec(
        problem=experiment_task.experiment.env_config.name,
        env_args=experiment_task.experiment.env_config.args,
    )
    traj_mapper = task.create_trajectory_mapper(
        env_spec=env_spec,
        reward_period=experiment_task.experiment.daaf_config.reward_period,
        traj_mapping_method=experiment_task.experiment.daaf_config.traj_mapping_method,
        buffer_size_or_multiplier=(None, None),
    )
    # Policy Eval with DAAF
    logging.info("Starting DAAF Evaluation")
    policy = task.create_eval_policy(
        env_spec=env_spec, daaf_config=experiment_task.experiment.daaf_config
    )
    results = evaluate_policy(
        policy=policy,
        env_spec=env_spec,
        num_episodes=experiment_task.run_config.num_episodes,
        algorithm=experiment_task.experiment.daaf_config.algorithm,
        initial_state_values=initial_values(env_spec.mdp.env_desc.num_states),
        learnign_args=experiment_task.experiment.learning_args,
        generate_steps_fn=task.create_generate_episodes_fn(mapper=traj_mapper),
    )
    with utils.ExperimentLogger(
        experiment_task.run_config.output_dir,
        name=experiment_task.run_id,
        params={
            **dataclasses.asdict(experiment_task.experiment.daaf_config),
            **dataclasses.asdict(experiment_task.experiment.learning_args),
            **experiment_task.context,
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        for episode, (steps, state_values) in enumerate(results):
            if episode % experiment_task.run_config.log_episode_frequency == 0:
                logging.info(
                    "Task %s, Episode %d: %d steps",
                    experiment_task.run_id,
                    episode,
                    steps,
                )
                exp_logger.log(
                    episode=episode,
                    steps=steps,
                    returns=np.nan,
                    info={
                        "state_values": state_values.tolist(),
                    },
                )
        try:
            logging.info("\nEstimated values\n%s", state_values)
        except NameError:
            logging.info("Zero episodes!")
    env_spec.environment.close()


def evaluate_policy(
    policy: core.PyPolicy,
    env_spec: core.EnvSpec,
    num_episodes: int,
    algorithm: str,
    initial_state_values: np.ndarray,
    learnign_args: expconfig.LearningArgs,
    generate_steps_fn: Callable[
        [gym.Env, core.PyPolicy, int],
        Generator[core.TrajectoryStep, None, None],
    ],
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Runs policy evaluation with given algorithm, env, and policy spec.
    """
    if algorithm == constants.ONE_STEP_TD:
        results = onpolicy.one_step_td_state_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            lrs=schedules.LearningRateSchedule(
                initial_learning_rate=learnign_args.learning_rate,
                schedule=task.constant_learning_rate,
            ),
            gamma=learnign_args.discount_factor,
            state_id_fn=env_spec.discretizer.state,
            initial_values=initial_state_values,
            generate_episodes=generate_steps_fn,
        )
    elif algorithm == constants.FIRST_VISIT_MONTE_CARLO:
        results = onpolicy.first_visit_monte_carlo_state_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            gamma=learnign_args.discount_factor,
            state_id_fn=env_spec.discretizer.state,
            initial_values=initial_state_values,
            generate_episodes=generate_steps_fn,
        )
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}")

    return results


def initial_values(
    num_states: int,
    dtype: np.dtype = np.float64,
    random: bool = False,
    terminal_states: Optional[Set[int]] = None,
) -> np.ndarray:
    """
    The value of terminal states should be zero.
    """
    if random:
        if terminal_states is None:
            logging.warning("Creating Q-table with no terminal states")

        vtable = np.random.rand(
            num_states,
        )
        vtable[list(terminal_states or [])] = 0.0
        return vtable.astype(dtype)
    return np.zeros(shape=(num_states,), dtype=dtype)
