"""
In this module, we can do on-policy evaluation using periodic cumulative
rewards - for tabular problems.

We do this by using trajectory mappers that behave as follows:
  - Identity: given a sequence of unknown steps, returns them as is
  - Single Step Mapper: given a sequence of N steps, returns each one separatently (single step)
  - Weighted Attribution Mapper: given a sequence of N steps, returns each step separately,
    the rewards as `R' = sum(R_{t}) / N`
  - Least Squares Mapper: given a sequence of N steps, projects them into an
    (observation, cumulatie reward) instance which it uses to estimate the reward
    for each (state, action) pair.
  - Cumulative Reward Mapper: given a  sequence of N steps, returns every step,
    and changes the Kth to contain to have the reward equal to the sum of rewards
    for the last N steps (i.e. simulates cumulative periodic rewards).

Generally, we estimate rewards to then estimate Q.
"""


import logging
from typing import Optional, Set

import numpy as np
from rlplg import core, tracking

from daaf import expconfig, task


def daaf_policy_evalution(
    run_id: str,
    env_spec: core.EnvSpec,
    num_episodes: int,
    learning_args: expconfig.LearningArgs,
    daaf_config: expconfig.DaafConfig,
    output_dir: str,
    log_episode_frequency: int,
) -> None:
    """
    Runs on-policy evaluation with delayed, aggregated, anonymous feedback.

    Args:
        run_id: an Id for the run.
        policy: a policy to be estimated.
        env_spec: configuration of the environment, and state/action mapping functions.
        num_episodes: number of episodes to estimate the policy.
        learning_args: algorithm arguments, e.g. discount factor.
        daaf_args: configuration of cumulative rewards, e.g. rewad period.
        output_dir: a path to write execution logs.
        log_episode_frequency: frequency for writing execution logs.
    """
    traj_mapper = task.create_trajectory_mapper(
        env_spec=env_spec,
        reward_period=daaf_config.reward_period,
        traj_mapping_method=daaf_config.traj_mapping_method,
        buffer_size_or_multiplier=(None, None),
    )

    # Policy Eval with DAAF
    logging.info("Starting DAAF Evaluation")
    policy = task.create_eval_policy(env_spec=env_spec, daaf_config=daaf_config)
    results = task.run_fn(
        policy=policy,
        env_spec=env_spec,
        num_episodes=num_episodes,
        algorithm=daaf_config.algorithm,
        initial_state_values=initial_values(env_spec.mdp.env_desc.num_states),
        learnign_args=learning_args,
        generate_steps_fn=task.create_generate_episodes_fn(mapper=traj_mapper),
    )
    with tracking.ExperimentLogger(
        output_dir,
        name=f"qpolicy/daaf/mapper-{daaf_config.traj_mapping_method}",
        params={
            "algorithm": daaf_config.algorithm,
            "alpha": learning_args.learning_rate,
            "gamma": learning_args.discount_factor,
            "epsilon": learning_args.epsilon,
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        for episode, (steps, state_values) in enumerate(results):
            if episode % log_episode_frequency == 0:
                logging.info(
                    "Task %s, Episode %d: %d steps",
                    run_id,
                    episode,
                    steps,
                )
                exp_logger.log(
                    episode=episode,
                    steps=steps,
                    returns=0.0,
                    metadata={
                        "qtable": state_values.tolist(),
                    },
                )
        try:
            logging.info("\nEstimated values\n%s", state_values)
        except NameError:
            logging.info("Zero episodes!")


def main(args: expconfig.ExperimentTask):
    """
    Entry point running online evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec = task.create_env_spec(
        problem=args.experiment.env_config.env_name,
        env_args=args.experiment.env_config.env_args,
    )
    daaf_policy_evalution(
        run_id=args.run_id,
        env_spec=env_spec,
        num_episodes=args.run_config.num_episodes,
        learning_args=args.experiment.learning_args,
        daaf_config=args.experiment.daaf_config,
        output_dir=args.run_config.output_dir,
        log_episode_frequency=args.run_config.log_episode_frequency,
    )
    env_spec.environment.close()


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
