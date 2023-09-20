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
from rlplg import envspec, metrics, tracking
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import onpolicy

from daaf import constants, progargs, task


def daaf_policy_evalution(
    run_id: str,
    policy: policies.PyQGreedyPolicy,
    env_spec: envspec.EnvSpec,
    num_episodes: int,
    algorithm: str,
    num_states: int,
    num_actions: int,
    control_args: progargs.ControlArgs,
    daaf_args: progargs.DaafArgs,
    ref_state_values: np.ndarray,
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
        num_actions: number of actions in the problem.
        control_args: algorithm arguments, e.g. discount factor.
        daaf_args: configuration of cumulative rewards, e.g. rewad period.
        ref_state_values: the known value of the policy, to compare with the estimate.
        output_dir: a path to write execution logs.
        log_episode_frequency: frequency for writing execution logs.
    """
    mapper_fn = task.create_aggregate_reward_step_mapper_fn(
        env_spec=env_spec,
        num_states=num_states,
        num_actions=num_actions,
        reward_period=daaf_args.reward_period,
        cu_step_method=daaf_args.cu_step_mapper,
        buffer_size_or_multiplier=(
            daaf_args.buffer_size,
            daaf_args.buffer_size_multiplier,
        ),
    )
    generate_steps_fn = task.create_generate_nstep_episodes_fn(mapper=mapper_fn)

    logging.info("Starting DAAF Evaluation")

    # Policy Eval with DAAF
    if algorithm == constants.ONE_STEP_TD:
        results = onpolicy.one_step_td_state_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            lrs=schedules.LearningRateSchedule(
                initial_learning_rate=control_args.alpha,
                schedule=constant_learning_rate,
            ),
            gamma=control_args.gamma,
            state_id_fn=env_spec.discretizer.state,
            initial_values=initial_values(
                num_states=num_states,
            ),
            generate_episodes=generate_steps_fn,
        )
    elif algorithm == constants.FIRST_VISIT_MONTE_CARLO:
        results = onpolicy.first_visit_monte_carlo_state_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            gamma=control_args.gamma,
            state_id_fn=env_spec.discretizer.state,
            initial_values=initial_values(
                num_states=num_states,
            ),
            generate_episodes=generate_steps_fn,
        )
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}")

    with tracking.ExperimentLogger(
        output_dir,
        name=f"qpolicy/daaf/mapper-{daaf_args.cu_step_mapper}",
        params={
            "algorithm": algorithm,
            "alpha": control_args.alpha,
            "gamma": control_args.gamma,
            "epsilon": control_args.epsilon,
            "buffer_size": daaf_args.buffer_size_multiplier,
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        for episode, (steps, state_values) in enumerate(results):
            rmse = metrics.rmse(pred=state_values, actual=ref_state_values)
            rmsle = metrics.rmsle(
                pred=state_values, actual=ref_state_values, translate=True
            )
            mean_error = metrics.mean_error(pred=state_values, actual=ref_state_values)
            pearson_corr, _ = metrics.pearson_correlation(
                pred=state_values, actual=ref_state_values
            )
            spearman_corr, _ = metrics.spearman_correlation(
                pred=state_values, actual=ref_state_values
            )
            if episode % log_episode_frequency == 0:
                logging.info(
                    "Task %s, Episode %d: %d steps, %f RMSLE, %f RMSE",
                    run_id,
                    episode,
                    steps,
                    rmsle,
                    rmse,
                )
                exp_logger.log(
                    episode=episode,
                    steps=steps,
                    returns=0.0,
                    metadata={
                        "qtable": state_values.tolist(),
                        "rmse": str(rmse),
                        "rmsle": str(rmsle),
                        "mean_error": str(mean_error),
                        "pearson_corr": str(pearson_corr),
                        "spearman_corr": str(spearman_corr),
                    },
                )
        try:
            logging.info("\nBaseline values\n%s", ref_state_values)
            logging.info("\nEstimated values\n%s", state_values)
        except NameError:
            logging.info("Zero episodes!")


def main(args: progargs.ExperimentArgs):
    """
    Entry point running online evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec, mdp = task.create_env_spec_and_mdp(
        problem=args.env_name,
        env_args=args.env_args,
        mdp_stats_path=args.mdp_stats_path,
        mdp_stats_num_episodes=args.mdp_stats_num_episodes,
    )
    state_action_values = task.dynamic_prog_estimation(
        env_spec=env_spec, mdp=mdp, control_args=args.control_args
    )
    policy = policies.PyRandomPolicy(
        num_actions=mdp.env_desc().num_actions,
        emit_log_probability=True,
    )
    daaf_policy_evalution(
        run_id=args.run_id,
        policy=policy,
        env_spec=env_spec,
        num_episodes=args.num_episodes,
        algorithm=args.algorithm,
        num_states=mdp.env_desc().num_states,
        num_actions=mdp.env_desc().num_actions,
        control_args=args.control_args,
        daaf_args=args.daaf_args,
        ref_state_values=state_action_values.state_values,
        output_dir=args.output_dir,
        log_episode_frequency=args.log_episode_frequency,
    )
    env_spec.environment.close()


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    """
    Returns the initial learning rate.
    """
    del episode
    del step
    return initial_lr


def initial_values(
    num_states: int,
    dtype: np.dtype = np.float32,
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


if __name__ == "__main__":
    main(task.parse_args())
