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
from typing import Optional

import numpy as np
from rlplg import envspec, metrics, tracking
from rlplg.learning import utils
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import onpolicy

from daaf import constants, progargs, task
from daaf.policyeval import skipmissing


def cpr_policy_evalution(
    run_id: str,
    policy: policies.PyQGreedyPolicy,
    env_spec: envspec.EnvSpec,
    num_episodes: int,
    num_states: int,
    num_actions: int,
    control_args: progargs.ControlArgs,
    cpr_args: progargs.CPRArgs,
    ref_qtable: np.ndarray,
    output_dir: str,
    log_steps: int,
) -> None:
    """
    Runs on-policy evaluation under cumulative periodic rewards.

    Args:
        run_id: an Id for the run.
        policy: a policy to be estimated.
        env_spec: configuration of the environment, and state/action mapping functions.
        num_episodes: number of episodes to estimate the policy.
        num_states: number of states in the problem.
        num_actions: number of actions in the problem.
        control_args: algorithm arguments, e.g. discount factor.
        cpr_args: configuration of cumulative rewards, e.g. rewad period.
        ref_qtable: the known value of the policy, to compare with the estimate.
        output_dir: a path to write execution logs.
        log_steps: frequency for writing execution logs.
    """
<<<<<<< HEAD:src/daaf/periodic_reward/onpolicy_eval.py
    mapper_fn = task.create_cumulative_step_mapper_fn(
=======
    mapper_fn = task.create_aggregate_reward_step_mapper_fn(
>>>>>>> d305522 (Refactors common module (#3)):src/daaf/policyeval/evaluation.py
        env_spec=env_spec,
        num_states=num_states,
        num_actions=num_actions,
        reward_period=cpr_args.reward_period,
        cu_step_method=cpr_args.cu_step_mapper,
        buffer_size_or_multiplier=(
            cpr_args.buffer_size,
            cpr_args.buffer_size_multiplier,
        ),
    )
    generate_steps_fn = task.create_generate_nstep_episodes_fn(mapper=mapper_fn)

    initial_table = utils.initial_table(
        num_states=num_states,
        num_actions=num_actions,
    )

    logging.info("Starting CPR Evaluation")

    # Policy Eval with CPR
    if cpr_args.cu_step_mapper == constants.CUMULATIVE_REWARD_MAPPER:
        results = skipmissing.cpr_sarsa_prediction(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            alpha=control_args.alpha,
            gamma=control_args.gamma,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=initial_table,
            reward_period=cpr_args.reward_period,
            generate_episodes=generate_steps_fn,
        )
    else:
        results = onpolicy.sarsa_action_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            alpha=control_args.alpha,
            gamma=control_args.gamma,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=initial_table,
            generate_episodes=generate_steps_fn,
        )

    with tracking.ExperimentLogger(
        output_dir,
        name=f"qpolicy/cpr/mapper-{cpr_args.cu_step_mapper}",
        params={
            "algorithm": "SARSA/On-Policy",
            "alpha": control_args.alpha,
            "gamma": control_args.gamma,
            "epsilon": control_args.epsilon,
            "buffer_size": cpr_args.buffer_size_multiplier,
        },
    ) as exp_logger:
        qtable: Optional[np.ndarray] = None
        for episode, (steps, qtable) in enumerate(results):
            rmse = metrics.rmse(pred=qtable, actual=ref_qtable)
            rmsle = metrics.rmsle(pred=qtable, actual=ref_qtable, translate=True)
            mean_error = metrics.mean_error(pred=qtable, actual=ref_qtable)
            pearson_corr, _ = metrics.pearson_correlation(
                pred=qtable, actual=ref_qtable
            )
            spearman_corr, _ = metrics.spearman_correlation(
                pred=qtable, actual=ref_qtable
            )
            if episode % log_steps == 0:
                logging.info(
                    "Task %s, Episode %d: %d steps, %f RMSE",
                    run_id,
                    episode,
                    steps,
                    rmse,
                )
                exp_logger.log(
                    episode=episode,
                    steps=steps,
                    returns=0.0,
                    metadata={
                        "qtable": qtable.tolist(),
                        "rmse": str(rmse),
                        "rmsle": str(rmsle),
                        "mean_error": str(mean_error),
                        "pearson_corr": str(pearson_corr),
                        "spearman_corr": str(spearman_corr),
                    },
                )
        try:
            logging.info("\nBaseline Q-table\n%s", ref_qtable)
            logging.info("\nEstimated Q-table\n%s", qtable)
        except NameError:
            logging.info("Zero episodes!")


<<<<<<< HEAD:src/daaf/periodic_reward/onpolicy_eval.py
def main(args: task.Args):
=======
def main(args: progargs.ExperimentArgs):
>>>>>>> d305522 (Refactors common module (#3)):src/daaf/policyeval/evaluation.py
    """
    Entry point running online evaluation for CPR.

    Args:
        args: configuration for execution.
    """
    # init env and agent
<<<<<<< HEAD
    env_spec, mdp = task.create_problem_spec(
<<<<<<< HEAD:src/daaf/periodic_reward/onpolicy_eval.py
        problem=args.problem,
        env_args=args.problem_args,
=======
=======
    env_spec, mdp = task.create_env_spec_and_mdp(
>>>>>>> e918ed8 (Renames problem to env-name in evaluation module)
        problem=args.env_name,
        env_args=args.env_args,
>>>>>>> d305522 (Refactors common module (#3)):src/daaf/policyeval/evaluation.py
        mdp_stats_path=args.mdp_stats_path,
        mdp_stats_num_episodes=args.mdp_stats_num_episodes,
    )
    state_action_values = task.dynamic_prog_estimation(
        env_spec=env_spec, mdp=mdp, control_args=args.control_args
    )
    policy = policies.PyRandomPolicy(
        time_step_spec=env_spec.environment.time_step_spec(),
        action_spec=env_spec.environment.action_spec(),
        num_actions=mdp.env_desc().num_actions,
        emit_log_probability=True,
    )
    cpr_policy_evalution(
        run_id=args.run_id,
        policy=policy,
        env_spec=env_spec,
        num_episodes=args.num_episodes,
        num_states=mdp.env_desc().num_states,
        num_actions=mdp.env_desc().num_actions,
        control_args=args.control_args,
        cpr_args=args.cpr_args,
        ref_qtable=state_action_values.action_values,
        output_dir=args.output_dir,
        log_steps=args.log_steps,
    )
    env_spec.environment.close()


if __name__ == "__main__":
    main(task.parse_args())
