"""
In this module, we can do off-policy evaluation using periodic cumulative
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

import numpy as np
from tf_agents.policies import py_policy

from rlplg import envspec, metrics, tracking
from rlplg.learning import utils
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import offpolicy
from daaf.periodic_reward import baseline, common, constants, progargs


def cpr_policy_evalution(
    run_id: str,
    policy: policies.PyPolicy,
    collect_policy: py_policy.PyPolicy,
    env_spec: envspec.EnvSpec,
    num_episodes: int,
    num_states: int,
    num_actions: int,
    control_args: progargs.ControlArgs,
    cpr_args: progargs.CPRArgs,
    ref_qtable: np.ndarray,
    output_dir: str,
    log_steps: int,
):
    """
    Runs on-policy evaluation under cumulative periodic rewards.

    Args:
        run_id: an Id for the run.
        policy: a policy to be estimated.
        collect_policy: a policy to generate data to estimate `policy`.
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
    mapper_fn = common.create_cumulative_step_mapper_fn(
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
    generate_steps_fn = common.create_generate_nstep_episodes_fn(mapper=mapper_fn)

    if cpr_args.cu_step_mapper == constants.CUMULATIVE_REWARD_MAPPER:
        results = baseline.cpr_nstep_sarsa_prediction(
            policy=policy,
            collect_policy=collect_policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            alpha=control_args.alpha,
            gamma=control_args.gamma,
            nstep=1,
            policy_probability_fn=utils.policy_prob_fn,
            collect_policy_probability_fn=utils.collect_policy_prob_fn,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=utils.initial_table(
                num_states=num_states,
                num_actions=num_actions,
                terminal_states=set([num_states - 1]),
            ),
            reward_period=cpr_args.reward_period,
            generate_episodes=generate_steps_fn,
        )
    else:
        results = offpolicy.nstep_sarsa_action_values(
            policy=policy,
            collect_policy=collect_policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            alpha=control_args.alpha,
            gamma=control_args.gamma,
            nstep=1,
            policy_probability_fn=utils.policy_prob_fn,
            collect_policy_probability_fn=utils.collect_policy_prob_fn,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=utils.initial_table(
                num_states=num_states,
                num_actions=num_actions,
                terminal_states=set([num_states - 1]),
            ),
            generate_episodes=generate_steps_fn,
        )

    with tracking.ExperimentLogger(
        output_dir,
        name=f"qpolicy/cpr/mapper-{cpr_args.cu_step_mapper}",
        params={
            "algorithm": "n-step SARSA/Off-Policy",
            "alpha": control_args.alpha,
            "gamma": control_args.gamma,
            "epsilon": control_args.epsilon,
            "buffer_size": cpr_args.buffer_size_multiplier,
        },
    ) as exp_logger:
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
            logging.info(
                "\n---Estimation---\n%s\n---Policy---\n%s",
                np.around(qtable, 3),
                np.around(ref_qtable, 3),
            )
        except NameError:
            logging.info("Zero episodes!")


def main(args: common.Args):
    """
    Entry point running offline evaluation for CPR.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec, mdp = common.create_problem_spec(
        problem=args.problem,
        env_args=args.problem_args,
        mdp_stats_path=args.mdp_stats_path,
        mdp_stats_num_episodes=args.mdp_stats_num_episodes,
    )
    dp_state_action_values = common.dynamic_prog_estimation(
        env_spec=env_spec, mdp=mdp, control_args=args.control_args
    )
    policy = policies.PyRandomPolicy(
        time_step_spec=env_spec.environment.time_step_spec(),
        action_spec=env_spec.environment.action_spec(),
        num_actions=mdp.env_desc().num_states,
        emit_log_probability=True,
    )
    collect_policy = policies.PyRandomPolicy(
        time_step_spec=env_spec.environment.time_step_spec(),
        action_spec=env_spec.environment.action_spec(),
        num_actions=mdp.env_desc().num_actions,
        emit_log_probability=True,
    )
    cpr_policy_evalution(
        run_id=args.run_id,
        policy=policy,
        collect_policy=collect_policy,
        env_spec=env_spec,
        num_episodes=args.num_episodes,
        num_states=mdp.env_desc().num_states,
        num_actions=mdp.env_desc().num_actions,
        control_args=args.control_args,
        cpr_args=args.cpr_args,
        ref_qtable=dp_state_action_values.action_values,
        output_dir=args.output_dir,
        log_steps=args.log_steps,
    )
    env_spec.environment.close()


if __name__ == "__main__":
    main(common.parse_args())
