"""
In this module, we can do on-policy evaluation with
delayed aggregate feedback - for tabular problems.
"""

import dataclasses
import json
import logging
from typing import Any, Iterator, Mapping, Optional, Set

import numpy as np
from rlplg import core
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies, policyeval

from daaf import constants, expconfig, options, task, utils
from daaf.evalexps import methods


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
    traj_mappers = task.create_trajectory_mappers(
        env_spec=env_spec,
        reward_period=experiment_task.experiment.daaf_config.reward_period,
        traj_mapping_method=experiment_task.experiment.daaf_config.traj_mapping_method,
        buffer_size_or_multiplier=(None, None),
        drop_truncated_feedback_episodes=experiment_task.experiment.daaf_config.drop_truncated_feedback_episodes,
    )
    returns_collector = task.returns_collection_mapper()
    traj_mappers = tuple([returns_collector] + list(traj_mappers))
    # Policy Eval with DAAF
    logging.info("Starting DAAF Evaluation Experiments")
    policy = create_eval_policy(
        env_spec=env_spec, daaf_config=experiment_task.experiment.daaf_config
    )
    results = evaluate_policy(
        policy=policy,
        env_spec=env_spec,
        daaf_config=experiment_task.experiment.daaf_config,
        num_episodes=experiment_task.run_config.num_episodes,
        algorithm=experiment_task.experiment.daaf_config.algorithm,
        learnign_args=experiment_task.experiment.learning_args,
        generate_steps_fn=task.create_generate_episode_fn(mappers=traj_mappers),
    )
    env_info: Mapping[str, Any] = {
        "env": {
            "name": env_spec.name,
            "level": env_spec.level,
            "args": json.dumps(experiment_task.experiment.env_config.args),
        },
    }
    with utils.ExperimentLogger(
        log_dir=experiment_task.run_config.output_dir,
        exp_id=experiment_task.exp_id,
        run_id=experiment_task.run_id,
        params={
            **utils.json_from_dict(
                dataclasses.asdict(experiment_task.experiment.daaf_config),
                dict_encode_level=0,
            ),
            **dataclasses.asdict(experiment_task.experiment.learning_args),
            **experiment_task.context,
            **env_info,
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        try:
            for episode, snapshot in enumerate(results):
                state_values = snapshot.values
                if episode % experiment_task.run_config.log_episode_frequency == 0:
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=returns_collector.traj_returns[-1],
                        info={
                            "state_values": state_values.tolist(),
                        },
                    )

            logging.info(
                "\nEstimated values run %d of %s:\n%s",
                experiment_task.run_id,
                experiment_task.exp_id,
                state_values,
            )
        except Exception as err:
            raise RuntimeError(
                f"Task {experiment_task.exp_id}, run {experiment_task.run_id} failed"
            ) from err
    env_spec.environment.close()


def evaluate_policy(
    policy: core.PyPolicy,
    env_spec: core.EnvSpec,
    daaf_config: expconfig.DaafConfig,
    num_episodes: int,
    algorithm: str,
    learnign_args: expconfig.LearningArgs,
    generate_steps_fn: core.GeneratesEpisode,
) -> Iterator[policyeval.PolicyEvalSnapshot]:
    """
    Runs policy evaluation with given algorithm, env, and policy spec.
    """
    initial_state_values = create_initial_values(env_spec.mdp.env_desc.num_states)
    eval_fn: Optional[Iterator[policyeval.PolicyEvalSnapshot]] = None
    if algorithm == constants.ONE_STEP_TD:
        if daaf_config.traj_mapping_method == constants.DAAF_TRAJECTORY_MAPPER:
            eval_fn = methods.onpolicy_one_step_td_state_values_only_aggregate_updates(
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
                generate_episode=generate_steps_fn,
            )
        else:
            eval_fn = policyeval.onpolicy_one_step_td_state_values(
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
                generate_episode=generate_steps_fn,
            )
    elif algorithm == constants.NSTEP_TD:
        # To avoid misconfigured experiments (e.g. using an identity mapper
        # with the n-step DAAF aware evaluation fn) we verify the
        # mapper and functions match.
        if (
            daaf_config.traj_mapping_method
            == constants.DAAF_NSTEP_TD_UPDATE_MARK_MAPPER
        ):
            eval_fn = methods.nstep_td_state_values_on_aggregate_start_steps(
                policy=policy,
                environment=env_spec.environment,
                num_episodes=num_episodes,
                lrs=schedules.LearningRateSchedule(
                    initial_learning_rate=learnign_args.learning_rate,
                    schedule=task.constant_learning_rate,
                ),
                gamma=learnign_args.discount_factor,
                nstep=daaf_config.reward_period,
                state_id_fn=env_spec.discretizer.state,
                initial_values=initial_state_values,
                generate_episode=generate_steps_fn,
            )
        else:
            eval_fn = policyeval.onpolicy_nstep_td_state_values(
                policy=policy,
                environment=env_spec.environment,
                num_episodes=num_episodes,
                lrs=schedules.LearningRateSchedule(
                    initial_learning_rate=learnign_args.learning_rate,
                    schedule=task.constant_learning_rate,
                ),
                gamma=learnign_args.discount_factor,
                nstep=daaf_config.reward_period,
                state_id_fn=env_spec.discretizer.state,
                initial_values=initial_state_values,
                generate_episode=generate_steps_fn,
            )

    elif algorithm == constants.FIRST_VISIT_MONTE_CARLO:
        if daaf_config.traj_mapping_method == constants.DAAF_TRAJECTORY_MAPPER:
            eval_fn = methods.onpolicy_first_visit_monte_carlo_state_values_only_aggregate_updates(
                policy=policy,
                environment=env_spec.environment,
                num_episodes=num_episodes,
                gamma=learnign_args.discount_factor,
                state_id_fn=env_spec.discretizer.state,
                initial_values=initial_state_values,
                generate_episode=generate_steps_fn,
            )
        else:
            eval_fn = policyeval.onpolicy_first_visit_monte_carlo_state_values(
                policy=policy,
                environment=env_spec.environment,
                num_episodes=num_episodes,
                gamma=learnign_args.discount_factor,
                state_id_fn=env_spec.discretizer.state,
                initial_values=initial_state_values,
                generate_episode=generate_steps_fn,
            )

    if eval_fn is None:
        raise ValueError(f"Unsupported algorithm {algorithm}")
    return eval_fn


def create_initial_values(
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


def create_eval_policy(
    env_spec: core.EnvSpec, daaf_config: expconfig.DaafConfig
) -> core.PyPolicy:
    """
    Creates a policy to be evaluated.
    """
    if daaf_config.policy_type == constants.OPTIONS_POLICY:
        return options.UniformlyRandomCompositeActionPolicy(
            primitive_actions=tuple(range(env_spec.mdp.env_desc.num_actions)),
            options_duration=daaf_config.reward_period,
        )
    elif daaf_config.policy_type == constants.SINGLE_STEP_POLICY:
        return policies.PyRandomPolicy(
            num_actions=env_spec.mdp.env_desc.num_actions,
            emit_log_probability=True,
        )
    raise ValueError(f"Unknown policy {daaf_config.policy_type}")
