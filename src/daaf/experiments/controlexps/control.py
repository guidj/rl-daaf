"""
This module contains components for policy control
with delayed aggregate and anonymous feedback,
for tabular problems.
"""

import dataclasses
import json
import logging
from typing import Any, Callable, Iterator, Mapping, Optional, Set, Tuple

import numpy as np
from numpy.typing import DTypeLike

from daaf import constants, core, expconfig, task, utils
from daaf.core import GeneratesEpisode
from daaf.learning import opt
from daaf.learning import utils as learning_utils
from daaf.learning.tabular import policies, policycontrol


def run_fn(experiment_run: expconfig.ExperimentRun):
    """
    Entry point running on-policy evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec = task.create_env_spec(
        problem=experiment_run.experiment.env_config.name,
        env_args=experiment_run.experiment.env_config.args,
    )
    traj_mappers = task.create_trajectory_mappers(
        env_spec=env_spec,
        reward_period=experiment_run.experiment.daaf_config.reward_period,
        traj_mapping_method=experiment_run.experiment.daaf_config.traj_mapping_method,
        buffer_size_or_multiplier=(None, None),
        drop_truncated_feedback_episodes=experiment_run.experiment.daaf_config.drop_truncated_feedback_episodes,
    )
    # Collect returns on underlying MDP
    # before other mappers change it.
    returns_collector = task.returns_collection_mapper()
    traj_mappers = tuple([returns_collector] + list(traj_mappers))
    logging.debug("Starting DAAF Control Experiments")
    results = policy_control(
        env_spec=env_spec,
        daaf_config=experiment_run.experiment.daaf_config,
        num_episodes=experiment_run.run_config.num_episodes,
        learnign_args=experiment_run.experiment.learning_args,
        generate_steps_fn=task.create_generate_episode_fn(mappers=traj_mappers),
    )
    env_info: Mapping[str, Any] = {
        "env": {
            "name": env_spec.name,
            "uid": env_spec.uid,
            "args": json.dumps(experiment_run.experiment.env_config.args),
        },
    }
    with utils.ExperimentLogger(
        log_dir=experiment_run.run_config.output_dir,
        exp_id=experiment_run.exp_id,
        run_id=experiment_run.run_id,
        params={
            **env_info,
            **utils.json_from_dict(
                dataclasses.asdict(experiment_run.experiment.daaf_config),
                dict_encode_level=0,
            ),
            **dataclasses.asdict(experiment_run.experiment.learning_args),
            **experiment_run.context,
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        state_actions: Optional[np.ndarray] = None
        try:
            for episode, snapshot in enumerate(results):
                state_values = np.max(snapshot.action_values, axis=1)
                state_actions = np.argmax(snapshot.action_values, axis=1)
                if episode % experiment_run.run_config.log_episode_frequency == 0:
                    mean_returns = np.mean(returns_collector.traj_returns)
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=mean_returns.item(),
                        # Action values can be large tables
                        # especially for options policies
                        # so we log state values and best actions
                        info={
                            "state_values": state_values.tolist(),  # type: ignore
                            "action_argmax": state_actions.tolist(),  # type: ignore
                        },
                    )

            logging.debug(
                "\nEstimated values run %d of %s:\n%s",
                experiment_run.run_id,
                experiment_run.exp_id,
                state_values,
            )
        except Exception as err:
            raise RuntimeError(
                f"Task {experiment_run.exp_id}, run {experiment_run.run_id} failed"
            ) from err
    env_spec.environment.close()


def policy_control(
    env_spec: core.EnvSpec,
    daaf_config: expconfig.DaafConfig,
    num_episodes: int,
    learnign_args: expconfig.LearningArgs,
    generate_steps_fn: GeneratesEpisode,
) -> Iterator[policycontrol.PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    lrs = opt.LearningRateSchedule(
        initial_learning_rate=learnign_args.learning_rate,
        schedule=task.constant_learning_rate,
    )
    initial_action_values, create_egreedy_policy = create_qtable_and_egreedy_policy(
        env_spec=env_spec, daaf_config=daaf_config
    )
    if daaf_config.algorithm == constants.SARSA:
        if daaf_config.traj_mapping_method == constants.DAAF_TRAJECTORY_MAPPER:
            sarsa_fn = policycontrol.onpolicy_sarsa_control_only_aggregate_updates
        else:
            sarsa_fn = policycontrol.onpolicy_sarsa_control
        return sarsa_fn(
            environment=env_spec.environment,
            num_episodes=num_episodes,
            lrs=lrs,
            gamma=learnign_args.discount_factor,
            epsilon=learnign_args.epsilon,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            create_egreedy_policy=create_egreedy_policy,
            initial_qtable=initial_action_values,
            generate_episode=generate_steps_fn,
        )
    elif daaf_config.algorithm == constants.NSTEP_SARSA:
        # To avoid misconfigured experiments (e.g. using an identity mapper
        # with the n-step DAAF aware evaluation fn) we verify the
        # mapper and functions match.
        if (
            daaf_config.traj_mapping_method
            == constants.DAAF_NSTEP_TD_UPDATE_MARK_MAPPER
        ):
            nstep_fn = (
                policycontrol.onpolicy_nstep_sarsa_on_aggregate_start_steps_control
            )
        else:
            nstep_fn = policycontrol.onpolicy_nstep_sarsa_control

        return nstep_fn(
            environment=env_spec.environment,
            num_episodes=num_episodes,
            lrs=lrs,
            gamma=learnign_args.discount_factor,
            epsilon=learnign_args.epsilon,
            nstep=daaf_config.algorithm_args["nstep"],
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=initial_action_values,
            create_egreedy_policy=create_egreedy_policy,
            generate_episode=generate_steps_fn,
        )

    elif daaf_config.algorithm == constants.Q_LEARNING:
        if daaf_config.traj_mapping_method == constants.DAAF_TRAJECTORY_MAPPER:
            qlearn_fn = policycontrol.onpolicy_qlearning_control_only_aggregate_updates
        else:
            qlearn_fn = policycontrol.onpolicy_qlearning_control
        return qlearn_fn(
            environment=env_spec.environment,
            num_episodes=num_episodes,
            lrs=lrs,
            gamma=learnign_args.discount_factor,
            epsilon=learnign_args.epsilon,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            initial_qtable=initial_action_values,
            create_egreedy_policy=create_egreedy_policy,
            generate_episode=generate_steps_fn,
        )

    raise ValueError(f"Unsupported algorithm {daaf_config.algorithm}")


def create_qtable_and_egreedy_policy(
    env_spec: core.EnvSpec,
    daaf_config: expconfig.DaafConfig,
    dtype: DTypeLike = np.float64,
    random: bool = False,
    terminal_states: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, policycontrol.CreatesEGreedyPolicy]:
    if daaf_config.policy_type == constants.SINGLE_STEP_POLICY:
        qtable = _create_initial_values(
            num_states=env_spec.mdp.env_space.num_states,
            num_actions=env_spec.mdp.env_space.num_actions,
            dtype=dtype,
            random=random,
            terminal_states=terminal_states,
        )

        return qtable, learning_utils.create_egreedy_policy
    elif daaf_config.policy_type == constants.OPTIONS_POLICY:
        num_options = env_spec.mdp.env_space.num_actions**daaf_config.reward_period
        qtable = _create_initial_values(
            num_states=env_spec.mdp.env_space.num_states,
            num_actions=num_options,
            dtype=dtype,
            random=random,
            terminal_states=terminal_states,
        )

        return qtable, create_options_egreedy_policy_fn(
            env_space=env_spec.mdp.env_space, options_duration=daaf_config.reward_period
        )

    raise ValueError(f"Unsupported policy type {daaf_config.policy_type}")


def _create_initial_values(
    num_states: int,
    num_actions: int,
    dtype: DTypeLike = np.float64,
    random: bool = False,
    terminal_states: Optional[Set[int]] = None,
) -> np.ndarray:
    """
    The value of terminal states should be zero.
    """
    if random:
        if terminal_states is None:
            logging.warning("Creating Q-table with no terminal states")

        qtable = np.random.rand(num_states, num_actions)
        qtable[list(terminal_states or []), :] = 0.0
        return qtable.astype(dtype)
    return np.zeros(shape=(num_states, num_actions), dtype=dtype)


def create_options_egreedy_policy_fn(env_space: core.EnvSpace, options_duration: int):
    def create_options_egreedy_policy(
        initial_qtable: np.ndarray,
        state_id_fn: Callable[[Any], int],
        epsilon: float,
    ) -> policies.PyEpsilonGreedyPolicy:
        greedy_policy = policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        )
        return policies.OptionsQGreedyPolicy(
            policy=greedy_policy,
            options_duration=options_duration,
            primitive_actions=range(env_space.num_actions),
            epsilon=epsilon,
        )

    return create_options_egreedy_policy
