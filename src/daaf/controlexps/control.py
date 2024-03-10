"""
In this module, we can do on-policy evaluation with
delayed aggregate feedback - for tabular problems.
"""

import copy
import dataclasses
import json
import logging
from typing import Any, Callable, Dict, Generator, Iterator, Optional, Set

import gymnasium as gym
import numpy as np
from rlplg import core, envplay
from rlplg.learning import utils as rlplg_utils
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies, policycontrol

from daaf import constants, expconfig, options, task, utils


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
    # Collect returns on underlying MDP
    # before other mappers change it.
    returns_collector = task.returns_collection_mapper()
    traj_mappers = tuple([returns_collector] + list(traj_mappers))
    logging.info("Starting DAAF Control Experiments")
    results = policy_control(
        env_spec=env_spec,
        daaf_config=experiment_task.experiment.daaf_config,
        num_episodes=experiment_task.run_config.num_episodes,
        learnign_args=experiment_task.experiment.learning_args,
        generate_steps_fn=task.create_generate_episode_fn(mappers=traj_mappers),
    )
    with utils.ExperimentLogger(
        log_dir=experiment_task.run_config.output_dir,
        exp_id=experiment_task.exp_id,
        run_id=experiment_task.run_id,
        params={
            **dataclasses.asdict(experiment_task.experiment.daaf_config),
            **dataclasses.asdict(experiment_task.experiment.learning_args),
            **experiment_task.context,
            "env": {
                "name": env_spec.name,
                "level": env_spec.level,
                "args": json.dumps(experiment_task.experiment.env_config.args),
            },
        },
    ) as exp_logger:
        state_values: Optional[np.ndarray] = None
        try:
            for episode, snapshot in enumerate(results):
                state_values = snapshot.action_values
                if episode % experiment_task.run_config.log_episode_frequency == 0:
                    logging.info(
                        "Run %d of experiment %s, Episode %d: %d steps, %f returns",
                        experiment_task.run_id,
                        experiment_task.exp_id,
                        episode,
                        snapshot.steps,
                        returns_collector.traj_returns[-1],
                    )
                    mean_returns = np.mean(returns_collector.traj_returns)
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=mean_returns,
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


def policy_control(
    env_spec: core.EnvSpec,
    daaf_config: expconfig.DaafConfig,
    num_episodes: int,
    learnign_args: expconfig.LearningArgs,
    generate_steps_fn: Callable[
        [gym.Env, core.PyPolicy, int],
        Generator[core.TrajectoryStep, None, None],
    ],
) -> Iterator[policycontrol.PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    lrs = schedules.LearningRateSchedule(
        initial_learning_rate=learnign_args.learning_rate,
        schedule=task.constant_learning_rate,
    )
    initial_action_values, create_egreedy_policy = create_qtable_and_egreedy_policy(
        env_spec=env_spec, daaf_config=daaf_config
    )
    if daaf_config.algorithm == constants.SARSA:
        return policycontrol.onpolicy_sarsa_control(
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
            return onpolicy_nstep_sarsa_on_aggregate_start_steps_control(
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
        return policycontrol.onpolicy_nstep_sarsa_control(
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
        return policycontrol.onpolicy_qlearning_control(
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
    dtype: np.dtype = np.float64,
    random: bool = False,
    terminal_states: Optional[Set[int]] = None,
) -> np.ndarray:
    if daaf_config.policy_type == constants.SINGLE_STEP_POLICY:
        qtable = _create_initial_values(
            num_states=env_spec.mdp.env_desc.num_states,
            num_actions=env_spec.mdp.env_desc.num_actions,
            dtype=dtype,
            random=random,
            terminal_states=terminal_states,
        )

        return qtable, rlplg_utils.create_egreedy_policy
    elif daaf_config.policy_type == constants.OPTIONS_POLICY:
        num_options = env_spec.mdp.env_desc.num_actions**daaf_config.reward_period
        qtable = _create_initial_values(
            num_states=env_spec.mdp.env_desc.num_states,
            num_actions=num_options,
            dtype=dtype,
            random=random,
            terminal_states=terminal_states,
        )

        return qtable, create_options_egreedy_policy_fn(
            env_desc=env_spec.mdp.env_desc, options_duration=daaf_config.reward_period
        )

    raise ValueError(f"Unsupported policy type {daaf_config.policy_type}")


def _create_initial_values(
    num_states: int,
    num_actions: int,
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

        qtable = np.random.rand(num_states, num_actions)
        qtable[list(terminal_states or []), :] = 0.0
        return qtable.astype(dtype)
    return np.zeros(shape=(num_states, num_actions), dtype=dtype)


def create_options_egreedy_policy_fn(env_desc: core.EnvDesc, options_duration: int):
    def create_options_egreedy_policy(
        initial_qtable: np.ndarray,
        state_id_fn: Callable[[Any], int],
        epsilon: float,
    ) -> policies.PyEpsilonGreedyPolicy:
        greedy_policy = policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        )
        return options.OptionsQGreedyPolicy(
            policy=greedy_policy,
            options_duration=options_duration,
            primitive_actions=range(env_desc.num_actions),
            epsilon=epsilon,
        )

    return create_options_egreedy_policy


def onpolicy_nstep_sarsa_on_aggregate_start_steps_control(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    nstep: int,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    create_egreedy_policy: Callable[
        [np.ndarray, Callable[[Any], int], float], policies.PyEpsilonGreedyPolicy
    ],
    generate_episode: Callable[
        [
            gym.Env,
            core.PyPolicy,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episode,
) -> Generator[policycontrol.PolicyControlSnapshot, None, None]:
    """
    n-step SARSA.
    Policy control.
    Adapted for MDP with options.
    Only states that match the start of an option
    get updated with the reward observed at the end
    of the option, which matches with the end of the
    aggregate feedback window.
    Source: https://en.wikipedia.org/wiki/Temporal_difference_learning

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the V(s) table.
        initial_values: Initial state-value estimates.
        generate_episodes: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract reward access references by one.
    """
    qtable = copy.deepcopy(initial_qtable)
    egreedy_policy = create_egreedy_policy(qtable, state_id_fn, epsilon)
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, egreedy_policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        # In the absence of a step as terminal
        # or truncated, `empty_steps` prevents
        # infinite loops
        empty_steps = 0
        returns = 0.0
        while True:
            if step > final_step or empty_steps > nstep:
                break
            try:
                traj_step = next(trajectory)
            except StopIteration:
                empty_steps += 1
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1
                returns += traj_step.reward

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - nstep, None)

            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
                    final_step = step + 1
            tau = step - nstep + 1
            if tau >= 0 and experiences[tau].info["ok_nstep_tau"]:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step)
                nstep_returns = 0.0
                for i in range(min_idx, max_idx + 1):
                    nstep_returns += (gamma ** (i - tau - 1)) * experiences[
                        i - 1
                    ].reward
                if tau + nstep < final_step:
                    nstep_returns += (gamma**nstep) * qtable[
                        state_id_fn(experiences[tau + nstep].observation),
                        action_id_fn(experiences[tau + nstep].action),
                    ]
                state_id = state_id_fn(experiences[tau].observation)
                action_id = action_id_fn(experiences[tau].action)
                alpha = lrs(episode=episode, step=steps_counter)
                qtable[state_id, action_id] += alpha * (
                    nstep_returns - qtable[state_id, action_id]
                )
                # update the qtable before generating the
                # next step in the trajectory
                setattr(
                    egreedy_policy.exploit_policy,
                    "_state_action_value_table",
                    qtable,
                )
                steps_counter += 1
            step += 1
        # need to copy qtable because it's a mutable numpy array
        yield policycontrol.PolicyControlSnapshot(
            steps=traj_step_idx, returns=returns, action_values=copy.deepcopy(qtable)
        )
