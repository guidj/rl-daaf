"""
In this module, we can do on-policy evaluation with
delayed aggregate feedback - for tabular problems.
"""


import copy
import dataclasses
import json
import logging
from typing import Any, Callable, Generator, Iterator, Optional, Set

import gymnasium as gym
import numpy as np
from rlplg import core, envplay
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies, policyeval

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
        initial_state_values=create_initial_values(env_spec.mdp.env_desc.num_states),
        learnign_args=experiment_task.experiment.learning_args,
        generate_steps_fn=task.create_generate_episodes_fn(mappers=traj_mappers),
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
                state_values = snapshot.values
                if episode % experiment_task.run_config.log_episode_frequency == 0:
                    logging.info(
                        "Run %d of experiment %s, Episode %d: %d steps",
                        experiment_task.run_id,
                        experiment_task.exp_id,
                        episode,
                        snapshot.steps,
                    )
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=np.nan,
                        info={
                            "state_values": state_values.tolist(),
                        },
                    )

            logging.info("\nEstimated values\n%s", state_values)
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
    initial_state_values: np.ndarray,
    learnign_args: expconfig.LearningArgs,
    generate_steps_fn: Callable[
        [gym.Env, core.PyPolicy, int],
        Generator[core.TrajectoryStep, None, None],
    ],
) -> Iterator[policyeval.PolicyEvalSnapshot]:
    """
    Runs policy evaluation with given algorithm, env, and policy spec.
    """
    if algorithm == constants.ONE_STEP_TD:
        return policyeval.onpolicy_one_step_td_state_values(
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
    elif algorithm == constants.NSTEP_TD:
        # To avoid misconfigured experiments (e.g. using an identity mapper
        # with the n-step DAAF aware evaluation fn) we verify the
        # mapper and functions match.
        if (
            daaf_config.traj_mapping_method
            == constants.DAAF_NSTEP_TD_UPDATE_MARK_MAPPER
        ):
            return nstep_td_state_values_on_aggregate_start_steps(
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
                generate_episodes=generate_steps_fn,
            )
        return policyeval.onpolicy_nstep_td_state_values(
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
            generate_episodes=generate_steps_fn,
        )

    elif algorithm == constants.FIRST_VISIT_MONTE_CARLO:
        return policyeval.onpolicy_first_visit_monte_carlo_state_values(
            policy=policy,
            environment=env_spec.environment,
            num_episodes=num_episodes,
            gamma=learnign_args.discount_factor,
            state_id_fn=env_spec.discretizer.state,
            initial_values=initial_state_values,
            generate_episodes=generate_steps_fn,
        )

    raise ValueError(f"Unsupported algorithm {algorithm}")


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


def nstep_td_state_values_on_aggregate_start_steps(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    nstep: int,
    state_id_fn: Callable[[Any], int],
    initial_values: np.ndarray,
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
) -> Generator[policyeval.PolicyEvalSnapshot, None, None]:
    """
    n-step TD learning.
    Estimates V(s) for a fixed policy pi.
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
    values = copy.deepcopy(initial_values)
    steps_counter = 0
    for episode in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, 1))
        final_step = np.iinfo(np.int64).max
        for step, _ in enumerate(experiences):
            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
                    final_step = step + 1
            tau = step - nstep + 1
            if tau >= 0 and experiences[tau].info["ok_nstep_tau"]:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step)
                returns = 0.0

                for i in range(min_idx, max_idx + 1):
                    returns += (gamma ** (i - tau - 1)) * experiences[i - 1].reward
                if tau + nstep < final_step:
                    returns += (gamma**nstep) * values[
                        state_id_fn(experiences[tau + nstep].observation),
                    ]
                state_id = state_id_fn(experiences[tau].observation)
                alpha = lrs(episode=episode, step=steps_counter)
                values[state_id] += alpha * (returns - values[state_id])
            steps_counter += 1
        # need to copy qtable because it's a mutable numpy array
        yield policyeval.PolicyEvalSnapshot(
            steps=len(experiences), values=copy.deepcopy(values)
        )


def create_eval_policy(
    env_spec: core.EnvSpec, daaf_config: expconfig.DaafConfig
) -> core.PyPolicy:
    """
    Creates a policy to be evaluated.
    """
    if daaf_config.policy_type == constants.OPTIONS_POLICY:
        return options.UniformlyRandomCompositeActionPolicy(
            actions=tuple(range(env_spec.mdp.env_desc.num_actions)),
            options_duration=daaf_config.reward_period,
        )
    elif daaf_config.policy_type == constants.SINGLE_STEP_POLICY:
        return policies.PyRandomPolicy(
            num_actions=env_spec.mdp.env_desc.num_actions,
            emit_log_probability=True,
        )
    raise ValueError(f"Unknown policy {daaf_config.policy_type}")
