"""
Functions relying on ReplayBuffer are for TF classes (agents, environment, etc).
Generators are for Py classes (agents, environment, etc).
"""
import argparse
import dataclasses
import json
import logging
from typing import Any, Callable, Generator, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from rlplg import core, envplay, envspec, envsuite, runtime
from rlplg.learning import utils
from rlplg.learning.tabular import dynamicprog, markovdp, policies

from daaf import constants, progargs, replay_mapper
from daaf.envstats import envstats


@dataclasses.dataclass(frozen=True)
class StateActionValues:
    """
    Class holds state-value and action-value functions.
    """

    state_values: Optional[np.ndarray]
    action_values: Optional[np.ndarray]


def parse_args() -> progargs.ExperimentArgs:
    """
    Parses experiment arguments.
    """
    arg_parser = argparse.ArgumentParser(
        prog="Policy estimation with delayed aggregated anonymous feedback."
    )
    arg_parser.add_argument("--env-name", type=str, required=True)
    arg_parser.add_argument("--env-args", type=str, default=None)
    arg_parser.add_argument("--run-id", type=str, default=runtime.run_id())
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--reward-period", type=int, default=2)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument(
        "--algorithm",
        type=str,
        choices=constants.ALGORITHMS,
        default=constants.ONE_STEP_TD,
    )
    arg_parser.add_argument(
        "--cu-step-mapper",
        type=str,
        default=constants.REWARD_ESTIMATION_LSQ_MAPPER,
        choices=constants.CU_MAPPER_METHODS,
    )
    arg_parser.add_argument("--control-epsilon", type=float, default=1.0)
    arg_parser.add_argument("--control-alpha", type=float, default=0.1)
    arg_parser.add_argument("--control-gamma", type=float, default=1.0)
    arg_parser.add_argument("--buffer-size", type=int, default=None)
    arg_parser.add_argument("--buffer-size-multiplier", type=int, default=None)
    arg_parser.add_argument("--log-episode-frequency", type=int, default=1)
    arg_parser.add_argument("--mdp-stats-path", type=str, default=None)
    arg_parser.add_argument("--mdp-stats-num-episodes", type=int, default=None)

    known_args, _ = arg_parser.parse_known_args()
    mutable_args = vars(known_args)
    env_args_json_string = (
        mutable_args.pop("env_args") if mutable_args["env_args"] is not None else "{}"
    )
    try:
        env_args = json.loads(env_args_json_string)
    except json.decoder.JSONDecodeError as err:
        raise RuntimeError(
            f"Failed to parse env-args json string: {env_args_json_string}",
        ) from err

    return progargs.ExperimentArgs.from_flat_dict(
        {**mutable_args, **{"env_args": env_args}}
    )


def create_env_spec_and_mdp(
    problem: str,
    env_args: Mapping[str, Any],
    mdp_stats_path: str,
    mdp_stats_num_episodes: Optional[int],
) -> Tuple[envspec.EnvSpec, markovdp.MDP]:
    """
    Creates a environment spec and MDP for a problem.

    Args:
        problem: problem name.
        args: mapping of experiment parameters.
    """
    env_spec = envsuite.load(name=problem, **env_args)
    mdp = envstats.load_or_generate_inferred_mdp(
        path=mdp_stats_path, env_spec=env_spec, num_episodes=mdp_stats_num_episodes
    )
    return env_spec, mdp


def dynamic_prog_estimation(
    mdp: markovdp.MDP, control_args: progargs.ControlArgs
) -> StateActionValues:
    """
    Runs dynamic programming on an MDP to generate state-value and action-value
    functions.

    Args:
        env_spec: environment specification.
        mdp: Markov Decison Process dynamics
        control_args: control arguments.
    """
    observable_random_policy = policies.PyObservableRandomPolicy(
        num_actions=mdp.env_desc().num_actions,
    )
    state_values = dynamicprog.iterative_policy_evaluation(
        mdp=mdp, policy=observable_random_policy, gamma=control_args.gamma
    )
    logging.debug("State value V(s):\n%s", state_values)
    action_values = dynamicprog.action_values_from_state_values(
        mdp=mdp, state_values=state_values, gamma=control_args.gamma
    )
    logging.debug("Action value Q(s,a):\n%s", action_values)
    return StateActionValues(state_values=state_values, action_values=action_values)


def create_aggregate_reward_step_mapper_fn(
    env_spec: envspec.EnvSpec,
    num_states: int,
    num_actions: int,
    reward_period: int,
    cu_step_method: str,
    buffer_size_or_multiplier: Tuple[Optional[int], Optional[int]],
) -> replay_mapper.TrajMapper:
    """
    Creates an object that alters the trajectory data.

    Args:
        env_spec: environment specification.
        num_states: number of states in the problem.
        num_actions: number of actions in the problem.
        reward_period: the frequency with which rewards are generated.
        cu_step_method: the method to alter trajectory data.
        buffer_size_or_multiplier: number of elements kept in buffer or multiple for |S|x|A|xMultiplier.
    """
    mapper: Optional[replay_mapper.TrajMapper] = None
    if cu_step_method == constants.IDENTITY_MAPPER:
        mapper = replay_mapper.IdentifyMapper()
    elif cu_step_method == constants.REWARD_IMPUTATION_MAPPER:
        mapper = replay_mapper.ImputeMissingRewardMapper(
            reward_period=reward_period, impute_value=0.0
        )
    elif cu_step_method == constants.AVERAGE_REWARD_MAPPER:
        mapper = replay_mapper.AverageRewardMapper(reward_period=reward_period)
    elif cu_step_method == constants.REWARD_ESTIMATION_LSQ_MAPPER:
        _buffer_size, _buffer_size_mult = buffer_size_or_multiplier
        buffer_size = _buffer_size or int(
            num_states
            * num_actions
            * (_buffer_size_mult or constants.DEFAULT_BUFFER_SIZE_MULTIPLIER)
        )
        mapper = replay_mapper.LeastSquaresAttributionMapper(
            num_states=num_states,
            num_actions=num_actions,
            reward_period=reward_period,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            buffer_size=buffer_size,
            init_rtable=utils.initial_table(
                num_states=num_states, num_actions=num_actions
            ),
        )
    else:
        raise ValueError(
            f"Unknown cu-step-method {cu_step_method}. Choices: {constants.CU_MAPPER_METHODS}"
        )

    return mapper


def create_generate_nstep_episodes_fn(
    mapper: replay_mapper.TrajMapper,
) -> Callable[
    [gym.Env, core.PyPolicy, int],
    Generator[core.TrajectoryStep, None, None],
]:
    """
    Creates a function that transform trajectory events a provided
    `mapper`.

    Args:
        mapper: A TrajMapper that transforms trajectory events.
    """

    def generate_nstep_episodes(
        environment: gym.Env,
        policy: core.PyPolicy,
        num_episodes: int,
    ) -> Generator[core.TrajectoryStep, None, None]:
        """
        Generates events for `num_episodes` given an environment and policy.
        """
        # Unroll one trajectory at a time
        for _ in range(num_episodes):
            for traj_step in mapper.apply(
                envplay.generate_episodes(environment, policy, num_episodes=1)
            ):
                yield traj_step

    return generate_nstep_episodes
