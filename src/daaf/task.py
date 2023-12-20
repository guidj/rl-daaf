"""
Functions relying on ReplayBuffer are for TF classes (agents, environment, etc).
Generators are for Py classes (agents, environment, etc).
"""


from typing import Any, Callable, Generator, Mapping, Optional, Tuple

import gymnasium as gym
from rlplg import core, envplay, envsuite
from rlplg.learning import utils
from rlplg.learning.tabular import policies

from daaf import constants, expconfig, options, replay_mapper


def create_env_spec(
    problem: str,
    env_args: Mapping[str, Any],
) -> core.EnvSpec:
    """
    Creates a environment spec for a problem.

    Args:
        problem: problem name.
        args: mapping of experiment parameters.
    """

    return envsuite.load(name=problem, **env_args)


def create_trajectory_mapper(
    env_spec: core.EnvSpec,
    reward_period: int,
    traj_mapping_method: str,
    buffer_size_or_multiplier: Tuple[Optional[int], Optional[int]],
) -> replay_mapper.TrajMapper:
    """
    Creates an object that alters the trajectory data.

    Args:
        env_spec: environment specification.
        num_states: number of states in the problem.
        num_actions: number of actions in the problem.
        reward_period: the frequency with which rewards are generated.
        traj_mapping_method: the method to alter trajectory data.
        buffer_size_or_multiplier: number of elements kept in buffer or multiple for |S|x|A|xMultiplier.

    Returns:
        A trajectory mapper.
    """

    mapper: Optional[replay_mapper.TrajMapper] = None
    if traj_mapping_method == constants.IDENTITY_MAPPER:
        mapper = replay_mapper.IdentifyMapper()
    elif traj_mapping_method == constants.REWARD_IMPUTATION_MAPPER:
        mapper = replay_mapper.ImputeMissingRewardMapper(
            reward_period=reward_period, impute_value=0.0
        )
    elif traj_mapping_method == constants.AVERAGE_REWARD_MAPPER:
        mapper = replay_mapper.AverageRewardMapper(reward_period=reward_period)
    elif traj_mapping_method == constants.REWARD_ESTIMATION_LSQ_MAPPER:
        _buffer_size, _buffer_size_mult = buffer_size_or_multiplier
        buffer_size = _buffer_size or int(
            env_spec.mdp.env_desc.num_states
            * env_spec.mdp.env_desc.num_actions
            * (_buffer_size_mult or constants.DEFAULT_BUFFER_SIZE_MULTIPLIER)
        )
        mapper = replay_mapper.LeastSquaresAttributionMapper(
            num_states=env_spec.mdp.env_desc.num_states,
            num_actions=env_spec.mdp.env_desc.num_actions,
            reward_period=reward_period,
            state_id_fn=env_spec.discretizer.state,
            action_id_fn=env_spec.discretizer.action,
            buffer_size=buffer_size,
            init_rtable=utils.initial_table(
                num_states=env_spec.mdp.env_desc.num_states,
                num_actions=env_spec.mdp.env_desc.num_actions,
            ),
        )
    elif traj_mapping_method == constants.MDP_WITH_OPTIONS_MAPPER:
        mapper = replay_mapper.MdpWithOptionsMapper()
    else:
        raise ValueError(
            f"Unknown cu-step-method {traj_mapping_method}. Choices: {constants.AGGREGATE_MAPPER_METHODS}"
        )

    return mapper


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


def create_generate_episodes_fn(
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

    def generate_episodes(
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

    return generate_episodes


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    """
    Returns the initial learning rate.
    """
    del episode
    del step
    return initial_lr
