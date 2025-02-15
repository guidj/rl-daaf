"""
Functions relying on ReplayBuffer are for TF classes (agents, environment, etc).
Generators are for Py classes (agents, environment, etc).
"""

from typing import Any, Generator, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
from daaf import core, envplay, envsuite
from daaf.learning import utils

from daaf import constants, replay_mapper


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


def create_trajectory_mappers(
    env_spec: core.EnvSpec,
    reward_period: int,
    traj_mapping_method: str,
    buffer_size_or_multiplier: Tuple[Optional[int], Optional[int]],
    drop_truncated_feedback_episodes: bool,
) -> Sequence[replay_mapper.TrajMapper]:
    """
    Creates an object that alters the trajectory data.

    Args:
        env_spec: environment specification.
        num_states: number of states in the problem.
        num_actions: number of actions in the problem.
        reward_period: the frequency with which rewards are generated.
        traj_mapping_method: the method to alter trajectory data.
        buffer_size_or_multiplier: number of elements kept in buffer
            or multiple for |S|x|A|xMultiplier.

    Returns:
        A trajectory mapper.
    """

    mappers: List[replay_mapper.TrajMapper] = []
    if drop_truncated_feedback_episodes:
        mappers.append(
            replay_mapper.DaafDropEpisodeWithTruncatedFeedbackMapper(
                reward_period=reward_period
            )
        )
    if traj_mapping_method == constants.IDENTITY_MAPPER:
        mappers.append(replay_mapper.IdentityMapper())
    elif traj_mapping_method == constants.DAAF_TRAJECTORY_MAPPER:
        mappers.append(replay_mapper.DaafTrajectoryMapper(reward_period=reward_period))
    elif traj_mapping_method == constants.DAAF_IMPUTE_REWARD_MAPPER:
        mappers.append(
            replay_mapper.DaafImputeMissingRewardMapper(
                reward_period=reward_period, impute_value=0.0
            )
        )
    elif traj_mapping_method == constants.DAAF_LSQ_REWARD_ATTRIBUTION_MAPPER:
        _buffer_size, _buffer_size_mult = buffer_size_or_multiplier
        buffer_size = _buffer_size or int(
            env_spec.mdp.env_desc.num_states
            * env_spec.mdp.env_desc.num_actions
            * (_buffer_size_mult or constants.DEFAULT_BUFFER_SIZE_MULTIPLIER)
        )
        mappers.append(
            replay_mapper.DaafLsqRewardAttributionMapper(
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
                terminal_states=core.infer_env_terminal_states(env_spec.mdp.transition),
            )
        )
    elif traj_mapping_method == constants.MDP_WITH_OPTIONS_MAPPER:
        mappers.append(replay_mapper.DaafMdpWithOptionsMapper())
    elif traj_mapping_method == constants.DAAF_NSTEP_TD_UPDATE_MARK_MAPPER:
        mappers.append(
            replay_mapper.DaafNStepTdUpdateMarkMapper(reward_period=reward_period)
        )
    else:
        raise ValueError(
            f"""Unknown cu-step-method {traj_mapping_method}.
            Choices: {constants.AGGREGATE_MAPPER_METHODS}"""
        )
    return mappers


def returns_collection_mapper() -> replay_mapper.CollectReturnsMapper:
    """
    Returns:
        A returns collection mapper.
    """
    return replay_mapper.CollectReturnsMapper()


def create_generate_episode_fn(
    mappers: Sequence[replay_mapper.TrajMapper],
) -> core.GeneratesEpisode:
    """
    Creates a function that transform trajectory events a provided
    `mapper`.

    Args:
        mapper: A TrajMapper that transforms trajectory events.
    """

    def generate_episode(
        environment: gym.Env,
        policy: core.PyPolicy,
        max_steps: Optional[int] = None,
    ) -> Generator[core.TrajectoryStep, None, None]:
        """
        Generates events for `num_episodes` given an environment and policy.
        """
        trajectory = envplay.generate_episode(environment, policy, max_steps=max_steps)
        for mapper in mappers:
            trajectory = mapper.apply(trajectory)
        yield from trajectory

    return generate_episode


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    """
    Returns the initial learning rate.
    """
    del episode
    del step
    return initial_lr
