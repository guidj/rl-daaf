import collections
import copy
import dataclasses
from typing import Callable, DefaultDict, Dict, Generator, List, Tuple

import gymnasium as gym
import numpy as np

from daaf import core, envplay
from daaf.core import MapsToIntId
from daaf.learning.opt import schedules

MCUpdate = collections.namedtuple("MCUpdate", ["returns", "cu_sum", "value", "weight"])


@dataclasses.dataclass(frozen=True)
class PolicyEvalSnapshot:
    steps: int
    values: np.ndarray


def onpolicy_first_visit_monte_carlo_action_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    gamma: float,
    state_id_fn: MapsToIntId,
    action_id_fn: MapsToIntId,
    initial_qtable: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """
    First-Visit Monte Carlo Prediction.
    Estimates Q(s, a) for a fixed policy pi.
    Source: http://www.incompleteideas.net/book/ebook/node52.html

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """

    def visit_key(experience) -> Tuple[int, int]:
        return state_id_fn(experience.observation), action_id_fn(experience.action)

    qtable = copy.deepcopy(initial_qtable)
    state_action_updates: DefaultDict[Tuple[int, int], int] = collections.defaultdict(
        int
    )
    state_action_visits_remaining: DefaultDict[Tuple[int, int], int] = (
        collections.defaultdict(int)
    )

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences_ = list(generate_episode(environment, policy))
        num_steps = len(experiences_)
        # reverse list and ammortize state visits
        experiences: List[core.TrajectoryStep] = []
        while len(experiences_) > 0:
            experience = experiences_.pop()
            state_action_visits_remaining[visit_key(experience)] += 1
            experiences.append(experience)

        episode_return = 0.0
        while True:
            try:
                experience = experiences.pop(0)
            except IndexError:
                break
            key = visit_key(experience)
            state_action_visits_remaining[key] -= 1
            reward = experience.reward
            episode_return = gamma * episode_return + reward

            if state_action_visits_remaining[key] == 0:
                state_action_updates[key] += 1
                state_id, action_id = key
                if experience.terminated:
                    qtable[state_id, action_id] = 0.0
                else:
                    if state_action_updates[key] == 1:
                        # first value
                        qtable[state_id, action_id] = episode_return
                    else:
                        # average returns
                        qtable[state_id, action_id] = qtable[state_id, action_id] + (
                            (episode_return - qtable[state_id, action_id])
                            / state_action_updates[key]
                        )

        # need to copy values because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=num_steps, values=copy.deepcopy(qtable))


def onpolicy_sarsa_action_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    state_id_fn: MapsToIntId,
    action_id_fn: MapsToIntId,
    initial_qtable: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """
    On-policy Sarsa Prediction.
    Estimates Q(s, a) for a fixed policy pi.
    Source: https://homes.cs.washington.edu/~bboots/RL-Spring2020/Lectures/TD_notes.pdf
    In the document, they refer to Algorithm 15 as Algorithm 16.

    Note to self: As long you don't use the table you're updating,
    the current approach is fine

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    qtable = copy.deepcopy(initial_qtable)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        while True:
            try:
                traj_step = next(trajectory)
            except StopIteration:
                break
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - 2, None)

            state_id = state_id_fn(experiences[step].observation)
            action_id = action_id_fn(experiences[step].action)
            reward = experiences[step].reward

            next_state_id = state_id_fn(experiences[step + 1].observation)
            next_action_id = action_id_fn(experiences[step + 1].action)
            alpha = lrs(episode=episode, step=steps_counter)

            if experiences[step].terminated:
                qtable[state_id, action_id] = 0.0
            else:
                qtable[state_id, action_id] += alpha * (
                    reward
                    + gamma * qtable[next_state_id, next_action_id]
                    - qtable[state_id, action_id]
                )
            steps_counter += 1
            step += 1

        # need to copy qtable because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=traj_step_idx, values=copy.deepcopy(qtable))


def onpolicy_first_visit_monte_carlo_state_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    gamma: float,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """
    First-Visit Monte Carlo Prediction.
    Estimates V(s) for a fixed policy pi.
    Source: http://www.incompleteideas.net/book/ebook/node51.html

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the V(s) table.
        initial_values: Initial state-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    values = copy.deepcopy(initial_values)
    state_updates: DefaultDict[int, int] = collections.defaultdict(int)
    state_visits: DefaultDict[int, int] = collections.defaultdict(int)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences_ = list(generate_episode(environment, policy))
        num_steps = len(experiences_)
        # reverse list and ammortize state visits
        experiences: List[core.TrajectoryStep] = []
        while len(experiences_) > 0:
            experience = experiences_.pop()
            state_visits[state_id_fn(experience.observation)] += 1
            experiences.append(experience)

        episode_return = 0.0
        while True:
            try:
                experience = experiences.pop(0)
            except IndexError:
                break
            state_id = state_id_fn(experience.observation)
            reward = experience.reward
            episode_return = gamma * episode_return + reward
            state_visits[state_id] -= 1

            # update non-terminal states
            if state_visits[state_id] == 0:
                if experience.terminated:
                    values[state_id] = 0.0
                else:
                    if state_updates[state_id] == 0:
                        # first value
                        values[state_id] = episode_return
                    else:
                        values[state_id] = values[state_id] + (
                            (episode_return - values[state_id])
                            / state_updates[state_id]
                        )
                state_updates[state_id] += 1

        # need to copy values because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=num_steps, values=copy.deepcopy(values))


def onpolicy_one_step_td_state_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """
    TD(0) or one-step TD.
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
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    values = copy.deepcopy(initial_values)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        while True:
            try:
                traj_step = next(trajectory)
            except StopIteration:
                break
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - 2, None)

            state_id = state_id_fn(experiences[step].observation)
            next_state_id = state_id_fn(experiences[step + 1].observation)
            alpha = lrs(episode=episode, step=steps_counter)

            if experiences[step].terminated:
                values[state_id] = 0.0
            else:
                values[state_id] += alpha * (
                    experiences[step].reward
                    + gamma * values[next_state_id]
                    - values[state_id]
                )
            steps_counter += 1
            step += 1

        # need to copy values because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=traj_step_idx, values=copy.deepcopy(values))


def onpolicy_nstep_td_state_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    nstep: int,
    state_id_fn: MapsToIntId,
    initial_values: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
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
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and v-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract reward access references by one.
    """

    values = copy.deepcopy(initial_values)
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        # In the absence of a step as terminal
        # or truncated, `empty_steps` prevents
        # infinite loops
        empty_steps = 0
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

            # TD requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - nstep, None)

            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
                    final_step = step + 1
            tau = step - nstep + 1
            if tau >= 0:
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
                if experiences[tau].terminated:
                    values[state_id] = 0.0
                else:
                    values[state_id] += alpha * (returns - values[state_id])
                steps_counter += 1
            step += 1
        # need to copy qtable because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=traj_step_idx, values=copy.deepcopy(values))


def offpolicy_monte_carlo_action_values(
    policy: core.PyPolicy,
    collect_policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    gamma: float,
    policy_probability_fn: Callable[
        [core.PyPolicy, core.TrajectoryStep],
        float,
    ],
    collect_policy_probability_fn: Callable[
        [core.PyPolicy, core.TrajectoryStep],
        float,
    ],
    state_id_fn: MapsToIntId,
    action_id_fn: MapsToIntId,
    initial_qtable: np.ndarray,
    generate_episode: core.GeneratesEpisode = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """Off-policy MC Prediction.
    Estimates Q (table) for a fixed policy pi.

    Questions: when do you update the propensity of each action?
    At each step, do you use the propensity at that step the one
    reached at the end of the episode?

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        collect_policy: A behavior policy, used to generate episodes.
        environment: The environment used to generate episodes for evaluation.
        event_buffer: A buffer to store episode steps.
        num_episodes: The number of episodes to generate for evaluation.
        gamma: The discount rate.
        policy_probability_fn: a mapper of the probability
            of a given action in a state for the target
            policy.
        collect_policy_probability_fn: a mapper of the probability
            of a given action in a state for the collection
            policy.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.
    """
    for name, _policy in zip(("policy", "collect_policy"), (policy, collect_policy)):
        if not _policy.emit_log_probability:
            raise ValueError(
                f"{name} must have emit_log_probability=True. They are used for importance sampling."
            )

    qtable = copy.deepcopy(initial_qtable)
    cu_sum = np.zeros_like(initial_qtable)

    for _ in range(num_episodes):
        experiences = list(generate_episode(environment, collect_policy))
        num_steps = len(experiences)
        returns, weight = 0.0, 1.0
        # process from the ending
        while weight != 0.0:
            try:
                experience = experiences.pop()
            except IndexError:
                break
            state_id, action_id, reward = (
                state_id_fn(experience.observation),
                action_id_fn(experience.action),
                experience.reward,
            )
            policy_prob = policy_probability_fn(policy, experience)
            collect_policy_prob = collect_policy_probability_fn(
                collect_policy, experience
            )
            rho = policy_prob / collect_policy_prob

            (
                returns,
                cu_sum[state_id, action_id],
                state_value,
                weight,
            ) = offpolicy_monte_carlo_action_values_step(
                reward=reward,
                returns=returns,
                cu_sum=cu_sum[state_id, action_id],
                weight=weight,
                value=qtable[state_id, action_id],
                rho=rho,
                gamma=gamma,
            )
            if experience.terminated:
                qtable[state_id, action_id] = 0.0
            else:
                qtable[state_id, action_id] = state_value
        # need to copy qtable because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=num_steps, values=copy.deepcopy(qtable))


def offpolicy_monte_carlo_action_values_step(
    reward: float,
    returns: float,
    cu_sum: float,
    weight: float,
    value: float,
    rho: float,
    gamma: float,
) -> MCUpdate:
    """Single step update for Off-policy MC prediction for estimating Q_{pi}.
    G <- gamma * G + R_{t+1}
    C(S_{t},A_{t}) <- C(S_{t},A_{t}) + W
    Q(S_{t},A_{t}) <- Q(S_{t},A_{t}) + W/C(S_{t},A_{t}) * (G - Q(S_{t},A_{t}))
    rho <- pi(S_{t}|A_{t}) / b(S_{t}|A_{t})
    W <- W * rho
    """
    new_returns = gamma * returns + reward
    new_cu_sum = cu_sum + weight
    new_value = value + (weight / new_cu_sum) * (new_returns - value)
    new_weight = weight * rho
    return MCUpdate(
        returns=new_returns, cu_sum=new_cu_sum, value=new_value, weight=new_weight
    )


def offpolicy_nstep_sarsa_action_values(
    policy: core.PyPolicy,
    collect_policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    nstep: int,
    policy_probability_fn: Callable[
        [core.PyPolicy, core.TrajectoryStep],
        float,
    ],
    collect_policy_probability_fn: Callable[
        [core.PyPolicy, core.TrajectoryStep],
        float,
    ],
    state_id_fn: MapsToIntId,
    action_id_fn: MapsToIntId,
    initial_qtable: np.ndarray,
    generate_episode: Callable[
        [
            gym.Env,
            core.PyPolicy,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episode,
) -> Generator[PolicyEvalSnapshot, None, None]:
    """
    Off-policy n-step Sarsa Prediction.
    Estimates Q (table) for a fixed policy pi.

    The algorithm assumes the starting states is a non-terminal
    state.

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        collect_policy: A behavior policy, used to generate episodes.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        nstep: The number of steps used in TD updates.
        policy_probability_fn: a mapper of the probability
            of a given action in a state for the target
            policy.
        collect_policy_probability_fn: a mapper of the probability
            of a given action in a state for the collection
            policy.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episode: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract reward access references by one.
    """
    if nstep < 1:
        raise ValueError(f"nstep must be > 1: {nstep}")
    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, collect_policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        # In the absence of a step as terminal
        # or truncated, `empty_steps` prevents
        # infinite loops
        empty_steps = 0
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

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - nstep, None)

            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
                    final_step = step + 1
            tau = step - nstep + 1
            if tau >= 0:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step)
                rho = 1.0
                returns = 0.0
                for i in range(min_idx, max_idx + 1):
                    if i < max_idx:
                        rho *= policy_probability_fn(
                            policy, experiences[i]
                        ) / collect_policy_probability_fn(
                            collect_policy, experiences[i]
                        )
                    returns += (gamma ** (i - tau - 1)) * experiences[i - 1].reward
                if tau + nstep < final_step:
                    returns += (gamma**nstep) * qtable[
                        state_id_fn(experiences[tau + nstep].observation),
                        action_id_fn(experiences[tau + nstep].action),
                    ]

                state_id = state_id_fn(experiences[tau].observation)
                action_id = action_id_fn(experiences[tau].action)
                alpha = lrs(episode=episode, step=steps_counter)

                if experiences[tau].terminated:
                    qtable[state_id, action_id] = 0.0
                else:
                    qtable[state_id, action_id] += (
                        alpha * rho * (returns - qtable[state_id, action_id])
                    )
                steps_counter += 1
            if tau == final_step - 1:
                break
            step += 1

        # need to copy qtable because it's a mutable numpy array
        yield PolicyEvalSnapshot(steps=traj_step_idx, values=copy.deepcopy(qtable))
