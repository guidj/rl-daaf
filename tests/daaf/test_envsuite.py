import math
import uuid
from typing import Any, Mapping, Sequence
from typing import SupportsFloat
import numpy as np
import pytest

from daaf import core
from daaf.core import EnvTransition


from daaf import envsuite
from daaf.core import TimeStep


@pytest.mark.parametrize(
    "env_name", envsuite.SUPPORTED_LIB_ENVS | envsuite.SUPPORTED_GYM_ENVS
)
def test_envsuite_load(env_name: str, args: Mapping[str, Sequence[Mapping[str, Any]]]):
    for kwargs in args[env_name]:
        env_spec = envsuite.load(name=env_name, **kwargs)
        assert isinstance(env_spec, core.EnvSpec)
        assert env_spec.name == env_name
        assert_transition_mapping(
            env_spec.mdp.transition, env_desc=env_spec.mdp.env_desc
        )
        terminal_states = core.infer_env_terminal_states(env_spec.mdp.transition)
        # reset env and state (get initial values)
        # play for one episode
        obs, _ = env_spec.environment.reset()
        time_step: TimeStep = obs, math.nan, False, False, {}
        assert 0 <= env_spec.discretizer.state(obs) <= env_spec.mdp.env_desc.num_states

        while True:
            obs, _, terminated, truncated, _ = time_step
            action = np.random.default_rng().integers(
                0, env_spec.mdp.env_desc.num_actions
            )
            next_time_step = env_spec.environment.step(action)
            next_obs, next_reward, _, _, _ = next_time_step
            assert (
                0 <= env_spec.discretizer.state(obs) <= env_spec.mdp.env_desc.num_states
            )
            assert (
                0
                <= env_spec.discretizer.action(action)
                <= env_spec.mdp.env_desc.num_actions
            )
            if env_spec.discretizer.state(obs) in terminal_states:
                assert next_reward == 0.0
                assert env_spec.discretizer.state(obs) == env_spec.discretizer.state(
                    next_obs
                )
            if terminated or truncated:
                break
            time_step = next_time_step
            env_spec.environment.close()


def test_envsuite_load_with_unsupported_env():
    with pytest.raises(ValueError):
        envsuite.load(str(uuid.uuid4()))


@pytest.fixture
def args() -> Mapping[str, Sequence[Mapping[str, Any]]]:
    return {
        "ABCSeq": [{"length": 3, "distance_penalty": False}],
        "CliffWalking-v0": [{"max_episode_steps": 100}],
        "FrozenLake-v1": [{"is_slippery": False}],
        "GridWorld": [{"grid": "xooo\nsoxg"}],
        "RedGreenSeq": [{"cure": ["red", "green", "wait", "green"]}],
        "StateRandomWalk": [{"steps": 3}],
        "IceWorld": [{"map_name": "4x4"}, {"map": "FFFG\nSFHH"}],
        "TowerOfHanoi": [{"num_disks": 4}],
    }


def assert_transition_mapping(
    transition_mapping: EnvTransition, env_desc: core.EnvDesc
):
    assert len(transition_mapping) == env_desc.num_states
    for state, action_transitions in transition_mapping.items():
        assert 0 <= state < env_desc.num_states
        assert len(action_transitions) == env_desc.num_actions
        for action, transitions in action_transitions.items():
            assert 0 <= action < env_desc.num_actions
            for prob, next_state, reward, done in transitions:
                assert 0 <= prob <= 1.0
                assert 0 <= next_state < env_desc.num_states
                assert isinstance(reward, SupportsFloat)
                assert isinstance(done, bool)
