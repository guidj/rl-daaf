import hypothesis
from hypothesis import strategies as st
from rlplg import core

from daaf import options


@hypothesis.given(
    num_actions=st.integers(min_value=2, max_value=5),
    options_duration=st.integers(min_value=2, max_value=5),
)
def test_uniformly_random_composite_actions_options_policy_init(
    num_actions: int, options_duration: int
):
    policy = options.UniformlyRandomCompositeActionPolicy(
        actions=tuple(range(num_actions)), options_duration=options_duration
    )

    assert policy.actions == tuple(range(num_actions))
    assert policy.options_duration == options_duration
    assert getattr(policy, "_num_options") == num_actions**options_duration
    assert policy.get_initial_state() == {"option_id": None, "option_step": -1}


def test_uniformly_random_composite_actions_options_policy_action():
    # Four possible options: (a, a), (a, b), (b, a), (b, b)
    policy = options.UniformlyRandomCompositeActionPolicy(
        actions=(
            "A",
            "B",
        ),
        options_duration=2,
    )

    output = policy.action(
        observation=(), policy_state={"option_id": None, "option_step": -1}
    )
    assert output.action in ("A", "B")
    assert output.state["option_id"] in range(4)
    assert output.state["option_step"] == 0
    assert output.info["option_id"] in range(4)
    assert output.info["option_terminated"] is False

    # choose last step in option next
    assert policy.action(
        observation=(), policy_state={"option_id": 2, "option_step": 0}
    ) == core.PolicyStep(
        action="A",
        state={"option_id": 2, "option_step": 1},
        info={"option_id": 2, "option_terminated": True},
    )

    # last step in option has been chosen
    # choose a new option
    output = policy.action(
        observation=(), policy_state={"option_id": 2, "option_step": 1}
    )
    assert output.action in ("A", "B")
    assert output.state["option_id"] in range(4)
    assert output.state["option_step"] == 0
    assert output.info["option_id"] in range(4)
    assert output.info["option_terminated"] is False


@hypothesis.given(
    num_actions=st.integers(min_value=2, max_value=5),
    options_duration=st.integers(min_value=2, max_value=5),
    num_trials=st.integers(min_value=2, max_value=5),
)
def test_uniformly_random_composite_actions_options_policy_state_action_prob(
    num_actions: int, options_duration: int, num_trials: int
):
    policy = options.UniformlyRandomCompositeActionPolicy(
        actions=tuple(range(num_actions)), options_duration=options_duration
    )
    for _ in range(num_trials):
        assert policy.state_action_prob((), ()) == 1.0 / (num_actions**options_duration)
