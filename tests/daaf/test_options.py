import hypothesis
from hypothesis import strategies as st

from daaf import options


@hypothesis.given(
    num_actions=st.integers(min_value=2, max_value=5),
    options_duration=st.integers(min_value=2, max_value=5),
)
def test_uniformly_random_composite_actions_options_policy_init(
    num_actions: int, options_duration: int
):
    policy = options.UniformlyRandomCompositeActionPolicy(
        num_actions=num_actions, options_duration=options_duration
    )

    assert policy.options_duration == options_duration
    assert len(getattr(policy, "_options")) == num_actions**options_duration
