from typing import Sequence, Tuple

import hypothesis
from hypothesis import strategies as st

from daaf import tracking

MIN_REWARD = -100
MAX_REWARD = 100


def test_init():
    stats = tracking.EpisodeStats()

    assert stats.episodic_reward == 0.0
    assert stats.total_reward == 0.0
    assert stats.episode_count == 0
    assert stats.successful_episodes == 0
    assert (
        str(stats)
        == "Episode: 0, Episode(r): 0.0, Total(r): 0.0, Successful Attempts: 0"
    )


@hypothesis.given(st.floats(allow_nan=False, allow_infinity=False))
def test_new_reward_single_step(reward: float):
    stats = tracking.EpisodeStats()
    stats.new_reward(reward)

    assert stats.episodic_reward == reward
    assert stats.total_reward == reward
    assert stats.episode_count == 0
    assert stats.successful_episodes == 0
    # We add 0.0 to the reward in case it's -0.0 or +0.0
    assert (
        str(stats)
        == f"Episode: 0, Episode(r): {reward + 0.0}, Total(r): {reward + 0.0}, Successful Attempts: 0"
    )


@hypothesis.given(
    st.integers(min_value=MIN_REWARD, max_value=MAX_REWARD), st.booleans()
)
def test_end_episode_with_zero_steps_episodes(episodes: int, success: bool):
    stats = tracking.EpisodeStats()
    for episode in range(episodes):
        stats.end_episode(success)
        assert stats.episodic_reward == 0.0
        assert stats.total_reward == 0.0
        assert stats.episode_count == episode + 1
        assert stats.successful_episodes == (episode + 1) * int(success)
        assert (
            str(stats)
            == f"Episode: {episode + 1}, Episode(r): 0.0, Total(r): 0.0, Successful Attempts: {(episode + 1) * int(success)}"
        )


@hypothesis.given(
    st.lists(st.integers(min_value=MIN_REWARD, max_value=MAX_REWARD)), st.booleans()
)
def test_episode_stats_with_multiple_steps_episodes(
    rewards: Sequence[int], success: bool
):
    stats = tracking.EpisodeStats()
    for reward in rewards:
        stats.new_reward(reward)

    assert stats.episodic_reward == sum(rewards)
    assert stats.total_reward == sum(rewards)
    assert stats.episode_count == 0
    assert stats.successful_episodes == 0
    assert (
        str(stats)
        == f"Episode: 0, Episode(r): {float(sum(rewards))}, Total(r): {float(sum(rewards))}, Successful Attempts: 0"
    )

    stats.end_episode(success)

    assert stats.episodic_reward == 0.0
    assert stats.total_reward == sum(rewards)
    assert stats.episode_count == 1
    assert stats.successful_episodes == int(success)
    assert (
        str(stats)
        == f"Episode: 1, Episode(r): 0.0, Total(r): {float(sum(rewards))}, Successful Attempts: {int(success)}"
    )


@hypothesis.given(
    st.lists(
        st.tuples(
            st.lists(st.integers(min_value=MIN_REWARD, max_value=MAX_REWARD)),
            st.booleans(),
        )
    )
)
def test_episode_stats_with_different_variations(
    episodes: Sequence[Tuple[Sequence[int], bool]],
):
    stats = tracking.EpisodeStats()
    expected_success = 0
    expected_total_rewards = 0.0
    for episode, (rewards, success) in enumerate(episodes):
        expected_episode_reward = 0.0
        for reward in rewards:
            stats.new_reward(reward)
            expected_episode_reward += reward
            expected_total_rewards += reward

            assert stats.episodic_reward == expected_episode_reward
            assert stats.total_reward == expected_total_rewards
            assert (
                str(stats)
                == f"Episode: {episode}, Episode(r): {expected_episode_reward}, Total(r): {expected_total_rewards}, Successful Attempts: {expected_success}"
            )

        stats.end_episode(success)
        expected_success += int(success)

        assert stats.episodic_reward == 0.0
        assert stats.total_reward == expected_total_rewards
        assert stats.episode_count == episode + 1
        assert stats.successful_episodes == expected_success
        assert (
            str(stats)
            == f"Episode: {episode + 1}, Episode(r): 0.0, Total(r): {expected_total_rewards}, Successful Attempts: {expected_success}"
        )
