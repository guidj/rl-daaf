"""
Module for classes and functions to log stats and episodic information,
to std output or files.
"""


class EpisodeStats:
    """
    Tracks episode states - success, attempts, rewards.
    """

    def __init__(self):
        self._episode_count: int = 0
        self._episode_reward: float = 0.0
        self._total_reward: float = 0.0
        self._successful_episodes: int = 0

    def new_reward(self, reward: float) -> None:
        """
        Increments the reward.
        """
        self._episode_reward += reward
        self._total_reward += reward

    def end_episode(self, success: bool) -> None:
        """
        Resets the reward accumulator, and increments the episode.
        If success is True, it increments the success count.
        """
        self._episode_count += 1
        self._successful_episodes += int(success)
        self._episode_reward = 0.0

    @property
    def episodic_reward(self) -> float:
        """
        Returns cumulative reward for the current episode.
        """
        return self._episode_reward

    @property
    def total_reward(self) -> float:
        """
        Returns the total reward across episodes.
        """
        return self._total_reward

    @property
    def episode_count(self) -> int:
        """
        Returns the episode count.
        """
        return self._episode_count

    @property
    def successful_episodes(self) -> int:
        """
        Returns the successful episode count.
        """
        return self._successful_episodes

    def __str__(self) -> str:
        """
        Class reprensted as a logging message.
        """
        return f"Episode: {self.episode_count}, Episode(r): {self.episodic_reward}, Total(r): {self.total_reward}, Successful Attempts: {self.successful_episodes}"
