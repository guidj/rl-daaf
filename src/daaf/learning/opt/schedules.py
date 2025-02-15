"""
This module contains schedules for learning algorithms.
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class LearningRateSchedule:
    """
    This class updates the learning rate based on the episode,
    step or both - using a given `schedule` function.
    """

    initial_learning_rate: float
    # fn(initial learning rate, episode, step) -> learning rate
    schedule: Callable[[float, int, int], float]
    verbose: bool = False

    def __call__(self, episode: int, step: int):
        return self.schedule(self.initial_learning_rate, episode, step)
