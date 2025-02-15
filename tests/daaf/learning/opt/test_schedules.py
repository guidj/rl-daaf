import hypothesis
from hypothesis import strategies as st

from daaf.learning.opt import schedules


@hypothesis.given(st.floats(allow_nan=False, allow_infinity=False))
def test_learning_rate_schedule_init(initial_learning_rate: float):
    def schedule(ilr: float, episode: int, step: int):
        del episode
        del step
        return ilr

    lrs = schedules.LearningRateSchedule(
        initial_learning_rate=initial_learning_rate, schedule=schedule
    )

    assert lrs.initial_learning_rate == initial_learning_rate


@hypothesis.given(step=st.integers())
def test_learning_rate_schedule_call_with_episode_schedule(step: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del step
        return initial_learning_rate * (0.9**episode)

    assert (
        schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=1, step=step
        )
        == 0.9
    )
    # increase episode
    assert (
        schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=2, step=step
        )
        == 0.81
    )
    # change initial learning rate
    assert (
        schedules.LearningRateSchedule(initial_learning_rate=100.0, schedule=schedule)(
            episode=2, step=step
        )
        == 81
    )


@hypothesis.given(episode=st.integers())
def test_learning_rate_schedule_call_with_step_schedule(episode: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del episode
        return initial_learning_rate * (0.9**step)

    assert (
        schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=episode, step=1
        )
        == 0.9
    )
    # increase step
    assert (
        schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)(
            episode=episode, step=2
        )
        == 0.81
    )
    # change initial learning rate
    assert (
        schedules.LearningRateSchedule(initial_learning_rate=100.0, schedule=schedule)(
            episode=episode, step=2
        )
        == 81
    )


@hypothesis.given(step=st.integers())
def test_learning_rate_schedule_call_with_decaying_schedule(step: int):
    def schedule(initial_learning_rate: float, episode: int, step: int):
        del step
        if episode < 10:
            return initial_learning_rate
        return initial_learning_rate * 0.9

    lrs = schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)

    assert lrs(episode=1, step=step) == 1
    assert lrs(episode=9, step=step) == 1
    assert lrs(episode=10, step=step) == 0.9
    assert lrs(episode=21, step=step) == 0.9
