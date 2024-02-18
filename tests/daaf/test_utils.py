import json
import os.path
import random
import tempfile

import hypothesis
import hypothesis.strategies as st
import pytest

from daaf import utils


def test_experiment_logger():
    with tempfile.TemporaryDirectory() as tempdir:
        params = {"param-1": 1, "param-2": "z"}
        with utils.ExperimentLogger(
            log_dir=tempdir, exp_id="EXP-1", run_id=3, params=params
        ) as experiment_logger:
            experiment_logger.log(episode=1, steps=10, returns=100)
            experiment_logger.log(episode=2, steps=4, returns=40, info={"error": 0.123})

        with open(
            os.path.join(tempdir, utils.ExperimentLogger.PARAM_FILE_NAME),
            "r",
            encoding="UTF-8",
        ) as readable:
            output_params = json.load(readable)
            expected_params = {**params, "exp_id": "EXP-1", "run_id": 3}
            assert output_params == expected_params

        with open(
            os.path.join(tempdir, utils.ExperimentLogger.LOG_FILE_NAME),
            "r",
            encoding="UTF-8",
        ) as readable:
            expected_output = [
                {
                    "episode": 1,
                    "steps": 10,
                    "returns": 100,
                    "info": {},
                },
                {"episode": 2, "steps": 4, "returns": 40, "info": {"error": 0.123}},
            ]
            output_logs = [json.loads(line) for line in readable]
            assert output_logs == expected_output


def test_experiment_logger_with_logging_uninitialized():
    with tempfile.TemporaryDirectory() as tempdir:
        params = {"param-1": 1, "param-2": "z"}
        experiment_logger = utils.ExperimentLogger(
            log_dir=tempdir, exp_id="EXP-1", run_id=1, params=params
        )

        with pytest.raises(RuntimeError):
            experiment_logger.log(episode=1, steps=10, returns=100)


def test_experiment_logger_with_nonexisitng_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        params = {"param-1": 1, "param-2": "z"}
        log_dir = os.path.join(tempdir, "subdir")
        with utils.ExperimentLogger(
            log_dir=log_dir, exp_id="EXP-1", run_id=2, params=params
        ) as experiment_logger:
            experiment_logger.log(episode=1, steps=10, returns=100)
            experiment_logger.log(episode=2, steps=4, returns=40, info={"error": 0.123})

        with open(
            os.path.join(log_dir, utils.ExperimentLogger.PARAM_FILE_NAME),
            "r",
            encoding="UTF-8",
        ) as readable:
            output_params = json.load(readable)
            expected_params = {**params, "exp_id": "EXP-1", "run_id": 2}
            assert output_params == expected_params

        with open(
            os.path.join(log_dir, utils.ExperimentLogger.LOG_FILE_NAME),
            "r",
            encoding="UTF-8",
        ) as readable:
            expected_output = [
                {
                    "episode": 1,
                    "steps": 10,
                    "returns": 100,
                    "info": {},
                },
                {"episode": 2, "steps": 4, "returns": 40, "info": {"error": 0.123}},
            ]
            output_logs = [json.loads(line) for line in readable]
            assert output_logs == expected_output


def test_partition():
    assert utils.partition([1, 2, 3], num_partitions=1) == [[1, 2, 3]]
    assert utils.partition([1, 2, 3], num_partitions=2) == [[1, 2], [3]]
    assert utils.partition([1, 2, 3], num_partitions=3) == [[1], [2], [3]]
    assert utils.partition([1, 2, 3], num_partitions=4) == [[1], [2], [3]]


def test_bundle():
    assert utils.bundle([1, 2, 3], bundle_size=1) == [[1], [2], [3]]
    assert utils.bundle([1, 2, 3], bundle_size=2) == [[1, 2], [3]]
    assert utils.bundle([1, 2, 3], bundle_size=3) == [[1, 2, 3]]
    assert utils.bundle([1, 2, 3], bundle_size=4) == [[1, 2, 3]]


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=10),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_interger_to_sequence(space_size: int, sequence_length: int):
    sample_index = random.randint(0, space_size**sequence_length - 1)
    assert 0 <= sample_index < space_size**sequence_length
    seq = utils.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=sample_index
    )
    assert len(seq) == sequence_length
    assert all([element in set(range(space_size)) for element in seq])


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=100),
    samples=st.integers(min_value=1, max_value=100),
)
@hypothesis.settings(deadline=None)
def test_sequence_to_integer(space_size: int, sequence_length: int, samples: int):
    for _ in range(samples):
        sequence = [random.randint(0, space_size - 1) for _ in range(sequence_length)]
        # for seq in sequences:
        index = utils.sequence_to_integer(space_size, sequence=sequence)
        # assert isinstance(index, int)
        assert 0 <= index < space_size**sequence_length

    # largest sequence
    sequence = [space_size - 1] * sequence_length
    index = utils.sequence_to_integer(space_size, sequence=sequence)
    assert 0 <= index < (space_size**sequence_length)


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_interger_to_sequence_round_trip(space_size: int, sequence_length: int):
    index = random.randint(0, space_size**sequence_length - 1)
    seq = utils.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=index
    )
    output = utils.sequence_to_integer(space_size=space_size, sequence=seq)
    assert output == index
    assert len(seq) == sequence_length
    assert all([element in set(range(space_size)) for element in seq])


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_sequence_to_integer_round_trip(space_size: int, sequence_length: int):
    sequence = tuple(
        [random.randint(0, space_size - 1) for _ in range(sequence_length)]
    )
    output_integer = utils.sequence_to_integer(space_size=space_size, sequence=sequence)
    output_sequence = utils.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=output_integer
    )
    assert sequence == output_sequence
