import json
import os.path
import tempfile

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


def test_json_from_dict():
    input_ = {
        "context": {"event_id": 1, "time": 5},
        "data": {
            "user_id": 5,
            "preferences": {
                "items": [1, 2, 3],
            },
        },
        "version": 5,
    }

    level_0_ser = {
        "context": json.dumps({"event_id": 1, "time": 5}),
        "data": json.dumps(
            {
                "user_id": 5,
                "preferences": {
                    "items": [1, 2, 3],
                },
            }
        ),
        "version": 5,
    }
    level_1_ser = {
        "context": {"event_id": 1, "time": 5},
        "data": {
            "user_id": 5,
            "preferences": json.dumps(
                {
                    "items": [1, 2, 3],
                }
            ),
        },
        "version": 5,
    }

    assert utils.json_from_dict(input_) == input_
    assert utils.json_from_dict(input_, dict_encode_level=0) == level_0_ser
    assert utils.json_from_dict(input_, dict_encode_level=1) == level_1_ser
