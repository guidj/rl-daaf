import json
import os.path
import tempfile

import pytest

from daaf import utils


def test_experiment_logger():
    with tempfile.TemporaryDirectory() as tempdir:
        name = "EXP-1"
        params = {"param-1": 1, "param-2": "z"}
        with utils.ExperimentLogger(
            log_dir=tempdir, name=name, params=params
        ) as experiment_logger:
            experiment_logger.log(episode=1, steps=10, returns=100)
            experiment_logger.log(episode=2, steps=4, returns=40, info={"error": 0.123})

        with open(
            os.path.join(tempdir, utils.ExperimentLogger.PARAM_FILE_NAME)
        ) as readable:
            output_params = json.load(readable)
            expected_params = {**params, "name": name}
            assert output_params == expected_params

        with open(
            os.path.join(tempdir, utils.ExperimentLogger.LOG_FILE_NAME)
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
        name = "EXP-1"
        params = {"param-1": 1, "param-2": "z"}
        experiment_logger = utils.ExperimentLogger(
            log_dir=tempdir, name=name, params=params
        )

        with pytest.raises(RuntimeError):
            experiment_logger.log(episode=1, steps=10, returns=100)


def test_experiment_logger_with_nonexisitng_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        name = "EXP-1"
        params = {"param-1": 1, "param-2": "z"}
        log_dir = os.path.join(tempdir, "subdir")
        with utils.ExperimentLogger(
            log_dir=log_dir, name=name, params=params
        ) as experiment_logger:
            experiment_logger.log(episode=1, steps=10, returns=100)
            experiment_logger.log(episode=2, steps=4, returns=40, info={"error": 0.123})

        with open(
            os.path.join(log_dir, utils.ExperimentLogger.PARAM_FILE_NAME)
        ) as readable:
            output_params = json.load(readable)
            expected_params = {**params, "name": name}
            assert output_params == expected_params

        with open(
            os.path.join(log_dir, utils.ExperimentLogger.LOG_FILE_NAME)
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
