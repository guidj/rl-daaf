from typing import Sequence

import numpy as np
import ray
import ray.data

from daaf.policyeval import results_agg_pipeline


def test_pipeline():
    def entry(
        exp_id: str,
        state_values: Sequence[float],
        episode: int = 1,
        algorithm: str = "MC",
    ):
        return {
            "exp_id": exp_id,
            "episode": episode,
            "info": {"state_values": state_values},
            "meta": {"algorithm": algorithm},
        }

    with ray.init(num_cpus=1):
        ds_input = ray.data.from_items(
            [
                # exp 1 - 2 runs
                entry(exp_id="1", state_values=[1, 2]),
                entry(exp_id="1", state_values=[2, 3]),
                # exp 2 - 3 runs
                entry(exp_id="2", state_values=[3, 4]),
                entry(exp_id="2", state_values=[4, 5]),
                entry(exp_id="2", state_values=[5, 6]),
                # exp 4 - 1 run
                entry(exp_id="3", state_values=[6, 7]),
                entry(exp_id="3", state_values=[7, 8], episode=2),
                entry(exp_id="3", state_values=[8, 9], episode=2),
            ]
        )

        expected = [
            {
                "episode": 1,
                "exp_id": "1",
                "meta": {"algorithm": "MC"},
                "state_values": [[1, 2], [2, 3]],
            },
            {
                "episode": 1,
                "exp_id": "2",
                "meta": {"algorithm": "MC"},
                "state_values": [[3, 4], [4, 5], [5, 6]],
            },
            {
                "episode": 1,
                "exp_id": "3",
                "meta": {"algorithm": "MC"},
                "state_values": [[6, 7]],
            },
            {
                "episode": 2,
                "exp_id": "3",
                "meta": {"algorithm": "MC"},
                "state_values": [[7, 8], [8, 9]],
            },
        ]
        output = ray.get(results_agg_pipeline.pipeline.remote(ds_input)).take_all()
        np.testing.assert_equal(output, expected)
