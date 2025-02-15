import collections

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st
from daaf.learning.tabular import tools


@hypothesis.given(
    num_values=st.integers(min_value=4, max_value=10),
    max_permutation_size=st.integers(min_value=1, max_value=4),
)
def test_nsize_permutations(num_values: int, max_permutation_size: int):
    expected_size = sum([num_values**exp for exp in range(1, max_permutation_size + 1)])
    expected_elems = [-1] * int(max_permutation_size > 1) + list(range(num_values))
    output = tools.nsize_permutations(
        num_values=num_values, max_permutation_size=max_permutation_size
    )
    assert len(output) == expected_size
    assert sorted(np.unique(output).tolist()) == expected_elems


def test_nsize_permutations_simple_cases():
    inputs = [1, 2, 3]
    expectations = [
        collections.Counter([(0,)]),
        collections.Counter([(0, 7), (0, 0), (0, 1), (1, 7), (1, 0), (1, 1)]),
        collections.Counter(
            [
                (0, 7, 7),
                (0, 0, 7),
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (0, 1, 7),
                (0, 1, 0),
                (0, 1, 1),
                (0, 1, 2),
                (0, 2, 7),
                (0, 2, 0),
                (0, 2, 1),
                (0, 2, 2),
                (1, 7, 7),
                (1, 0, 7),
                (1, 0, 0),
                (1, 0, 1),
                (1, 0, 2),
                (1, 1, 7),
                (1, 1, 0),
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 7),
                (1, 2, 0),
                (1, 2, 1),
                (1, 2, 2),
                (2, 7, 7),
                (2, 0, 7),
                (2, 0, 0),
                (2, 0, 1),
                (2, 0, 2),
                (2, 1, 7),
                (2, 1, 0),
                (2, 1, 1),
                (2, 1, 2),
                (2, 2, 7),
                (2, 2, 0),
                (2, 2, 1),
                (2, 2, 2),
            ]
        ),
    ]

    outputs = list(
        map(
            lambda num_values: collections.Counter(
                tools.nsize_permutations(
                    num_values=num_values, max_permutation_size=num_values, stub=7
                )
            ),
            inputs,
        )
    )

    for output, expected in zip(outputs, expectations):
        assert output == expected


@hypothesis.given(
    num_values=st.integers(),
    increment=st.integers(min_value=1),
)
def test_nsize_permutations_with_higher_max_permutation_size(
    num_values: int, increment: int
):
    with pytest.raises(ValueError):
        tools.nsize_permutations(
            num_values=num_values, max_permutation_size=num_values + increment
        )
