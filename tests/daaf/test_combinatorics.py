import hypothesis
import hypothesis.strategies as st
import numpy as np

from daaf import combinatorics


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=10),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_interger_to_sequence(space_size: int, sequence_length: int):
    sample_index = np.random.default_rng().integers(0, space_size**sequence_length)
    assert 0 <= sample_index < space_size**sequence_length
    seq = combinatorics.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=sample_index
    )
    assert len(seq) == sequence_length
    assert all([element in set(range(space_size)) for element in seq])


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=10),
    sequence_length=st.integers(min_value=1, max_value=10),
    samples=st.integers(min_value=1, max_value=100),
)
@hypothesis.settings(deadline=None)
def test_sequence_to_integer(space_size: int, sequence_length: int, samples: int):
    for _ in range(samples):
        sequence = tuple(
            np.random.default_rng().integers(0, space_size, size=sequence_length)
        )
        index = combinatorics.sequence_to_integer(space_size, sequence=sequence)  # type: ignore
        assert 0 <= index < space_size**sequence_length

    # largest sequence
    sequence = tuple([space_size - 1] * sequence_length)
    index = combinatorics.sequence_to_integer(space_size, sequence=sequence)
    assert index == (space_size**sequence_length) - 1


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=10),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_interger_to_sequence_round_trip(space_size: int, sequence_length: int):
    index = np.random.default_rng().integers(0, space_size**sequence_length)
    seq = combinatorics.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=index
    )
    output = combinatorics.sequence_to_integer(space_size=space_size, sequence=seq)
    assert output == index
    assert len(seq) == sequence_length
    assert all([element in set(range(space_size)) for element in seq])


@hypothesis.given(
    space_size=st.integers(min_value=1, max_value=10),
    sequence_length=st.integers(min_value=1, max_value=10),
)
def test_sequence_to_integer_round_trip(space_size: int, sequence_length: int):
    sequence = tuple(
        np.random.default_rng().integers(0, space_size, size=sequence_length)
    )
    output_integer = combinatorics.sequence_to_integer(
        space_size=space_size, sequence=sequence
    )
    output_sequence = combinatorics.interger_to_sequence(
        space_size=space_size, sequence_length=sequence_length, index=output_integer
    )
    assert sequence == output_sequence
