"""
Utils for combinatorial problems.
"""

import math
from typing import Sequence


def sequence_to_integer(space_size: int, sequence: Sequence[int]) -> int:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.

    Based on https://2ality.com/2013/03/permutations.html.

    Args:
        space_size: the number of possible digits
        sequence_size: the length of the sequence of digits.
        index: the index of the unique sequence.
    """
    id = 0
    for idx, value_index in enumerate(reversed(sequence)):
        id = id + value_index * int(pow(space_size, idx))
    return id


def interger_to_sequence(
    space_size: int, sequence_length: int, index: int
) -> Sequence[int]:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.

    Based on https://2ality.com/2013/03/permutations.html.

    Args:
        space_size: the number of possible digits
        sequence_length: the length of the sequence of digits.
        index: the index of the unique sequence.
    """
    xs = []
    for pw in reversed(range(sequence_length)):
        if pw == 0:
            xs.append(index)
        else:
            mult = space_size**pw
            digit = math.floor(index / mult)
            xs.append(digit)
            index = index % mult
    return tuple(xs)
