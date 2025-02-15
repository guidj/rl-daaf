import itertools
from typing import Any, Sequence


def nsize_permutations(
    num_values: int, max_permutation_size: int, stub: int = -1
) -> Sequence[Any]:
    """
    Generates a list of permutations of 1 to k elements from a set of m values.
    m = num_values, k = max_permutation_size.
    Permutations of size < k have padded to size k using `stub`.
    Args:
        num_values: The number of values in the set.
        max_permutation_size: A number indicating the largest permutation set size.
        stub: A value used to pad permutations sets smaller than `max_permutation_size`.
    Returns:
        A list of permutations of actions, sized 1 to `num_values`.
        Total list size is sum([A**k for k in range(1, K  + 1)]).
    E.g. m = 3, k = 2, no. elements = 3 ** 1 + 3 ** 2 = 3 + 9 = 12
    (0, -1)
    (0, 0)
    (0, 1)
    (0, 2)
    (1, -1)
    (1, 0)
    (1, 1)
    (1, 2)
    (2, -1)
    (2, 0)
    (2, 1)
    (2, 2)
    """
    if max_permutation_size > num_values:
        raise ValueError(
            f"max_permutation_size cannot be larger than num_values: {max_permutation_size} > num_values"
        )
    kstep_actions_permutations = []
    for kstep in range(1, max_permutation_size + 1):
        iterators = [range(num_values) for _ in range(kstep)]
        kstep_actions = list(itertools.product(*iterators))
        padded_nstep_actions = [
            actions + (stub,) * (max_permutation_size - len(actions))
            for actions in kstep_actions
        ]
        kstep_actions_permutations.extend(padded_nstep_actions)
    return kstep_actions_permutations
