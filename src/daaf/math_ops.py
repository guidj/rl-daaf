"""
Math ops support module.
"""

import numpy as np
import tensorflow as tf
from scipy import linalg

_SOLVER_ROW_SWITCH = 650
_SOLVER_COL_SWITCH = 320


def meets_least_squares_sufficient_conditions(matrix: np.ndarray):
    """
    Checks if value are present for every factor to compute
    a least square estimate.
    """
    shape = matrix.shape
    if len(shape) != 2:
        raise ValueError(f"Only 2D tensors are supported. Got shape: {shape}")

    return shape[0] >= shape[1] and least_squares_factors_present(matrix)


def least_squares_factors_present(matrix: np.ndarray) -> bool:
    """
    Determines if a matrix has a non-zero value set for every column
    on at least one row.

    Args:
        examples: A 2D tensor, A, shaped m x n.

    Returns:
        True if every column has its value set at least among the rows.
    """
    shape = matrix.shape
    if len(shape) != 2:
        raise ValueError(f"Only 2D tensors are supported. Got shape: {shape}")

    columns_check = np.sum(matrix, axis=0)
    missing: int = np.sum(columns_check == 0)
    return missing == 0


def solve_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Computes the solution to n variables using Least-Squares.

    Args:
        matrix: A 2D numpy array, A, shaped m x n.
        rhs: A set of outcomes, b, shaped m.
        l2_regularizer: A regualization factor.

    Returns:
        The solution x to the least-squares problem Ax=b, shaped n.
    """
    shape = matrix.shape
    if len(shape) != 2:
        raise ValueError(f"Only 2D tensors are supported. Got shape: {shape}")
    return _np_solve_least_squares(matrix=matrix, rhs=rhs)


def _tf_np_solve_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Computes the solution to n variables using Least-Squares.

    Note: we switch between the scipy and tf solvers
    based on benchmarks on an 2,4 GHz 8-Core Intel Core i9,
    with 16 GB 2667 MHz DDR4, using square arrays,
    and sliced square arrays (fewer rows than columns).

    Args:
        matrix: A 2D tensor, A, shaped m x n.
        rhs: A set of outcomes, b, shaped m.
        l2_regularizer: A regualization factor.

    Returns:
        The solution x to the least-squares problem Ax=b, shaped n.
    """
    shape = matrix.shape
    if len(shape) != 2:
        raise ValueError(f"Only 2D tensors are supported. Got shape: {shape}")

    if shape[0] > _SOLVER_ROW_SWITCH or shape[1] > _SOLVER_COL_SWITCH:
        return _tf_solve_least_squares(matrix=matrix, rhs=rhs)
    return _np_solve_least_squares(matrix=matrix, rhs=rhs)


def _np_solve_least_squares(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Computes the solution to n variables using Least-Squares.

    Args:
        matrix: A 2D tensor, A, shaped m x n.
        rhs: A set of outcomes, b, shaped m.
        l2_regularizer: A regualization factor.

    Returns:
        The solution x to the least-squares problem Ax=b, shaped n.
    """
    try:
        solution, _, _, _ = linalg.lstsq(a=matrix, b=rhs, lapack_driver="gelsy")
        return solution # type: ignore
    except linalg.LinAlgError as err:
        # the computation failed, likely due to the matix being unsuitable (no solution).
        raise ValueError("Failed to solve linear system") from err


def _tf_solve_least_squares(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Computes the solution to n variables using Least-Squares.

    Args:
        matrix: A 2D tensor, A, shaped m x n.
        rhs: A set of outcomes, b, shaped m.
        l2_regularizer: A regualization factor.

    Returns:
        The solution x to the least-squares problem Ax=b, shaped n.
    """
    matrix = tf.expand_dims(matrix, axis=0)
    rhs = tf.expand_dims(rhs, axis=-1)
    try:
        return tf.squeeze(tf.linalg.lstsq(matrix=matrix, rhs=rhs)).numpy() # type: ignore
    except tf.errors.InvalidArgumentError as err:
        # the computation failed, likely due to the matix being unsuitable (no solution).
        raise ValueError("Failed to solve linear system") from err
