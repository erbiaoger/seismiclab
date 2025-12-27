"""
Dot product function for conjugate gradient solvers.

转换自 MATLAB: codes/solvers/cgdot.m
"""

import numpy as np


def cgdot(in1: np.ndarray, in2: np.ndarray) -> float:
    """
    Flexible dot product function for CGLS.

    Works with linear operators with input/output that are not vectors.

    Parameters
    ----------
    in1, in2 : np.ndarray
        Vectors, matrices, cubes, etc. of the same size

    Returns
    -------
    out : float
        Inner product (real scalar)

    Notes
    -----
    This is a flexible dot product that works with arrays of any shape.
    The result is always a real scalar.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> cgdot(a, b)
    32.0
    """
    temp = in1 * np.conj(in2)
    out = np.sum(temp.ravel())
    return np.real(out)
