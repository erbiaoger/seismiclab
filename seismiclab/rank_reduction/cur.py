"""
CUR decomposition for low-rank approximation.

转换自 MATLAB: codes/rank_reduction/cur.m
"""

import numpy as np
from numpy.random import default_rng
from scipy.linalg import pinv


def cur(M: np.ndarray, p: int) -> np.ndarray:
    """
    CUR decomposition for low-rank approximation.

    Parameters
    ----------
    M : np.ndarray
        Input matrix (n1 x n2)
    p : int
        Desired rank (number of columns/rows to sample)

    Returns
    -------
    DD : np.ndarray
        Low-rank approximation C * U * R

    Notes
    -----
    The CUR decomposition approximates a matrix M as C * U * R where:
    - C contains p columns of M
    - R contains p rows of M
    - U is a small p x p matrix

    Columns and rows are sampled according to their importance (leverage scores).

    Examples
    --------
    >>> import numpy as np
    >>> M = np.random.randn(100, 50)
    >>> M_approx = cur(M, 5)
    """
    n1, n2 = M.shape

    # Compute column importance (leverage scores)
    e = np.sum(np.abs(M) ** 2, axis=0)
    normalization = np.sum(e)
    Py = e / normalization

    # Compute row importance (leverage scores)
    e = np.sum(np.abs(M) ** 2, axis=1)
    normalization = np.sum(e)
    Px = e / normalization

    # Sample columns and rows according to importance
    rng = default_rng()
    i2 = rng.choice(n2, size=p, replace=False, p=Py)
    i1 = rng.choice(n1, size=p, replace=False, p=Px)

    # Extract columns and rows
    C = M[:, i2]
    R = M[i1, :]

    # Compute pseudo-inverses
    C_inv = pinv(C)
    R_inv = pinv(R)

    # Form middle matrix
    U = C_inv @ M @ R_inv

    # Construct CUR approximation
    DD = C @ U @ R

    return DD
