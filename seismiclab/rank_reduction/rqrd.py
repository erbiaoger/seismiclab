"""
Randomized QR decomposition for low-rank approximation.

转换自 MATLAB: codes/rank_reduction/rqrd.m
"""

import numpy as np
from scipy.linalg import qr


def rqrd(in_mat: np.ndarray, p: int) -> np.ndarray:
    """
    Randomized QR decomposition for low-rank approximation.

    Parameters
    ----------
    in_mat : np.ndarray
        Input matrix (m x n)
    p : int
        Desired rank

    Returns
    -------
    out : np.ndarray
        Reduced-rank matrix

    Notes
    -----
    Reference: Liberty et al., 2007, Randomized algorithms for the low-rank
    approximation of matrices, PNAS, 104 (51) 20167-20172

    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.randn(100, 50)
    >>> A_rank5 = rqrd(A, 5)
    """
    _, ny = in_mat.shape

    # Random projection
    omega = np.random.randn(ny, p)

    # Sketch
    y = in_mat @ omega

    # QR decomposition (economy mode)
    U, _ = qr(y, mode='economic')

    # Project back
    out = U @ (U.T @ in_mat)

    return out
