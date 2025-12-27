"""
Randomized SVD for low-rank approximation.

转换自 MATLAB: codes/rank_reduction/rand_svd.m
"""

import numpy as np
from scipy.linalg import svd


def rand_svd(A: np.ndarray, k: int, l: int = None, i: int = 2) -> np.ndarray:
    """
    Randomized SVD to compute low-rank approximation.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (m x n)
    k : int
        Desired rank
    l : int, optional
        Oversampling parameter (default: l = 2*k)
    i : int, optional
        Number of power iterations (default: i = 2)

    Returns
    -------
    A_out : np.ndarray
        Reduced-rank matrix

    Notes
    -----
    l = 2*k seems to work well
    i = 2 is also a good option

    Reference: Liberty et al., 2007, Randomized algorithms for the low-rank
    approximation of matrices, PNAS, 104 (51) 20167-20172

    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.randn(100, 50)
    >>> A_rank5 = rand_svd(A, k=5)
    """
    m, n = A.shape

    if l is None:
        l = 2 * k

    # Random l x m matrix
    GG = np.random.randn(l, m)

    # Compute (A*A')^i
    AA = np.eye(m)
    for ii in range(i):
        AA = AA @ (A @ A.T)

    # Form range finder
    RR = GG @ AA @ A

    # SVD of RR'
    UUq, EEq, VVq = svd(RR.T, full_matrices=False)

    # Take first k columns
    QQ = UUq[:, :k]

    # Form sketch
    TT = A @ QQ

    # SVD of sketch
    UU, EE, W = svd(TT, full_matrices=False)

    # Form right singular vectors
    VV = QQ @ W

    # Construct rank-k approximation
    A_out = UU @ np.diag(EE) @ VV.T

    return A_out
