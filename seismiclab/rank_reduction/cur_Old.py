"""
CUR decomposition (Old version).

转换自 MATLAB: codes/rank_reduction/cur_Old.m
"""

import numpy as np
from scipy.linalg import pinv


def cur_Old(M: np.ndarray, nr: int, nc: int, c: float = 0.2) -> tuple:
    """
    CUR decomposition (Old version with custom sampling).

    Parameters
    ----------
    M : np.ndarray
        Input data matrix (nx x ny)
    nr : int
        Number of rows to sample
    nc : int
        Number of columns to sample
    c : float, optional
        Weighting factor for probability function (default: 0.2)
        Smaller values (in [0,1]) typically give better results

    Returns
    -------
    C : np.ndarray
        Selected columns (nx x nc)
    U : np.ndarray
        Middle matrix (nc x nr)
    R : np.ndarray
        Selected rows (nr x ny)

    Notes
    -----
    CUR is a matrix decomposition where A = C*U*R.
    - Columns of C are randomly picked based on L2 norm of each column
    - Rows of R are picked similarly
    - U = pseudoinv(C) * M * pseudoinv(R)

    This is an older implementation with a different sampling strategy.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.random.randn(100, 50)
    >>> C, U, R = cur_Old(M, nr=5, nc=5)
    >>> M_approx = C @ U @ R
    """
    nx, ny = M.shape

    C = np.zeros((p, p))
    R = np.zeros((p, p))

    # Compute row and column norms
    ex = np.sum(np.abs(M) ** 2, axis=1)
    ey = np.sum(np.abs(M) ** 2, axis=0)
    ex = np.sqrt(ex)
    ey = np.sqrt(ey)

    exmax = np.max(ex)
    eymax = np.max(ey)

    # Sample rows
    count_R = 0
    ix = 0
    ex_marked = np.copy(ex)

    while ix < nx:
        b = exmax * np.random.rand()
        b2 = c * ex[ix]
        if b < b2:
            ex_marked[ix] = -1
            count_R += 1
        if count_R == nr:
            break
        ix += 1
        if count_R < nr and ix == nx:
            ix = 0

    # Extract marked rows
    for ix in range(nx):
        if ex_marked[ix] < 0:
            R = np.vstack([R, M[ix, :]])

    # Sample columns
    count_C = 0
    iy = 0
    ey_marked = np.copy(ey)

    while iy < ny:
        b = eymax * np.random.rand()
        if b < c * ey[iy]:
            ey_marked[iy] = -1
            count_C += 1
        if count_C == nc:
            break
        iy += 1
        if count_C < nc and iy == ny:
            iy = 0

    # Extract marked columns (transpose in original)
    for iy in range(ny):
        if ey_marked[iy] < 0:
            C = np.vstack([C, M[:, iy].T])

    C = C.T
    C_inv = pinv(C)
    R_inv = pinv(R)
    U = C_inv @ M @ R_inv

    return C, U, R
