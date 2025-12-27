"""
Conjugate Gradient for Least Squares with weighting.

转换自 MATLAB: codes/solvers/cglsw.m
"""

import numpy as np
from typing import Callable, Any
from .cgdot import cgdot


def cglsw(x0: np.ndarray, b: np.ndarray, operator: Callable, Param: Any,
          Wr: np.ndarray, Wx: np.ndarray, mu: float, max_iter: int,
          tol: float, prnt: int = 0) -> np.ndarray:
    """
    Solve weighted least squares via conjugate gradient.

    Minimizes J = ||Wr ( A.Wx.x - b) ||_2^2 + mu ||x||_2^2

    Parameters
    ----------
    x0 : np.ndarray
        Starting point
    b : np.ndarray
        Data
    operator : callable
        Linear operator:
        - operator(in, Param, 1) = A * in
        - operator(in, Param, -1) = A^T * in
    Param : Any
        Parameters for the operator
    Wr : np.ndarray
        Element-wise weights for residuals
    Wx : np.ndarray
        Element-wise weights for model parameters
    mu : float
        Trade-off parameter
    max_iter : int
        Maximum number of iterations
    tol : float
        Stopping criterion (e.g., 1e-6)
    prnt : int, optional
        1 to print diagnostics, 0 for silent (default: 0)

    Returns
    -------
    x : np.ndarray
        Solution

    Notes
    -----
    This program is needed by irls.py
    Based on code by Saunders (https://web.stanford.edu/group/SOL/software/cgls/)

    Examples
    --------
    >>> import numpy as np
    >>> # Define a simple operator
    >>> def operator(x, param, mode):
    ...     if mode == 1:
    ...         return x  # A = I
    ...     else:
    ...         return x  # A^T = I
    >>> x0 = np.zeros(100)
    >>> b = np.random.randn(100)
    >>> Wr = np.ones_like(b)
    >>> Wx = np.ones_like(x0)
    >>> x = cglsw(x0, b, operator, None, Wr, Wx, mu=0.1, max_iter=100, tol=1e-6)
    """
    b = Wr * b

    x = x0.copy()
    r = b - Wr * operator(Wx * x, Param, 1)
    s = Wx * operator(Wr * r, Param, -1) - mu * x
    p = s.copy()

    gamma = cgdot(s, s)
    norms0 = np.sqrt(gamma)  # norm of gradient used to stop
    k = 0
    flag = 0

    if prnt:
        print(' ============================================== ')
        print(' ================= CGLS ======================= ')
        print(' ============================================== ')
        print('      k           |grad|       |grad|/|grad_0|')
        print(f'   {k:3.0f}       {norms0:12.5g}           {1.0:8.3g}')

    while (k < max_iter) and (flag == 0):
        q = Wr * operator(Wx * p, Param, 1)
        delta = cgdot(q, q) + mu * cgdot(p, p)
        if delta == 0:
            delta = 1e-10
        alpha = gamma / delta
        x = x + alpha * p
        r = r - alpha * q
        s = Wx * operator(Wr * r, Param, -1) - mu * x

        gamma1 = cgdot(s, s)
        norms = np.sqrt(gamma1)
        beta = gamma1 / gamma
        gamma = gamma1

        p = s + beta * p

        flag = (norms <= norms0 * tol)
        nres = norms / norms0 if norms0 > 0 else 0

        k = k + 1

        if prnt:
            print(f'   {k:3.0f}       {norms:12.5g}           {nres:8.3g}')

    # Diagnostics
    if k == max_iter:
        flag = 2

    if prnt:
        print(' ============================================== ')
        if flag == 1:
            print(' ====== CGLS converged before max_iter ======== ')
        if flag == 2:
            print(' ====== CGLS reached max_iter ================= ')
        print(' ============================================== \n \n')

    return x
