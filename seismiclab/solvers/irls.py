"""
Iteratively Re-weighted Least Squares for sparse solutions.

转换自 MATLAB: codes/solvers/irls.m
"""

import numpy as np
from typing import Callable, Any
from .cglsw import cglsw


def irls(x0: np.ndarray, b: np.ndarray, operator: Callable, Param: Any,
         mu: float, max_iter_cgls: int, max_iter_irls: int,
         tol1: float, tol2: float) -> tuple:
    """
    Iterative Re-weighted Least Squares for sparse solutions.

    Finds x that minimizes J = ||A x - b||_2^2 + mu ||x||_1.
    The solution is estimated by solving a sequence of quadratic problems.

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
    mu : float
        Trade-off parameter
    max_iter_cgls : int
        Maximum iterations for inner CGLS
    max_iter_irls : int
        Maximum number of IRLS iterations
    tol1 : float
        Tolerance for CGLS (e.g., 1e-6)
    tol2 : float
        Tolerance for IRLS outer loop (e.g., 1e-4)

    Returns
    -------
    x : np.ndarray
        Solution
    J : np.ndarray
        Cost vs iteration

    Notes
    -----
    tol1: Stopping criterion for CGLS. Stops when normalized l2 norm of
    the gradient of the quadratic cost is less than tol1.

    tol2: Outer loop stops when normalized l2 norm of the gradient of J
    is less than tol2.

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
    >>> x, J = irls(x0, b, operator, None, mu=0.1,
    ...             max_iter_cgls=100, max_iter_irls=50,
    ...             tol1=1e-6, tol2=1e-4)
    """
    Wr = np.ones_like(b)
    Wx = np.ones_like(x0)
    u0 = np.zeros_like(x0)

    Diff = 99999.
    J = np.zeros(max_iter_irls)

    for j in range(max_iter_irls):
        # Solve weighted least squares
        u = cglsw(u0, b, operator, Param, Wr, Wx, mu,
                  max_iter_cgls, tol1, 0)
        x = Wx * u

        # Update weights
        Wx = np.sqrt(np.abs(x) + 0.00001)

        # Compute residual and cost
        e = Wr * (operator(x, Param, 1) - b)
        J[j] = (np.sum(np.abs(e.ravel()) ** 2) +
                mu * np.sum(np.abs(x.ravel())))

        # Check convergence
        if j > 1:
            Diff = abs(J[j] - J[j - 1]) / ((J[j] + J[j - 1]) / 2)

        if Diff < tol2:
            break

    print(f' IRLS ended after {j + 1:4.0f} iterations of {max_iter_irls:4.0f}')

    return x, J[:j + 1]
