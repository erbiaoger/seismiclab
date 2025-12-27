"""
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

转换自 MATLAB: codes/solvers/fista.m
"""

import numpy as np
from typing import Callable, Any, Tuple
from .power_method import power_method
from .thresholding import thresholding


def fista(x0: np.ndarray, y: np.ndarray, Hop: Callable, PARAM: Any,
          mu: float, Nit: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve l2-l1 problem via Fast Iterative Shrinkage-Thresholding Algorithm.

    Minimizes J = ||H x - y||_2^2 + mu ||x||_1

    Parameters
    ----------
    x0 : np.ndarray
        Initial solution (used to get size)
    y : np.ndarray
        Data vector
    Hop : callable
        Linear operator:
        - Hop(x, PARAM, 1) = H x (forward)
        - Hop(y, PARAM, -1) = H' y (adjoint)
    PARAM : Any
        Parameters for Hop operator
    mu : float
        Trade-off parameter for L1 regularization
    Nit : int
        Maximum number of iterations
    tol : float
        Convergence tolerance (relative change in cost)

    Returns
    -------
    x : np.ndarray
        Sparse solution
    J : np.ndarray
        Cost function vs iteration

    Notes
    -----
    Reference: Beck and Teboulle, 2009, A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems, SIAM J. Imaging Science, Vol 2 (1), 183-202

    Examples
    --------
    >>> import numpy as np
    >>> # Define a simple forward operator
    >>> def Hop(x, param, mode):
    ...     if mode == 1:
    ...         return x  # H = I
    ...     else:
    ...         return x  # H' = I
    >>> x0 = np.zeros(100)
    >>> y = np.random.randn(100)
    >>> x, J = fista(x0, y, Hop, None, mu=0.1, Nit=100, tol=1e-4)
    """
    # Use power method to get step length alpha
    x0_rand = np.random.randn(*x0.shape)
    alpha = 1.05 * power_method(x0_rand, Hop, PARAM)

    J = np.zeros(Nit)  # Objective function
    x = np.zeros_like(x0)
    T = mu / (2 * alpha)

    t = 1
    yk = x.copy()

    Diff = 10000

    for k in range(Nit):
        tmpx = x.copy()
        Hx = Hop(yk, PARAM, 1)

        # Gradient step + soft thresholding
        grad = Hop(y - Hx, PARAM, -1)
        x = thresholding(yk + grad / alpha, 's', T)

        # Compute cost function
        J[k] = (np.sum(np.abs(Hx.ravel() - y.ravel()) ** 2) +
                mu * np.sum(np.abs(x.ravel())))

        # Check convergence
        if k > 1:
            Diff = abs(J[k] - J[k - 1]) / ((J[k] + J[k - 1]) / 2)

        if Diff < tol:
            break

        # FISTA acceleration
        tmpt = t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        yk = x + (tmpt - 1) / t * (x - tmpx)

    print(f'FISTA ended after {k + 1:4.0f} iterations of {Nit:4.0f}')

    return x, J[:k + 1]
