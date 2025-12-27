"""
Power iteration method for eigenvalue computation.

转换自 MATLAB: codes/solvers/power_method.m
"""

import numpy as np
from typing import Callable, Any


def power_method(x0: np.ndarray, Hop: Callable, PARAM: Any) -> float:
    """
    Power iteration method to compute max eigenvalue of H'H.

    This is needed to evaluate the step parameter of FISTA.

    Parameters
    ----------
    x0 : np.ndarray
        Initial seed with dimensions such that H'*H*x0 does not abort
    Hop : callable
        Linear operator that encapsulates H and H':
        - Hop(x, PARAM, 1) = H x
        - Hop(y, PARAM, -1) = H' y
    PARAM : Any
        Set of parameters needed by Hop (typically a structure or dict)

    Returns
    -------
    value : float
        Maximum eigenvalue of H'H

    Notes
    -----
    Uses 10 iterations of power method with normalization at each step.

    Examples
    --------
    >>> import numpy as np
    >>> # Define a simple operator H (identity)
    >>> def Hop(x, param, mode):
    ...     if mode == 1:
    ...         return x  # H = I
    ...     else:
    ...         return x  # H' = I
    >>> x0 = np.random.randn(100)
    >>> max_eig = power_method(x0, Hop, None)
    """
    x = x0.copy()

    for k in range(10):
        # aux = L x
        aux = Hop(x, PARAM, 1)
        # y = L'aux = L'L x
        y = Hop(aux, PARAM, -1)
        # Normalize
        n = np.linalg.norm(y.ravel())
        x = y / n
        value = n

        print(f'{k:6.0f} {value:10.4f}')

    return value
