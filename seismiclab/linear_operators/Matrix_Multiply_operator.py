"""
Matrix multiply operator wrapper for iterative solvers.

转换自 MATLAB: codes/linear_operators/Matrix_Multiply_operator.m
"""

import numpy as np
from typing import Any


def Matrix_Multiply_operator(in_vec: np.ndarray, Param: Any, flag: int) -> np.ndarray:
    """
    Wrapper to use cgls/fista/irls when the linear operator is a matrix.

    Parameters
    ----------
    in_vec : np.ndarray
        Input vector or matrix
    Param : object
        Parameters with attribute 'A' (the matrix)
    flag : int
        1 for forward operator (A * in), -1 for adjoint (A^T * in)

    Returns
    -------
    out : np.ndarray
        Result of matrix multiplication

    Notes
    -----
    This is a simple wrapper for using CGLS/FISTA/IRLS with explicit matrices.
    Param should be an object with attribute A (e.g., a SimpleNamespace or dict).

    Examples
    --------
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> A = np.random.randn(50, 100)
    >>> Param = SimpleNamespace(A=A)
    >>> x = np.random.randn(100)
    >>> y = Matrix_Multiply_operator(x, Param, 1)  # A * x
    >>> x_adj = Matrix_Multiply_operator(y, Param, -1)  # A^T * y
    """
    A = Param.A

    if flag == 1:
        out = A @ in_vec
    else:
        out = A.T @ in_vec

    return out
