"""
Thresholding operations for sparse optimization.

转换自 MATLAB: codes/solvers/thresholding.m
"""

import numpy as np


def thresholding(x: np.ndarray, sorh: str, t: float) -> np.ndarray:
    """
    Perform soft or hard thresholding.

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix
    sorh : str
        Thresholding type: 's' for soft, 'h' for hard
    t : float
        Threshold value

    Returns
    -------
    y : np.ndarray
        Thresholded output

    Notes
    -----
    Soft thresholding: Y = SIGN(X) * (|X| - T)+
    Hard thresholding: Y = X * 1_(|X| > T)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    >>> thresholding(x, 's', 1.0)
    array([0. , 0. , 0. , 0.5, 1. ])
    >>> thresholding(x, 'h', 1.0)
    array([0. , 0. , 0. , 1.5, 2. ])
    """
    x = np.asarray(x)

    if sorh == 's':
        # Soft thresholding: Y = SIGN(X) * ((|X|-T)+)
        tmp = (np.abs(x) - t)
        tmp = (tmp + np.abs(tmp)) / 2
        y = np.sign(x) * tmp

    elif sorh == 'h':
        # Hard thresholding: Y = X * 1_(|X| > T)
        y = x * (np.abs(x) > t)

    else:
        raise ValueError(f"Invalid thresholding type: {sorh}. Must be 's' or 'h'.")

    return y
