"""
Mutes (taper) operator for seismic data processing.

转换自 MATLAB: codes/linear_operators/Mutes_operator.m
"""

import numpy as np
from typing import Any


def Mutes_operator(in_data: np.ndarray, Param: Any, flag: int = 1) -> np.ndarray:
    """
    Apply mutes (taper) to seismic data.

    Parameters
    ----------
    in_data : np.ndarray
        Input seismic data
    Param : object
        Parameters with attribute 'Mutes' (the mute/taper pattern)
    flag : int, optional
        Operator flag (not used, kept for API compatibility)

    Returns
    -------
    out : np.ndarray
        Data with mutes applied (element-wise multiplication)

    Notes
    -----
    This operator applies a mute pattern to seismic data via element-wise
    multiplication. The mute pattern typically has values between 0 (muted)
    and 1 (pass).

    Examples
    --------
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> data = np.random.randn(100, 50)
    >>> mutes = np.ones((100, 50))
    >>> mutes[:20, :] = 0  # Mute first 20 samples
    >>> Param = SimpleNamespace(Mutes=mutes)
    >>> muted = Mutes_operator(data, Param)
    """
    Mutes = Param.Mutes

    out = in_data * Mutes

    return out
