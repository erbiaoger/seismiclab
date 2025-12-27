"""
Phase correction for seismic data.

转换自 MATLAB: codes/dephasing/phase_correction.m
"""

import numpy as np
from scipy.fft import fft, ifft


def phase_correction(din: np.ndarray, c: float) -> np.ndarray:
    """
    Apply a constant phase correction to seismic data.

    Parameters
    ----------
    din : np.ndarray
        Seismic data (nt x nx, traces in columns)
    c : float
        Constant phase rotation in degrees

    Returns
    -------
    dout : np.ndarray
        Data after phase correction

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 50)
    >>> corrected = phase_correction(data, c=30)  # 30 degree phase shift
    """
    c_rad = c * np.pi / 180
    nt, nx = din.shape
    nf = int(2 ** np.ceil(np.log2(nt)))

    Din = fft(din, n=nf, axis=0)

    # Create phase correction filter
    Phase = np.concatenate([
        [0],
        -c_rad * np.ones(nf // 2 - 1),
        [0],
        c_rad * np.ones(nf // 2 - 1)
    ])
    Phase = np.exp(-1j * Phase)

    # Apply phase correction
    for k in range(nx):
        Din[:, k] = Din[:, k] * Phase

    dout = ifft(Din, n=nf, axis=0)
    dout = np.real(dout[:nt, :])

    return dout


def kurtosis_of_traces(x: np.ndarray) -> float:
    """
    Compute kurtosis of one or more time series.

    Parameters
    ----------
    x : np.ndarray
        Data (vector or matrix)

    Returns
    -------
    K : float
        Kurtosis

    Notes
    -----
    Kurtosis is defined as K = E(x^4) / (E(x^2))^2

    - K = 3 for a Gaussian series
    - Kurtosis Excess k' = K - 3, where k' = 0 for Gaussian

    Reference: Longbottom, J., Walden, A.T. and White, R.E. (1988)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> K = kurtosis_of_traces(x)  # Should be ~3 for Gaussian
    """
    ndims = x.ndim

    if ndims == 1:
        n1 = len(x)
        Ns = n1
        sum1 = np.sum(x ** 4) / Ns
        sum2 = (np.sum(x ** 2) / Ns) ** 2
    else:
        n1, n2 = x.shape
        Ns = n1 * n2
        sum1 = np.sum(np.sum(x ** 4) / Ns)
        sum2 = (np.sum(np.sum(x ** 2)) / Ns) ** 2

    K = sum1 / sum2

    return K
