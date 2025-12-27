"""
Band-pass filter for seismic data processing.

转换自 MATLAB: codes/bp_filter/bp_filter.m
"""

import numpy as np
from scipy.fft import fft, ifft


def bp_filter(d: np.ndarray, dt: float, f1: float, f2: float,
              f3: float, f4: float) -> np.ndarray:
    """
    Apply a band-pass filter to seismic data.

    Parameters
    ----------
    d : np.ndarray
        Input data (first dimension is time)
    dt : float
        Sampling interval in seconds
    f1, f2, f3, f4 : float
        Frequency corners in Hz defining the trapezoidal bandpass:
        - f1-f2: low-frequency ramp up
        - f2-f3: passband
        - f3-f4: high-frequency ramp down

    Returns
    -------
    o : np.ndarray
        Filtered data (same shape as input)

    Notes
    -----
    The filter has a trapezoidal frequency response:

        ^
        |     ___________
        |    /           \\   Amplitude spectrum
        |   /             \\
        |  /               \\
        |------------------------>
           f1 f2        f3 f4

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(1000, 50)
    >>> dout = bp_filter(d, dt=0.004, f1=5, f2=10, f3=40, f4=60)
    """
    ndims = d.ndim
    N = d.shape
    nt = N[0]

    # Reshape to (nt, ntraces)
    x = d.reshape(nt, -1)

    # FFT size
    k = int(np.ceil(np.log2(nt)))
    nf = 4 * (2 ** k)

    # Frequency indices
    i1 = int(np.floor(nf * f1 * dt)) + 1
    i2 = int(np.floor(nf * f2 * dt)) + 1
    i3 = int(np.floor(nf * f3 * dt)) + 1
    i4 = int(np.floor(nf * f4 * dt)) + 1

    # Create filter response
    up = np.arange(1, i2 - i1 + 1) / (i2 - i1)
    down = np.arange(i4 - i3, 0, -1) / (i4 - i3)
    aux1 = np.concatenate([
        np.zeros(i1),
        up,
        np.ones(i3 - i2),
        down,
        np.zeros(nf // 2 + 1 - i4)
    ])
    aux2 = np.flip(aux1[1:nf // 2])

    # Zero phase filter
    c = 0
    F = np.concatenate([aux1, aux2])
    Phase = (np.pi / 180.0) * np.concatenate([
        [0.0],
        -c * np.ones(nf // 2 - 1),
        [0.0],
        c * np.ones(nf // 2 - 1)
    ])
    Transfer = F * np.exp(-1j * Phase)

    # Apply filter
    X = fft(x, n=nf, axis=0)
    Y = Transfer[:, np.newaxis] * X

    o = ifft(Y, n=nf, axis=0)

    # Extract original time samples
    o = np.real(o[:nt, :])

    # Reshape back to original shape
    o = o.reshape([nt] + list(N[1:]))

    return o
