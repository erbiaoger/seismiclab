"""
Taper and gain functions for seismic data processing.

转换自 MATLAB: codes/scaling_tapering/taper.m, gain.m
"""

import numpy as np
from scipy.signal import convolve2d, windows


def taper(d: np.ndarray, perc_beg: float, perc_end: float) -> tuple:
    """
    Apply a triangular taper to the beginning/end of traces.

    Parameters
    ----------
    d : np.ndarray
        Data (nt x nx, columns are traces)
    perc_beg : float
        Percentage of data to be tapered at the beginning
    perc_end : float
        Percentage of data to be tapered at the end

    Returns
    -------
    dout : np.ndarray
        Data with tapered beginning/end
    win : np.ndarray
        The taper window (nt,)

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 50)
    >>> tapered, win = taper(data, 5, 5)  # 5% taper at both ends
    """
    nt, nx = d.shape

    i1 = int(np.floor(nt * perc_beg / 100)) + 1
    i2 = int(np.floor(nt * perc_end / 100)) + 1

    # Create triangular taper
    win = np.concatenate([
        np.arange(1, i1 + 1) / i1,
        np.ones(nt - i1 - i2),
        np.arange(i2, 0, -1) / i2
    ])

    dout = d * win[:, np.newaxis]

    return dout, win


def gain(d: np.ndarray, dt: float, option1: str, parameters: np.ndarray,
         option2: int = 0) -> np.ndarray:
    """
    Apply gain to seismic traces.

    Parameters
    ----------
    d : np.ndarray
        Traces (first dimension is time)
    dt : float
        Sampling interval
    option1 : str
        'time' for time-varying gain, 'agc' for automatic gain control
    parameters : np.ndarray
        - For 'time': [a, b] where gain = t^a * exp(-b*t)
        - For 'agc': [agc_gate] length of AGC gate in seconds
    option2 : int, optional
        0: No normalization
        1: Normalize by amplitude
        2: Normalize by RMS value

    Returns
    -------
    dout : np.ndarray
        Traces after gain application

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 50)
    >>> # AGC with 0.05s gate
    >>> gained = gain(data, 0.004, 'agc', np.array([0.05]), 1)
    """
    N = d.shape
    nt = N[0]
    x = d.reshape(nt, -1)

    if option1 == 'time':
        # Geometrical spreading-like gain
        a, b = parameters[0], parameters[1]
        t = np.arange(nt) * dt
        tgain = (t ** a) * np.exp(b * t)

        xout = x * tgain[:, np.newaxis]

    elif option1 == 'agc':
        # AGC
        L = int(parameters[0] / dt) + 1
        L = int(np.floor(L / 2))
        h = windows.hamming(2 * L + 1)

        xout = np.zeros_like(x)
        for k in range(x.shape[1]):
            aux = x[:, k]
            e = aux ** 2
            rms = np.sqrt(convolve2d(e.reshape(-1, 1), h.reshape(-1, 1), mode='same'))
            epsi = 1e-10 * np.max(rms)
            op = rms / (rms ** 2 + epsi)
            xout[:, k] = x[:, k] * op.ravel()
    else:
        xout = x.copy()

    # Normalize if requested
    if option2 == 1:
        # Normalize by amplitude
        xout = xout / np.max(np.abs(xout))
    elif option2 == 2:
        # Normalize by RMS
        xout = xout / np.sqrt(np.mean(xout ** 2))

    dout = xout.reshape(N)

    return dout


def envelope(d: np.ndarray) -> np.ndarray:
    """
    Compute envelope of seismic traces via Hilbert transform.

    Parameters
    ----------
    d : np.ndarray
        Input data

    Returns
    -------
    env : np.ndarray
        Envelope of the data

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 50)
    >>> env = envelope(data)
    """
    from scipy.signal import hilbert

    analytic = hilbert(d, axis=0)
    env = np.abs(analytic)

    return env
