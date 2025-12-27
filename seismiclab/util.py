from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d, windows


def add_noise(din: np.ndarray, snr: float, length: int) -> tuple[np.ndarray, float]:
    """
    Add Gaussian noise with a prescribed SNR (power ratio).
    """

    din = np.asarray(din, dtype=float)
    nt = din.shape[0]
    x = din.reshape(nt, -1)
    h = windows.hamming(length)
    noise = np.random.randn(*x.shape)
    noise = convolve2d(noise, h[:, None], mode="same")
    alpha = np.sqrt(np.sum(x**2) / (snr * np.sum(noise**2)))
    noise_added = alpha * noise
    y = x + noise_added
    dout = y.reshape(din.shape)
    sigma = float(np.std(noise_added))
    return dout, sigma


def quality(d: np.ndarray, d0: np.ndarray) -> float:
    """
    Compute reconstruction quality in dB.
    """

    d = np.asarray(d)
    d0 = np.asarray(d0)
    err = (d - d0) ** 2
    return -10 * np.log10(np.sum(err) / np.sum(d0**2))


def chi2(dp: np.ndarray, d: np.ndarray, sigma: float) -> float:
    """
    Compute Chi^2 misfit.

    Parameters
    ----------
    dp : np.ndarray
        Estimated/predicted signal
    d : np.ndarray
        Observed signal
    sigma : float
        Noise level (standard deviation)

    Returns
    -------
    Chi2 : float
        Chi-squared misfit

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(100)
    >>> dp = d + 0.1 * np.random.randn(100)
    >>> chi2_val = chi2(dp, d, sigma=0.1)
    """
    e = (dp - d) ** 2
    Chi2 = np.sum(e) / sigma ** 2
    return Chi2


def perc(d: np.ndarray, a: float) -> float:
    """
    Get clip value for imagesc/display.

    Parameters
    ----------
    d : np.ndarray
        Input data of any dimension
    a : float
        Clip fraction (0, 1)

    Returns
    -------
    x : float
        The clip value

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(100, 50)
    >>> clip_val = perc(d, 0.9)
    >>> # Use for display: imshow(d, vmin=-clip_val, vmax=clip_val)
    """
    x = a * np.array([-1, 1]) * np.max(np.abs(d))
    return x
