from __future__ import annotations

import numpy as np
from scipy.signal import windows, convolve2d


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(2 ** np.ceil(np.log2(n)))


def fk_spectra(d: np.ndarray, dt: float, dx: float, L: int) -> tuple:
    """
    FK spectrum of a seismic gather.

    Parameters
    ----------
    d : np.ndarray
        Data (traces in columns)
    dt : float
        Time interval
    dx : float
        Spatial increment between traces
    L : int
        Apply spectral smoothing using separable 2D Hamming window of LxL samples

    Returns
    -------
    S : np.ndarray
        FK spectrum
    k : np.ndarray
        Wavenumber axis in cycles/m (if dx is in meters)
    f : np.ndarray
        Frequency axis in Hz

    Notes
    -----
    When plotting spectra (S), use log(S) or S^alpha (alpha=0.1-0.3) to
    increase the visibility of small events.

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(1000, 50)
    >>> dt, dx = 0.004, 10.0
    >>> S, k, f = fk_spectra(d, dt, dx, 6)
    """
    nt, nx = d.shape

    nk = 4 * (2 ** _nextpow2(nx))
    nf = 4 * (2 ** _nextpow2(nt))

    # Compute 2D FFT
    S = np.fft.fftshift(np.abs(np.fft.fft2(d, s=(nf, nk))))
    S = S ** 2

    # Create 2D Hamming window
    H = windows.hamming(L)
    H2D = np.outer(H, H)
    H2D = H2D / np.sum(H2D)

    # Smooth spectrum
    S = convolve2d(S, H2D, mode='same')

    # Extract positive frequencies
    S = S[nf // 2:, :]

    # Frequency and wavenumber axes
    f = np.arange(nf // 2 + 1) / (nf * dt)
    k = np.arange(-nk // 2 + 1, nk // 2 + 1) / (nk * dx)

    return S, k, f


def smooth_spectrum(
    d: np.ndarray, dt: float, L: int, io: str = "li"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Power spectrum estimation by smoothing the periodogram.
    """

    d = np.asarray(d, dtype=float)
    nt = d.shape[0]
    aux = d.reshape(nt, -1)

    wind = windows.hamming(2 * L + 1)
    nf = max(2 * 2 ** (int(np.ceil(np.log2(nt)))), 2048)
    f = (np.arange(nf // 2 + 1)) / (dt * nf)

    D = np.fft.fft(aux, n=nf, axis=0)
    D = np.sum(np.abs(D) ** 2, axis=1)
    D = np.convolve(D, wind, mode="same")
    A = np.sqrt(D)
    D = D[: nf // 2 + 1]
    D = D / np.max(D)

    if io == "db":
        P = 10 * np.log10(D)
        P[P < -40] = -40
    else:
        P = D

    # Optional zero-phase wavelet estimate (matches MATLAB API shape)
    f0 = 30.0
    Lw = int(3 / (f0 * dt))
    w = np.real(np.fft.fftshift(np.fft.ifft(A)))
    mid = len(w) // 2
    w = w[mid - Lw : mid + Lw + 1]
    w = w * windows.hamming(len(w))
    w = w / np.max(np.abs(w))
    tw = np.arange(-Lw, Lw + 1) * dt

    return P, f, w, tw
