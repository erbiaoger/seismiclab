from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, windows
import matplotlib.pyplot as plt


def _convmtx(w: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(w).flatten()
    m = len(w)
    col = np.concatenate([w, np.zeros(n - 1)])
    row = np.concatenate([[w[0]], np.zeros(n - 1)])
    return toeplitz(col, row)[: m + n - 1, :n]


def taper(d: np.ndarray, perc_beg: float, perc_end: float):
    d = np.asarray(d, dtype=float)
    nt, nx = d.shape if d.ndim == 2 else (d.shape[0], 1)
    i1 = int(np.floor(nt * perc_beg / 100.0)) + 1
    i2 = int(np.floor(nt * perc_end / 100.0)) + 1
    win = np.concatenate([np.arange(1, i1 + 1) / i1, np.ones(nt - i1 - i2), np.arange(i2, 0, -1) / i2])
    if d.ndim == 1:
        dout = d * win
    else:
        dout = d * win[:, None]
    return dout, win


def sparse_decon(d: np.ndarray, w: np.ndarray, mu: float, iter_max: int, dt: float | None = None):
    w = np.asarray(w).flatten()
    d = np.asarray(d, dtype=float)
    nw = len(w)
    nt, ntraces = d.shape
    d_pad = np.vstack([d, np.zeros((nw - 1, ntraces))])
    W = _convmtx(w, nt)
    R = W.T @ W
    R0 = np.trace(R)
    Q_base = 0.001 * R0
    refl = np.zeros((nt, ntraces))

    for itrace in range(ntraces):
        s = d_pad[:, itrace]
        r = np.zeros(nt)
        Q = Q_base * np.eye(nt)
        for _ in range(iter_max):
            g = W.T @ s
            Matrix = R + Q
            r = np.linalg.solve(Matrix, g)
            sc = 0.001
            Q = mu * np.diag(1.0 / (np.abs(r) + sc))
        refl[:, itrace] = r

    n2 = nw // 2
    dp = convolve2d(refl, w[:, None], mode="full")[:nt, :]
    refl_out = np.vstack([np.zeros((n2, ntraces)), refl[: nt - n2, :]])
    return refl_out, dp


def spiking(d: np.ndarray, NF: int, mu: float):
    d = np.asarray(d, dtype=float)
    NF_order = NF - 1
    ns = d.shape[0]
    dmax = np.max(np.abs(d))

    def autocorr(x):
        ac = np.correlate(x, x, mode="full")
        mid = len(ac) // 2
        # Apply Hamming window of length 2*NF_order+1 (as in MATLAB)
        ac = ac[mid - NF_order : mid + NF_order + 1]
        ac = ac * np.hamming(2 * NF_order + 1)
        return ac[NF_order:]

    r = autocorr(d[:, 0])
    if d.ndim > 1 and d.shape[1] > 1:
        for k in range(1, d.shape[1]):
            r += autocorr(d[:, k])
        r /= d.shape[1]
    r[0] = r[0] * (1 + mu / 100.0)
    R = toeplitz(r)
    e = np.zeros_like(r)
    e[0] = 1.0
    f = np.linalg.solve(R, e)

    o = convolve2d(d, f[:, None], mode="full")[:ns, :]
    omax = np.max(np.abs(o))
    if omax != 0:
        o = o * dmax / omax
    return f.reshape(-1, 1), o


def ls_inv_filter(w: np.ndarray, NF: int, Lag: int, mu: float):
    w = np.asarray(w).flatten()
    NW = len(w)
    NO = NW + NF - 1
    b = np.zeros(NO)
    b[Lag - 1] = 1.0
    C = _convmtx(w, NF)
    R = C.T @ C
    r0 = R[0, 0]
    R = R + r0 * mu / 100.0
    rhs = C.T @ b
    f = np.linalg.solve(R, rhs)
    o = np.convolve(f, w)
    return f.reshape(-1, 1), o


def delay(d1: np.ndarray, d2: np.ndarray, max_delay: int):
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    nt, nx = d1.shape
    r = np.zeros(2 * max_delay + 1)
    for k in range(nx):
        r += np.correlate(d1[:, k], d2[:, k], mode="full")[nt - 1 - max_delay : nt + max_delay]
    Lag = np.argmax(r)
    time_to_move = Lag - max_delay
    if time_to_move > 0:
        dout = np.vstack([np.zeros((time_to_move, nx)), d2])[:nt, :]
    elif time_to_move < 0:
        ll = -time_to_move
        dout = np.vstack([d2[ll:, :], np.zeros((ll, nx))])
    else:
        dout = d2
    return dout


def predictive(w: np.ndarray, NF: int, L: int, mu: float) -> tuple:
    """
    Predictive deconvolution filter.

    转换自 MATLAB: codes/decon/predictive.m

    Parameters
    ----------
    w : np.ndarray
        The wavelet or input trace
    NF : int
        Length of the inverse filter
    L : int
        Prediction distance
    mu : float
        Percentage of prewhitening

    Returns
    -------
    f : np.ndarray
        The filter
    o : np.ndarray
        The output or convolution of the filter with the wavelet/trace

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.randn(100)
    >>> f, o = predictive(w, 35, 20, 0.1)
    """
    w = np.asarray(w).flatten()
    NW = len(w)
    NO = NW + NF - 1

    # Ensure w is a column vector
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    b = np.zeros(NO)
    b[L:NW] = w[L:]
    b = np.concatenate([b, np.zeros(NO - NW + L)])

    C = _convmtx(w.flatten(), NF)
    R = C.T @ C
    r0 = R[0, 0]
    R = R + r0 * mu / 100.0
    rhs = C.T @ b
    f = np.linalg.solve(R, rhs)

    # Create the final filter
    if L == 1:
        f_final = np.concatenate([[1.0], -f])
    else:
        f_final = np.concatenate([[1.0], np.zeros(L - 1), -f])

    o = np.convolve(f_final, w.flatten())
    o = o[: len(w)]

    return f_final.reshape(-1, 1), o


def zeros_wav(w: np.ndarray) -> np.ndarray:
    """
    Compute the zeros of a wavelet.

    转换自 MATLAB: codes/decon/zeros_wav.m

    Parameters
    ----------
    w : np.ndarray
        A wavelet

    Returns
    -------
    z : np.ndarray
        Complex zeros

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.randn(50)
    >>> z = zeros_wav(w)
    """
    w = np.asarray(w).flatten()
    # Reverse the wavelet
    wr = w[::-1]
    z = np.roots(wr)
    return z


def kolmog(w: np.ndarray, type_of_input: str, L: int = None) -> np.ndarray:
    """
    Kolmogoroff spectral factorization.

    Given a wavelet, retrieves the minimum phase wavelet using
    Kolmogoroff factorization. If the input is a trace, the spectral
    factorization is applied to the autocorrelation after smoothing.

    转换自 MATLAB: codes/decon/kolmog.m

    Parameters
    ----------
    w : np.ndarray
        A wavelet of arbitrary phase if type_of_input = 'w'
        or a seismic trace if type_of_input = 't'
    type_of_input : str
        'w' for wavelet, 't' for trace
    L : int, optional
        Length of wavelet if type_of_input='t'

    Returns
    -------
    w_min : np.ndarray
        A minimum phase wavelet

    Notes
    -----
    Reference: Claerbout, 1976, Fundamentals of geophysical data processing

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.randn(50)
    >>> wmin = kolmog(w, 'w')
    """
    w = np.asarray(w).flatten()

    if type_of_input == 'w':
        nw = len(w)
        nfft = 8 * (2 ** _nextpow2(nw))
        W = np.log(np.abs(np.fft.fft(w, n=nfft)) + 0.00001)
        W = np.fft.ifft(W)
        # Zero out negative frequencies
        W[nfft // 2 + 1:] = 0
        W = 2.0 * W
        W[0] = W[0] / 2.0
        W = np.exp(np.fft.fft(W))
        w_min = np.real(np.fft.ifft(W))
        w_min = w_min[:nw]

    else:  # type_of_input == 't'
        nt = len(w)
        nfft = 8 * (2 ** _nextpow2(nt))
        nw = L
        # Cross-correlation
        A = np.correlate(w, w, mode='full')
        mid = len(A) // 2
        A = A[mid - L : mid + L + 1]
        A = A * windows.hamming(2 * L + 1)
        W = np.log(np.sqrt(np.abs(np.fft.fft(A, n=nfft))) + 0.00001)
        W = np.fft.ifft(W)
        # Zero out negative frequencies
        W[nfft // 2 + 1:] = 0
        W = 2.0 * W
        W[0] = W[0] / 2.0
        W = np.exp(np.fft.fft(W))
        w_min = np.real(np.fft.ifft(W))
        w_min = w_min[:nw]
        w_min_max = np.max(np.abs(w_min))
        if w_min_max != 0:
            w_min = w_min / w_min_max

    return w_min


def polar_plot(z: np.ndarray):
    """
    Plot the roots of a wavelet in polar coordinates.

    转换自 MATLAB: codes/decon/polar_plot.m

    Parameters
    ----------
    z : np.ndarray
        Zeros of the wavelet (can be computed with zeros_wav)

    Notes
    -----
    Some zeros might end up outside the plot (check axis).

    Examples
    --------
    >>> import numpy as np
    >>> from seismiclab_py.synthetics import ricker
    >>> w, _ = ricker(20, 0.004)
    >>> z = zeros_wav(w)
    >>> polar_plot(z)
    """
    # Create unit circle
    theta = np.arange(0, 2 * np.pi, 0.1)
    x = np.cos(theta)
    y = np.sin(theta)

    # Find zeros inside unit circle
    II = np.abs(z) < 1
    z1 = z[II]

    plt.figure()
    plt.plot(x, y)
    plt.hold(True)
    plt.plot(np.real(z), np.imag(z), 'sk', label='All zeros')
    plt.plot(np.real(z1), np.imag(z1), '*r', label='Zeros inside unit circle')
    plt.axis('equal')
    plt.axis([-2, 2, -2, 2])
    plt.legend()
    plt.grid(True)
    plt.title('Wavelet Zeros in Z-Plane')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(np.ceil(np.log2(n)))

