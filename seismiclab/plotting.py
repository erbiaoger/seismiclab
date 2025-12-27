from __future__ import annotations

import numpy as np
from matplotlib import colors, pyplot as plt


def seismic_colormap(iop: int = 1) -> colors.ListedColormap:
    """
    Return a ListedColormap roughly matching ``seismic.m`` options.
    """

    N = 40
    L = 40

    if iop == 1:
        u1 = np.concatenate(
            [0.5 * np.ones(N), np.linspace(0.5, 1, 128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)]
        )
        u2 = np.concatenate(
            [0.25 * np.ones(N), np.linspace(0.25, 1, 128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)]
        )
        u3 = np.concatenate(
            [np.zeros(N), np.linspace(0.0, 1, 128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)]
        )
    elif iop == 2:
        u1 = np.concatenate([np.ones(N), np.ones(128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)])
        u2 = np.concatenate([np.zeros(N), np.linspace(0.0, 1, 128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)])
        u3 = np.concatenate([np.zeros(N), np.linspace(0.0, 1, 128 - N), np.linspace(1, 0, 128 - N), np.zeros(N)])
    elif iop == 3:
        u1 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0.0, 1, 128 - N - L // 2),
                np.ones(L),
                np.linspace(1, 0.5, 128 - L // 2),
            ]
        )
        u2 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0.0, 1, 128 - N - L // 2),
                np.ones(L),
                np.linspace(1, 0.0, 128 - N - L // 2),
                np.zeros(N),
            ]
        )
        u3 = np.concatenate(
            [
                np.linspace(0.5, 1, 128 - L // 2),
                np.ones(L),
                np.linspace(1, 0.0, 128 - N - L // 2),
                np.zeros(N),
            ]
        )
    else:
        n = np.arange(256)
        u1 = np.interp(n, [0, 20, 130, 256], [1, 0, 1, 1])
        u2 = np.interp(n, [0, 20, 256], [1, 0, 0])
        u3 = np.interp(n, [0, 20, 256], [1, 1, 0])

    M = np.stack([u1, u2, u3], axis=1)
    return colors.ListedColormap(M)


def clip(d: np.ndarray, cmin: float, cmax: float | None = None) -> np.ndarray:
    """
    Clip amplitudes symmetrically (single percentage) or asymmetrically.
    """

    d = np.asarray(d).copy()
    if cmax is None:
        dmax = (cmin / 100.0) * np.max(d)
        dmin = -dmax
    else:
        dmax = (cmax / 100.0) * np.max(d)
        dmin = (cmin / 100.0) * np.min(d)
    d = np.asarray(d).copy()
    d[d > dmax] = dmax
    d[d < dmin] = dmin
    return d


def wigb(a: np.ndarray, scal: float = 1.0, x: np.ndarray | None = None, z: np.ndarray | None = None, amx=None):
    """
    Basic wiggle plot matching the original signature.
    """

    nz, nx = a.shape
    if x is None:
        x = np.arange(nx)
    if z is None:
        z = np.arange(nz)
    if amx is None:
        amx = np.mean(np.max(np.abs(a), axis=0))

    dx = np.median(np.abs(np.diff(x))) if nx > 1 else 1.0
    a_scaled = a * dx / amx * scal

    plt.gca().invert_yaxis()
    for i in range(nx):
        trace = a_scaled[:, i] + x[i]
        plt.plot(trace, z, color="black", linewidth=0.8)
        plt.fill_betweenx(z, x[i], trace, where=trace > x[i], color="black", alpha=0.3)


def plot_wb(t: np.ndarray, w: np.ndarray, a: float):
    """
    Horizontal plotting with bias between columns.
    """

    w = w / np.max(np.abs(w))
    nt, nx = w.shape
    x = np.arange(1, nx + 1)
    for i in range(nx):
        plt.plot(t, (1 + a / 100.0) * w[:, i] + x[i], "b", linewidth=1.0)
    plt.axis("tight")


def pimage(xaxis: np.ndarray, yaxis: np.ndarray, data: np.ndarray):
    """
    Thin wrapper around imshow that sets axes similar to MATLAB's imagesc.
    """

    extent = [xaxis[0], xaxis[-1], yaxis[-1], yaxis[0]]
    plt.imshow(data, aspect="auto", extent=extent, cmap=seismic_colormap())
    plt.gca().invert_yaxis()


def plot_spectral_attributes(
    t0: float, w: np.ndarray, dt: float, fmax: float,
    N: int, pos1: int, pos2: int
):
    """
    Plot amplitude and phase of a wavelet.

    转换自 MATLAB: codes/seismic_plots/plot_spectral_attributes.m

    Parameters
    ----------
    t0 : float
        Time in sec of the first sample of the wavelet
    w : np.ndarray
        Input data or wavelet
    dt : float
        Sampling interval in secs
    fmax : float
        Max frequency to display in Hz
    N : int
        Plot phase in the interval -N*180, N*180
    pos1 : int
        Subplot position for amplitude (e.g., 221)
    pos2 : int
        Subplot position for phase (e.g., 222)

    Examples
    --------
    >>> import numpy as np
    >>> from seismiclab_py.synthetics import rotated_wavelet
    >>> dt = 4./1000
    >>> w, tw = rotated_wavelet(dt, 2, 90, 90)
    >>> plot_spectral_attributes(min(tw), w, dt, 125, 1, 221, 222)
    """
    nf = 4 * (2 ** int(np.ceil(np.log2(len(w)))))
    n2 = nf // 2 + 1

    X = np.fft.fft(w, nf)
    fa = np.arange(nf) / nf / dt

    # Tell the DFT that the first sample is at t0 (to avoid unwrapping)
    Phase_Shift = np.exp(-1j * 2 * np.pi * fa * t0)
    X = X * Phase_Shift

    n2 = int(np.floor(fmax * (nf * dt))) + 1
    X = X[:n2]
    f = np.arange(1, n2 + 1)
    f = (f - 1) / nf / dt

    A = np.abs(X)
    A = A / np.max(A)
    Theta = np.unwrap(np.angle(X))
    Theta[0] = 0.0
    Theta = Theta * 180 / np.pi

    # Plot amplitude and phase
    plt.subplot(pos1)
    plt.plot(f, A)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.xlim(0, fmax)
    plt.ylim(0, 1.1)
    plt.grid(True)

    plt.subplot(pos2)
    plt.plot(f, Theta)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [Deg]')
    plt.xlim(0, fmax)
    plt.ylim(-N * 180, N * 180)
    plt.grid(True)


def sgray(alpha: float):
    """
    Non-linear transformation of a gray colormap. Similar to clipping.

    转换自 MATLAB: codes/seismic_plots/sgray.m

    Parameters
    ----------
    alpha : float
        Degree of BW color scale clustering (try 0.5)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from seismiclab_py.synthetics import linear_events
    >>> d = linear_events()
    >>> plt.imshow(d)
    >>> sgray(0.5)
    """
    i0 = 32
    i = np.arange(1, 65)
    t = np.arctan((i - i0) / alpha)
    s = t[63]
    t = (t - np.min(t)) / (np.max(t) - np.min(t))

    m = np.zeros((64, 3))
    m[:, 1] = 1 - t
    m[:, 0] = 1 - t
    m[:, 2] = 1 - t

    plt.colormap(m.tolist())
