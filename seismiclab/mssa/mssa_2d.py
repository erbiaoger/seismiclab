"""
2D Multichannel Singular Spectrum Analysis (MSSA) for seismic data denoising.

转换自 MATLAB: codes/mssa/mssa_2d.m
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.sparse.linalg import svds
from typing import Tuple


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(2 ** np.ceil(np.log2(n)))


def mssa_2d(DATA: np.ndarray, dt: float, P: int, flow: float,
            fhigh: float, meth: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D MSSA filtering of a 2D gather.

    Parameters
    ----------
    DATA : np.ndarray
        2D array data (nt x nx), columns are traces DATA(t,x)
    dt : float
        Sampling interval in seconds
    P : int
        Size of subspace for noise attenuation (desired rank = number of dips)
    flow : float
        Minimum frequency in Hz
    fhigh : float
        Maximum frequency in Hz
    meth : int, optional
        1 for standard SVD, 2 for Randomized SVD (default: 1)

    Returns
    -------
    DATA_f : np.ndarray
        Filtered data
    sing : np.ndarray
        Singular values (empty for randomized SVD)

    Notes
    -----
    Reference: Oropeza and Sacchi, 2011, Simultaneous seismic de-noising and
    reconstruction via Multichannel Singular Spectrum Analysis (MSSA),
    Geophysics 76(3)

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 50)
    >>> filtered, sing = mssa_2d(data, dt=0.004, P=10, flow=5, fhigh=60, meth=1)
    """
    nt, nx = DATA.shape
    nf = 2 * (2 ** _nextpow2(nt))

    DATA_FX_f = np.zeros((nf, nx), dtype=complex)
    sing = []

    # First and last samples of the DFT
    ilow = int(np.floor(flow * dt * nf)) + 1
    if ilow < 1:
        ilow = 1

    ihigh = int(np.floor(fhigh * dt * nf)) + 1
    if ihigh > int(np.floor(nf / 2)) + 1:
        ihigh = int(np.floor(nf / 2)) + 1

    # Transform to FX
    DATA_FX_tmp = fft(DATA, n=nf, axis=0)
    samp = np.arange(1, nf + 1)
    f = (samp - 1) / (nf * dt)
    f = f[:ihigh]

    # Size of Hankel Matrix
    Lcol = int(np.floor(nx / 2)) + 1
    Lrow = nx - Lcol + 1

    # Form level-1 block Hankel matrix
    for j in range(ilow - 1, ihigh):
        M = np.zeros((Lrow, Lcol), dtype=complex)

        for lc in range(Lcol):
            M[:, lc] = DATA_FX_tmp[j, lc:lc + Lrow]

        # SVD decomposition with P largest singular values
        if meth == 1:
            # Standard SVD
            U, S, Vh = svds(M, k=min(10, min(M.shape) - 1))
            sing.append(S)
            Mout = U[:, :P] @ np.diag(S[:P]) @ Vh[:P, :]
        else:
            # Randomized SVD (RQRD)
            from ..rank_reduction import rqrd
            Mout = rqrd(M, P)

        # Sum along anti-diagonals to recover signal
        Count = np.zeros(nx)
        tmp2 = np.zeros(nx, dtype=complex)

        for ic in range(Lcol):
            for ir in range(Lrow):
                Count[ir + ic - 1] += 1
                tmp2[ir + ic - 1] += Mout[ir, ic]

        tmp2 = tmp2 / Count
        DATA_FX_f[j, :] = tmp2

    # Honor symmetries
    for k in range(nf // 2 + 1, nf):
        DATA_FX_f[k, :] = np.conj(DATA_FX_f[nf - k + 1, :])

    # Back to TX (the output)
    DATA_f = np.real(ifft(DATA_FX_f, axis=0))
    DATA_f = DATA_f[:nt, :]

    return DATA_f, np.array(sing) if sing else np.array([])
