"""
3D MSSA with interpolation for seismic data denoising and reconstruction.

转换自 MATLAB: codes/mssa/mssa_3d_interp.m
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.sparse.linalg import svds
from typing import Callable


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(2 ** np.ceil(np.log2(n)))


def mssa_3d_interp(T: Callable, DATA: np.ndarray, dt: float, P: int,
                   flow: float, fhigh: float, meth: int = 1) -> np.ndarray:
    """
    3D MSSA with interpolation for seismic data.

    Parameters
    ----------
    T : callable
        Spatial sampling operator (2D in this case)
    DATA : np.ndarray
        3D array data (nt x nx x ny)
    dt : float
        Sampling interval in seconds
    P : int
        Size of subspace for noise attenuation (desired rank)
    flow : float
        Minimum frequency in Hz
    fhigh : float
        Maximum frequency in Hz
    meth : int, optional
        1 for standard SVD, 2 for Randomized SVD (default: 1)

    Returns
    -------
    DATA_f : np.ndarray
        Filtered and reconstructed data

    Notes
    -----
    This method applies MSSA to interpolated 3D seismic data for simultaneous
    denoising and reconstruction.

    Reference: Oropeza and Sacchi, 2011

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 50, 30)
    >>> def T(x): return x  # Identity operator
    >>> filtered = mssa_3d_interp(T, data, dt=0.004, P=10, flow=5, fhigh=60)
    """
    nt, nx, ny = DATA.shape
    nf = 2 * (2 ** _nextpow2(nt))

    DATA_FX_f = np.zeros((nf, nx, ny), dtype=complex)

    # First and last samples of the DFT
    ilow = int(np.floor(flow * dt * nf)) + 1
    if ilow < 1:
        ilow = 1

    ihigh = int(np.floor(fhigh * dt * nf)) + 1
    if ihigh > int(np.floor(nf / 2)) + 1:
        ihigh = int(np.floor(nf / 2)) + 1

    # Transform to FX
    DATA_FX_tmp = fft(DATA, n=nf, axis=0)

    # Size of the Hankel Matrix for y
    Ncol = int(np.floor(ny / 2)) + 1
    Nrow = ny - Ncol + 1

    # Size of Hankel Matrix of Hankel Matrices in x
    Lcol = int(np.floor(nx / 2)) + 1
    Lrow = nx - Lcol + 1

    # Form level-2 block Hankel matrix and apply sampling operator
    for j in range(ilow - 1, ihigh):
        M = np.zeros((Lrow * Nrow, Lcol * Ncol), dtype=complex)

        for lc in range(Lcol):
            for lr in range(Lrow):
                tmp_fx = np.squeeze(DATA_FX_tmp[j, lr + lc - 1, :]).T

                for ic in range(Ncol):
                    for ir in range(Nrow):
                        row_idx = (lr * Nrow) - Nrow + ir
                        col_idx = (lc * Ncol) - Ncol + ic
                        M[row_idx, col_idx] = tmp_fx[ir + ic - 1]

        # Apply sampling operator
        M_sampled = T(M)

        if j == ilow - 1:
            print(f' ---- Size of Block Hankel Matrix:   {M_sampled.shape[0]} x  {M_sampled.shape[1]}')

        # SVD decomposition with P largest singular values
        if meth == 1:
            U, S, Vh = svds(M_sampled, k=P)
            Mout = U @ np.diag(S) @ Vh
        else:
            from ..rank_reduction import rqrd
            Mout = rqrd(M_sampled, P)

        # Sum along anti-diagonals to recover signal
        Count = np.zeros((ny, nx))
        tmp = np.zeros((ny, nx), dtype=complex)
        tmp2 = np.zeros((ny, nx), dtype=complex)

        for lc in range(Lcol):
            for lr in range(Lrow):
                for ic in range(Ncol):
                    for ir in range(Nrow):
                        row_idx = (lr * Nrow) - Nrow + ir
                        col_idx = (lc * Ncol) - Ncol + ic
                        Count[ir + ic - 1, lr + lc - 1] += 1
                        tmp[ir + ic - 1, lr + lc - 1] += Mout[row_idx, col_idx]

                safe_count = np.where(Count[:, lr + lc - 1] == 0, 1, Count[:, lr + lc - 1])
                tmp2[:, lr + lc - 1] = tmp[:, lr + lc - 1] / safe_count
                DATA_FX_f[j, lr + lc - 1, :] = tmp2[:, lr + lc - 1].T

    # Honor symmetries
    for k in range(nf // 2 + 1, nf):
        DATA_FX_f[k, :, :] = np.conj(DATA_FX_f[nf - k + 1, :, :])

    # Back to TX
    DATA_f = np.real(ifft(DATA_FX_f, axis=0))
    DATA_f = DATA_f[:nt, :, :]

    return DATA_f
