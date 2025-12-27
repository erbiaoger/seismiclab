"""
Frequency-domain Radon transform operator.

转换自 MATLAB: codes/linear_operators/Operator_Radon_Freq.m
"""

from __future__ import annotations

import numpy as np
from typing import Union


def operator_radon_freq(In: np.ndarray, Param: dict, flag: int) -> np.ndarray:
    """
    Frequency-domain Radon transform operator (forward or adjoint).

    转换自 MATLAB: codes/linear_operators/Operator_Radon_Freq.m

    Parameters
    ----------
    In : np.ndarray
        Radon panel m(nt, ntau) if flag = 1 (Forward transform)
        Data d(nt, nh) if flag = -1 (Adjoint transform)
    Param : dict
        Parameters dictionary containing:
        - dt: sampling interval in sec
        - h: offsets or positions of traces in meters
        - p: ray parameters (np array)
        - nt: number of time samples in output
        - ntau: number of tau samples
        - flow: minimum frequency in Hz
        - fhigh: maximum frequency in Hz
        - N: 1 for linear tau-p, 2 for parabolic tau-p
        - Mutes: mute operator (nt, nh) mask
    flag : int
        1 for forward transform (m to d), -1 for adjoint (d to m)

    Returns
    -------
    Out : np.ndarray
        Data d(nt, nh) if flag = 1
        Radon panel m(ntau, np) if flag = -1

    Notes
    -----
    Reference: Hampson, D., 1986, Inverse velocity stacking for multiple
    elimination, Journal of the CSEG, vol 22, no 1., 44-55.

    Examples
    --------
    >>> import numpy as np
    >>> Param = {'dt': 0.004, 'h': np.arange(0, 1000, 20), 'p': np.array([0.1, 0.2, 0.3]),
    ...          'nt': 500, 'ntau': 500, 'flow': 5, 'fhigh': 50, 'N': 2,
    ...          'Mutes': np.ones((500, 50))}
    >>> m = np.random.randn(500, 3)
    >>> d = operator_radon_freq(m, Param, 1)  # Forward transform
    """
    dt = Param['dt']
    h = Param['h']
    p = Param['p']
    nt = Param['nt']
    ntau = Param['ntau']
    flow = Param['flow']
    fhigh = Param['fhigh']
    N = Param['N']
    Mutes = Param.get('Mutes', 1.0)

    nh = len(h)
    np_len = len(p)
    nfft = 2 * (2 ** _nextpow2(ntau))

    if flag == 1:
        m = In.copy()
    else:
        d = Mutes * In

    if N == 2:
        h = (h / np.max(np.abs(h))) ** 2

    ilow = int(np.floor(flow * dt * nfft)) + 1
    ihigh = int(np.floor(fhigh * dt * nfft)) + 1

    A = np.outer(h, p)

    if flag == 1:
        # Forward transform: m to d
        M = np.fft.fft(m, n=nfft, axis=0)
        D = np.zeros((nfft, nh), dtype=complex)

        for ifreq in range(ilow, ihigh + 1):
            f = 2.0 * np.pi * (ifreq - 1) / nfft / dt
            L = np.exp(1j * f * A)
            x = M[ifreq, :].reshape(-1, 1)
            y = L @ x
            D[ifreq, :] = y.ravel()

        # Hermitian symmetry
        for k in range(nfft // 2 + 1, nfft):
            D[k, :] = np.conj(D[nfft - k + 2, :])

        d = nfft * np.fft.ifft(D, axis=0)
        d = np.real(d[:nt, :])

        Out = Mutes * d / (nh * np_len)

    else:
        # Adjoint transform: d to m
        D = np.fft.fft(d, n=nfft, axis=0)
        M = np.zeros((nfft, np_len), dtype=complex)

        for ifreq in range(ilow, ihigh + 1):
            f = 2.0 * np.pi * (ifreq - 1) / nfft / dt
            L = np.exp(1j * f * A)
            y = D[ifreq, :].reshape(-1, 1)
            x = L.conj().T @ y
            M[ifreq, :] = x.ravel()

        # Hermitian symmetry
        for k in range(nfft // 2 + 1, nfft):
            M[k, :] = np.conj(M[nfft - k + 2, :])

        m = nfft * np.fft.ifft(M, axis=0)
        m = np.real(m[:ntau, :])

        Out = m / (nh * np_len)

    return Out


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(np.ceil(np.log2(n)))
