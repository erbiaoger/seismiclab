"""
Operator for Radon-Stolt transform (forward and adjoint).

转换自 MATLAB: codes/linear_operators/Operator_Radon_Stolt.m
"""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve, correlate
from typing import Tuple, Union


def _find_element_index(arr: np.ndarray, element: float) -> int:
    """
    Find the index of an element in an array.

    转换自 MATLAB: codes/linear_operators/Operator_Radon_Stolt.m 中的 Find_element_index 函数

    Parameters
    ----------
    arr : np.ndarray
        Input array to search
    element : float
        Element to find

    Returns
    -------
    index : int
        Index of the element (0-based), returns 0 if not found
    """
    for i in range(len(arr)):
        if arr[i] == element:
            return i
    return 0


def _truncating_padding(
    input_data: np.ndarray, PARAM: dict, transform: str
) -> np.ndarray:
    """
    Truncate or pad data to extend apexes range.

    转换自 MATLAB: codes/linear_operators/Operator_Radon_Stolt.m 中的 truncating_padding 函数

    Parameters
    ----------
    input_data : np.ndarray
        Input data matrix
    PARAM : dict
        Parameter dictionary containing:
        - shift_x: apex shift positions
        - receivers: receiver positions
    transform : str
        'f' for forward (truncate), 'a' for adjoint (pad)

    Returns
    -------
    output : np.ndarray
        Truncated or padded data
    """
    if transform == 'f':  # Forward --> truncate
        nt, _ = input_data.shape
        nreceivers = len(PARAM['receivers'])
        output = np.zeros((nt, nreceivers), dtype=float)

        for iR in range(nreceivers):
            index = _find_element_index(PARAM['shift_x'], PARAM['receivers'][iR])
            if index > 0:
                output[:, iR] = input_data[:, index]

    elif transform == 'a':  # Adjoint --> pad
        nt, _ = input_data.shape
        nshift = len(PARAM['shift_x'])
        output = np.zeros((nt, nshift), dtype=float)

        for iSh in range(nshift):
            index = _find_element_index(PARAM['receivers'], PARAM['shift_x'][iSh])
            if index > 0:
                output[:, iSh] = input_data[:, index]

    return output


def _conv_xcorr(
    x: np.ndarray, w: np.ndarray, transform: str
) -> np.ndarray:
    """
    Convolve or cross-correlate with wavelet.

    转换自 MATLAB: codes/linear_operators/Operator_Radon_Stolt.m 中的 conv_xcorr 函数

    Parameters
    ----------
    x : np.ndarray
        Input data (nt, nx)
    w : np.ndarray
        Wavelet
    transform : str
        'f' for forward (convolve), 'a' for adjoint (cross-correlate)

    Returns
    -------
    out : np.ndarray
        Convolved or cross-correlated data
    """
    nt, nx = x.shape
    out = np.zeros((nt, nx), dtype=float)

    if transform == 'f':
        for ix in range(nx):
            tr = convolve(x[:, ix], w, mode='full')
            out[1:nt, ix] = tr[1:nt]

    elif transform == 'a':
        for ix in range(nx):
            tr = correlate(x[:, ix], w, mode='full')
            out[1:nt, ix] = tr[nt - 1:]

    return out


def operator_radon_stolt(
    in_data: np.ndarray, PARAM: dict, transform: str
) -> np.ndarray:
    """
    Radon-Stolt operator (forward or adjoint).

    转换自 MATLAB: codes/linear_operators/Operator_Radon_Stolt.m

    Parameters
    ----------
    in_data : np.ndarray
        Input data:
        - For adjoint: d(t, h) with shape (nt, nx)
        - For forward: m(tau, v, x) with shape (nt, nv, nx)
    PARAM : dict
        Parameter dictionary containing:
        - time: time vector (1D array)
        - receivers: x-axis vector (receiver positions)
        - v: velocities vector (1D array)
        - shift_x: shifted apexes location vector (1D array)
        - tpad: zero padding ratio for time
        - xpad: zero padding ratio for x-axis
        - freq_cut: cutoff frequency
        - use_w: 'yes' or 'no' - whether to use wavelet
        - w: wavelet (if use_w='yes')
    transform : str
        'f' for forward transform, 'a' for adjoint transform

    Returns
    -------
    out : np.ndarray
        Output data:
        - For adjoint: m(tau, v, x) with shape (nt, nv, nx)
        - For forward: d(t, h) with shape (nt, nx)

    Notes
    -----
    The Stolt migration operator performs a mapping from frequency-wavenumber
    domain using the dispersion relation:
    freq_map^2 = kx^2 * v^2 + freq^2

    Examples
    --------
    >>> import numpy as np
    >>> # Setup parameters
    >>> dt = 0.002
    >>> t = np.arange(0, 1.0, dt)
    >>> h = np.arange(-1000, 1000, 20)
    >>> v = np.arange(1500, 3500, 100)
    >>> PARAM = {
    ...     'time': t,
    ...     'receivers': h,
    ...     'v': v,
    ...     'shift_x': h,
    ...     'tpad': 0.5,
    ...     'xpad': 0.5,
    ...     'freq_cut': 80.0,
    ...     'use_w': 'no'
    ... }
    >>> # Forward transform
    >>> m = np.random.randn(len(t), len(v), len(h))
    >>> d = operator_radon_stolt(m, PARAM, 'f')
    """
    # Extract parameters
    time = PARAM['time']
    nt = len(time)
    dt_val = (time[-1] - time[0]) / (nt - 1)

    nx = len(PARAM['shift_x'])
    nR = len(PARAM['receivers'])
    dx = (PARAM['shift_x'][-1] - PARAM['shift_x'][0]) / (nx - 1)

    nv = len(PARAM['v'])

    # Cross-correlate with wavelet (if adjoint)
    if transform == 'a':
        if PARAM.get('use_w', 'no') == 'yes':
            in_data = _conv_xcorr(in_data, PARAM['w'], transform)

    # Extend apexes range (adjoint)
    if transform == 'a':
        in_data = _truncating_padding(in_data, PARAM, transform)

    # Zero padding
    tmax = time[-1] - time[0]
    tpad = PARAM['tpad'] * tmax
    nt_pad = int(round((tmax + tpad) / dt_val + 1))

    xmax = PARAM['shift_x'][-1] - PARAM['shift_x'][0]
    xpad = PARAM['tpad'] * xmax  # Note: using tpad for both dimensions in original
    nx_pad = int(round((xmax + xpad) / dx + 1))

    if transform == 'a':
        # Pad input data
        in_padded = np.zeros((nt_pad, nx_pad), dtype=complex)
        in_padded[:nt, :nx] = in_data
    elif transform == 'f':
        # Pad input data
        in_padded = np.zeros((nt_pad, nv, nx_pad), dtype=complex)
        in_padded[:nt, :, :nx] = in_data

    # FFT sizes
    nf = 2 ** int(np.ceil(np.log2(nt_pad)))
    nkx = 2 ** int(np.ceil(np.log2(nx_pad)))

    # Frequency and wavenumber axes
    ifreq = np.arange(1, nf + 1)
    ikx = np.arange(1, nkx + 1)

    # Wrap frequencies
    nf2 = nf // 2 + 2
    ifreq2 = ifreq - 1 - nf * np.floor(ifreq / nf2)
    freq = ifreq2 / nf / dt_val

    nkx2 = nkx // 2 + 2
    ikx2 = ikx - 1 - nkx * np.floor(ikx / nkx2)
    kx = ikx2 / nkx / dx

    # Cutoff frequency
    dfreq = 1 / nf / dt_val
    ifreq_cut = int(round(PARAM['freq_cut'] / dfreq)) + 1

    # Forward FFT
    if transform == 'a':  # Adjoint operator m(tau,v,x) = L' d(t,h)
        in_fk = np.fft.fft2(in_padded, s=(nf, nkx))
        out_fk = np.zeros((nf // 2 + 1, nv, nkx), dtype=complex)
        out_fx = np.zeros((nf, nv, nkx), dtype=complex)
        out = np.zeros((nt, nv, nx), dtype=float)

    elif transform == 'f':  # Forward operator d(t,h) = L m(tau,v,x)
        in_fk = np.zeros((nf, nv, nkx), dtype=complex)
        for iv in range(nv):
            in_fk[:, iv, :] = np.fft.fft2(
                np.squeeze(in_padded[:, iv, :]), s=(nf, nkx)
            )
        out_fk = np.zeros((nf // 2 + 1, nkx), dtype=complex)
        out_fx = np.zeros((nf, nkx), dtype=complex)
        out = np.zeros((nt, nx), dtype=float)

    # Stolt mapping
    for iv in range(nv):
        for ikx_idx in range(nkx):
            for ifreq_tau in range(ifreq_cut - 1):  # 0-indexed
                freq_map2 = kx[ikx_idx] ** 2 * PARAM['v'][iv] ** 2 + freq[ifreq_tau] ** 2
                freq_map = np.sqrt(freq_map2)

                ifreq_map = int(np.ceil(freq_map / dfreq)) + 1  # Mapped freq index

                if freq_map2 != 0:
                    scale = freq[ifreq_tau] / freq_map
                else:
                    scale = 1.0

                if 0 < ifreq_map < ifreq_cut:
                    if transform == 'a':  # adjoint m = L' d
                        out_fk[ifreq_tau, iv, ikx_idx] += (
                            in_fk[ifreq_map - 1, ikx_idx] * scale
                        )
                    elif transform == 'f':  # forward d = L m
                        out_fk[ifreq_map - 1, ikx_idx] += (
                            in_fk[ifreq_tau, iv, ikx_idx] * scale
                        )

    # Inverse FFT
    # Compute negative frequencies using symmetry
    if transform == 'a':
        for iv in range(nv):
            out_fx[: nf // 2 + 1, iv, :] = np.fft.ifft(
                np.squeeze(out_fk[:, iv, :]), axis=0
            )
            out_fx[nf // 2 + 1 :, iv, :] = np.conj(
                np.flipud(np.squeeze(out_fx[1: nf // 2, iv, :]))
            )

    elif transform == 'f':
        out_fx[: nf // 2 + 1, :] = np.fft.ifft(out_fk, axis=0)
        out_fx[nf // 2 + 1 :, :] = np.conj(np.flipud(out_fx[1: nf // 2, :]))

    # Inverse Fourier Transform and remove zero pads
    if transform == 'a':  # adjoint m = L' d
        for iv in range(nv):
            temp = np.fft.ifft(np.squeeze(out_fx[:, iv, :]), axis=0)
            out[:, iv, :] = np.real(temp[:nt, :nx])

    elif transform == 'f':  # forward d = L m
        temp = np.fft.ifft(out_fx, axis=0)
        out = np.real(temp[:nt, :nx])

    # Remove extended offsets (forward)
    if transform == 'f':
        out = _truncating_padding(out, PARAM, transform)

    # Convolve with wavelet (forward)
    if transform == 'f':
        if PARAM.get('use_w', 'no') == 'yes':
            out = _conv_xcorr(out, PARAM['w'], transform)

    return out
