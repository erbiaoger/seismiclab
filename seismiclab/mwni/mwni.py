"""
Multi-window Noise Inversion (MWNI) for seismic data interpolation.

转换自 MATLAB: codes/mwni/operator_nfft.m, mwni_irls.m, mwni.m
"""

import numpy as np
from scipy.fft import fft, ifft, fftn, ifftn
from typing import Union, List


def cgdot(x: np.ndarray, y: np.ndarray) -> float:
    """Dot product for complex arrays."""
    return np.real(np.sum(x * np.conj(y)))


def operator_nfft(input_data: np.ndarray, T: np.ndarray, NDim: int,
                   N: Union[int, List[int]], K: Union[int, List[int]],
                   option: int) -> np.ndarray:
    """
    N-D Fourier operator via FFTs (forward and adjoint).

    Parameters
    ----------
    input_data : np.ndarray
        Input data
    T : np.ndarray
        Sampling operator (1 for alive, 0 for dead traces)
    NDim : int
        Number of dimensions (2, 3, 4, or 5)
    N : int or list
        Number of samples in each spatial dimension
    K : int or list
        Number of wavenumbers in each direction
    option : int
        -1 for adjoint (space to wavenumber), 1 for forward (wavenumber to space)

    Returns
    -------
    output : np.ndarray
        Transformed data

    Notes
    -----
    Reference: Bin Liu, Mauricio D. Sacchi, 2004, Minimum weighted norm
    interpolation of seismic records, GEOPHYSICS, Vol. 69, No. 6
    """
    # Ensure K is a list
    if isinstance(K, int):
        K = [K]
    if isinstance(N, int):
        N = [N]

    if NDim == 2:
        c = K[0]
        if option == -1:
            # Adjoint: space to wavenumber
            aux = T * input_data
            output = fft(aux, K[0])
        else:
            # Forward: wavenumber to space
            aux = c * ifft(input_data)
            output = T * aux[:N[0]]

    elif NDim == 3:
        c = K[0] * K[1]
        if option == -1:
            aux = T * input_data
            output = fftn(aux, [K[0], K[1]])
        else:
            aux = c * ifftn(input_data)
            output = T * aux[:N[0], :N[1]]

    elif NDim == 4:
        c = K[0] * K[1] * K[2]
        if option == -1:
            aux = T * input_data
            output = fftn(aux, [K[0], K[1], K[2]])
        else:
            aux = c * ifftn(input_data)
            output = T * aux[:N[0], :N[1], :N[2]]

    elif NDim == 5:
        c = K[0] * K[1] * K[2] * K[3]
        if option == -1:
            aux = T * input_data
            output = fftn(aux, [K[0], K[1], K[2], K[3]])
        else:
            aux = c * ifftn(input_data)
            output = T * aux[:N[0], :N[1], :N[2], :N[3]]

    return output


def mwni_irls(d: np.ndarray, m0: np.ndarray, T: np.ndarray, NDim: int,
              N: Union[int, List[int]], K: Union[int, List[int]],
              itmax_internal: int, itmax_external: int,
              silence: int = 1) -> tuple:
    """
    MWNI via IRLS (Iteratively Reweighted Least Squares).

    Minimizes ||A x - d||_2 with x sparse, where x are Fourier coefficients
    and d is spatial data.

    Parameters
    ----------
    d : np.ndarray
        Complex spatial data
    m0 : np.ndarray
        Initial model
    T : np.ndarray
        Sampling operator
    NDim : int
        Number of dimensions
    N : int or list
        Number of samples in each dimension
    K : int or list
        Number of wavenumbers
    itmax_internal : int
        Max internal iterations
    itmax_external : int
        Max external iterations
    silence : int, optional
        1 for silent, 0 for verbose (default: 1)

    Returns
    -------
    x : np.ndarray
        Inverted Fourier coefficients
    misfit : np.ndarray
        Misfit vs iteration

    Notes
    -----
    Reference: Bin Liu, Mauricio D. Sacchi, 2004
    """
    z = np.zeros_like(m0)
    P = np.ones_like(z)
    kc = 0
    misfit_list = []
    x = z.copy()

    for l in range(itmax_external):
        # Forward operator
        di = operator_nfft(P * z, T, NDim, N, K, 1)

        # Residual
        r = d - di

        # Adjoint operator
        g = operator_nfft(r, T, NDim, N, K, -1)
        g = g * P
        s = g.copy()
        gammam = cgdot(g, g)

        k = 1
        while k <= itmax_internal:
            # Compute step size
            ss = operator_nfft(P * s, T, NDim, N, K, 1)
            den = cgdot(ss, ss)
            alpha = gammam / (den + 1e-8)

            # Update model
            z = z + alpha * s
            r = r - alpha * ss

            # Record misfit
            misfit_list.append(cgodot(r, r))

            # New gradient
            g = operator_nfft(r, T, NDim, N, K, -1)
            g = g * P
            gamma = cgdot(g, g)

            # Polak-Ribiere beta
            beta = gamma / (gammam + 1e-7)
            gammam = gamma
            s = g + beta * s

            if silence == 0:
                print(f'Iteration = {k} Misfit={misfit_list[-1]:.5g}')

            k += 1
            kc += 1

        # Update weights
        x = P * z
        y = x / np.max(np.abs(x))
        P = np.abs(y) + 0.001

    return x, np.array(misfit_list)


def mwni(d: np.ndarray, T: np.ndarray, NDim: int,
         N: Union[int, List[int]], K: Union[int, List[int]],
         mu: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Multi-Window Noise Inversion for seismic data interpolation.

    Parameters
    ----------
    d : np.ndarray
        Input spatial data with missing traces
    T : np.ndarray
        Sampling operator (1 for alive, 0 for missing)
    NDim : int
        Number of dimensions (2, 3, 4, or 5)
    N : int or list
        Number of samples in each spatial dimension
    K : int or list
        Number of wavenumbers in each direction
    mu : float, optional
        Regularization parameter (default: 0.01)
    niter : int, optional
        Number of iterations (default: 10)

    Returns
    -------
    x : np.ndarray
        Interpolated data

    Notes
    -----
    This is a simplified interface to mwni_irls.

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
    >>> T = np.ones_like(d)
    >>> T[:, ::2] = 0  # 50% missing traces
    >>> interpolated = mwni(d, T, NDim=2, N=[100, 50], K=[128, 64])
    """
    m0 = np.zeros_like(d)
    itmax_internal = 10
    itmax_external = niter

    x, misfit = mwni_irls(d, m0, T, NDim, N, K,
                          itmax_internal, itmax_external, silence=1)

    return x
