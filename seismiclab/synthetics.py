from __future__ import annotations

import numpy as np


def ricker(f: float, dt: float):
    nw = int(2.5 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2
    k = np.arange(1, nw + 1)
    alpha = (nc - k + 1) * f * dt * np.pi
    beta = alpha**2
    w = (1.0 - 2.0 * beta) * np.exp(-beta)
    tw = -(nc + 1 - np.arange(1, nw + 1)) * dt
    return w.astype(float), tw.astype(float)


def linear_events(dt=0.004, f0=20.0, tmax=1.0, h=None, tau=None, p=None, amp=None):
    if h is None:
        h = np.arange(0, 5 * (55 - 1) + 0.1, 5.0)
    if tau is None:
        tau = [0.8, 0.4, 0.33, 0.3]
    if p is None:
        p = [0.00031, -0.0003, 0.0001, 0.0]
    if amp is None:
        amp = [1.2, -1.0, 1.0, 0.10]

    h = np.abs(np.asarray(h, dtype=float))
    tau = np.asarray(tau, dtype=float)
    p = np.asarray(p, dtype=float)
    amp = np.asarray(amp, dtype=float)
    nt = int(np.floor(tmax / dt)) + 1
    nfft = 4 * (2 ** int(np.ceil(np.log2(nt))))
    nh = len(h)
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    D = np.zeros((nfft, nh), dtype=complex)
    delay = dt * (nw - 1) / 2

    for ifreq in range(nfft // 2 + 1):
        w = 2.0 * np.pi * ifreq / nfft / dt
        shift = np.exp(-1j * w * (tau[:, None] + h[None, :] * p[:, None] - delay))
        D[ifreq, :] += np.sum(amp[:, None] * W[ifreq] * shift, axis=0)
    D[1 : nfft // 2] = np.conj(np.flipud(D[1 : nfft // 2]))
    d = np.fft.ifft(D, axis=0).real[:nt, :]
    t = np.arange(nt) * dt
    return d, h, t


def data_cube(N, dt, f0, nt, dx, t0, p, A, opt="linear"):
    N = np.atleast_1d(N).astype(int)
    dx = np.atleast_1d(dx).astype(float)
    t0 = np.asarray(t0, dtype=float)
    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    ND = len(N)
    if ND not in (1, 2):
        raise NotImplementedError("data_cube translation currently supports 1D/2D cubes")

    nfft = 2 * (2 ** int(np.ceil(np.log2(nt))))
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    tdelay = nw * dt / 2

    if ND == 1:
        n1 = N[0]
        x1 = np.arange(n1) * dx[0]
        if opt == "parabolic":
            x1 = (x1 / np.max(x1)) ** 2
        ne = p.shape[1] if p.ndim > 1 else len(p)
        D = np.zeros((nfft, n1), dtype=complex)
        for iw in range(1, nfft // 2):
            w = 2 * np.pi * iw / (nfft * dt)
            T = np.zeros(n1, dtype=complex)
            for ie in range(ne):
                x11 = np.exp(-1j * w * x1 * p[0, ie] if p.ndim > 1 else -1j * w * x1 * p[ie])
                T += A[ie] * np.exp(-1j * w * (t0[ie] - tdelay)) * W[iw] * x11
            D[iw, :] = T
            D[nfft - iw, :] = np.conj(T)
        d = np.fft.ifft(D, axis=0).real[:nt, :]
        return d

    # ND == 2
    n1, n2 = N
    x1 = np.arange(n1) * dx[0]
    x2 = np.arange(n2) * dx[1]
    if opt == "parabolic":
        x1 = (x1 / np.max(x1)) ** 2
        x2 = (x2 / np.max(x2)) ** 2
    ne = p.shape[1]
    D = np.zeros((nfft, n1, n2), dtype=complex)

    for iw in range(1, nfft // 2):
        w = 2 * np.pi * iw / (nfft * dt)
        T = np.zeros((n1, n2), dtype=complex)
        for ie in range(ne):
            x11 = np.exp(-1j * w * x1 * p[0, ie])
            x22 = np.exp(-1j * w * x2 * p[1, ie])
            M1, M2 = np.meshgrid(x11, x22, indexing="ij")
            T += A[ie] * np.exp(-1j * w * (t0[ie] - tdelay)) * W[iw] * M1 * M2
        D[iw, :, :] = T
        D[nfft - iw, :, :] = np.conj(T)

    d = np.fft.ifft(D, axis=0).real[:nt, :, :]
    return d


def laplace(N: int, lambd: float) -> np.ndarray:
    y = np.random.rand(N)
    x = np.empty(N, dtype=float)
    for idx in range(N):
        if y[idx] <= 0.5:
            x[idx] = lambd * np.log(2 * y[idx])
        else:
            x[idx] = -lambd * np.log(2 * (1 - y[idx]))
    return x


def laplace_mixture(N: int, arg) -> np.ndarray:
    lambda1, lambda2, p = arg
    x1 = laplace(N, lambda1)
    x2 = laplace(N, lambda2)
    mix = np.empty(N, dtype=float)
    rnd = np.random.rand(N)
    mix[rnd < p] = x1[rnd < p]
    mix[rnd >= p] = x2[rnd >= p]
    return mix


def gauss_mixture(N: int, arg) -> np.ndarray:
    """
    Compute a reflectivity using a Gaussian Mixture model.

    转换自 MATLAB: codes/synthetics/gauss_mixture.m

    Parameters
    ----------
    N : int
        Number of samples
    arg : list or array
        [sigma1, sigma2, p] where:
        - sigma1: variance of distribution with probability p
        - sigma2: variance of distribution with probability 1-p
        - p: mixing parameter (0, 1)

    Returns
    -------
    r : np.ndarray
        Reflectivity series

    Examples
    --------
    >>> import numpy as np
    >>> r = gauss_mixture(1000, [0.01, 0.001, 0.7])
    """
    sigma1, sigma2, p = arg
    r1 = np.random.randn(N) * np.sqrt(sigma1)
    r2 = np.random.randn(N) * np.sqrt(sigma2)

    r = np.empty(N, dtype=float)
    rnd = np.random.rand(N)
    mask = rnd < p
    r[mask] = r1[mask]
    r[~mask] = r2[~mask]
    return r


def bernoulli_refl(N: int, lambda_: float, sigma: float) -> np.ndarray:
    """
    Random numbers with Bernoulli distribution.

    转换自 MATLAB: codes/synthetics/bernoulli_refl.m

    Parameters
    ----------
    N : int
        Length of series
    lambda_ : float
        Occurrence of a non-zero sample (0, 1)
    sigma : float
        Standard error for non-zero samples

    Returns
    -------
    r : np.ndarray
        Random numbers (series)

    Examples
    --------
    >>> import numpy as np
    >>> r = bernoulli_refl(1000, 0.8, 0.1)
    """
    r = np.zeros(N, dtype=float)

    for k in range(N):
        if np.random.rand() > lambda_:
            r[k] = sigma * np.random.randn()

    return r


def flat_events(snr: float = 10.0) -> tuple:
    """
    Generate data containing flat events with noise.

    转换自 MATLAB: codes/synthetics/flat_events.m

    Parameters
    ----------
    snr : float, optional
        Signal-to-noise ratio (default: 10.0)

    Returns
    -------
    d : np.ndarray
        Superposition of flat events + noise
    h : np.ndarray
        Offset axis
    t : np.ndarray
        Time axis

    Examples
    --------
    >>> d, h, t = flat_events(snr=5.0)
    """
    from scipy.signal import convolve2d, windows

    dt = 2.0 / 1000
    tmax = 0.8
    h = np.arange(0, 10 * (90 - 1) + 0.1, 10.0)
    tau = np.array([0.1, 0.2, 0.3, 0.6])
    p = np.array([0.0, -0.0, 0.0, -0.0])
    amp = np.array([1.2, -1.0, 1.0, 1.0])
    f0 = 20
    L = 5

    nt = int(np.floor(tmax / dt)) + 1
    nfft = 4 * (2 ** _nextpow2(nt))
    n_events = len(tau)
    nh = len(h)
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    D = np.zeros((nfft, nh), dtype=complex)

    delay = dt * (np.floor(nw / 2) + 1)

    for ifreq in range(1, nfft // 2 + 1):
        w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
        for k in range(n_events):
            Shift = np.exp(-1j * w * (tau[k] + h * p[k] - delay))
            D[ifreq, :] += amp[k] * W[ifreq] * Shift

    # w-domain symmetries
    for ifreq in range(1, nfft // 2):
        D[nfft + 1 - ifreq, :] = np.conj(D[ifreq, :])

    d = np.fft.ifft(D, axis=0)
    d = np.real(d[:nt, :])

    # My definition of snr = (Max Amp of Clean Data)/(Max Amp of Noise)
    dmax = np.max(np.abs(d))
    op = windows.hamming(L)
    Noise = convolve2d(np.random.randn(*d.shape), op[:, None], mode="same")

    Noisemax = np.max(np.abs(Noise))

    d = d + Noise * (dmax / Noisemax) / snr
    t = np.arange(nt) * dt

    return d, h, t


def hyperbolic_events(dt=0.004, f0=20.0, tmax=1.2, h=None, tau=None, v=None, amp=None):
    """
    Generate data containing hyperbolic events.

    转换自 MATLAB: codes/synthetics/hyperbolic_events.m

    Parameters
    ----------
    dt : float, optional
        Sampling interval in secs (default: 0.004)
    f0 : float, optional
        Central frequency of Ricker wavelet in Hz (default: 20.0)
    tmax : float, optional
        Maximum time of the simulation in secs (default: 1.2)
    h : np.ndarray, optional
        Vector of offsets in meters
    tau : np.ndarray, optional
        Vector of intercept times in secs
    v : np.ndarray, optional
        Vector of RMS velocities in m/s
    amp : np.ndarray, optional
        Vector of amplitudes

    Returns
    -------
    d : np.ndarray
        Data with hyperbolic moveout events
    h : np.ndarray
        Offset axis
    t : np.ndarray
        Time axis

    Examples
    --------
    >>> d, h, t = hyperbolic_events()
    """
    if h is None:
        h = np.arange(20, 1000 + 0.1, 20.0)
    if tau is None:
        tau = np.array([0.3, 0.5, 0.8])
    if v is None:
        v = np.array([2000, 2500, 3000])
    if amp is None:
        amp = np.array([1.0, -1.0, 1.0])

    nt = int(np.floor(tmax / dt)) + 1
    nfft = 4 * (2 ** _nextpow2(nt))
    n_events = len(tau)
    nh = len(h)
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    D = np.zeros((nfft, nh), dtype=complex)

    # Important: delay to have maximum of Ricker wavelet at right intercept time
    delay = dt * (np.floor(nw / 2) + 1)

    for ifreq in range(1, nfft // 2 + 1):
        w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
        for k in range(n_events):
            Shift = np.exp(-1j * w * (np.sqrt(tau[k] ** 2 + (h / v[k]) ** 2) - delay))
            D[ifreq, :] += amp[k] * W[ifreq] * Shift

    # Apply w-domain symmetries
    for ifreq in range(1, nfft // 2):
        D[nfft + 1 - ifreq, :] = np.conj(D[ifreq, :])

    d = np.fft.ifft(D, axis=0)
    d = np.real(d[:nt, :])
    t = np.arange(nt) * dt

    return d, h, t


def parabolic_events(dt=0.002, f0=20.0, tmax=1.2, h=None, tau=None, q=None, amp=None):
    """
    Generate t-x data containing parabolic events.

    转换自 MATLAB: codes/synthetics/parabolic_events.m

    Parameters
    ----------
    dt : float, optional
        Sampling interval in secs (default: 0.002)
    f0 : float, optional
        Central frequency of Ricker wavelet in Hz (default: 20.0)
    tmax : float, optional
        Maximum time of the simulation in secs (default: 1.2)
    h : np.ndarray, optional
        Vector of offsets in meters
    tau : np.ndarray, optional
        Vector of intercept times in secs
    q : np.ndarray, optional
        Vector of residual moveout at far offset in secs
    amp : np.ndarray, optional
        Vector of amplitudes

    Returns
    -------
    d : np.ndarray
        Data with parabolic moveout events
    h : np.ndarray
        Offset axis
    t : np.ndarray
        Time axis

    Examples
    --------
    >>> d, h, t = parabolic_events()
    """
    if h is None:
        h = np.arange(0, 1000 + 0.1, 20.0)
    if tau is None:
        tau = np.array([0.2, 0.5, 0.35, 0.35, 0.65, 0.75])
    if q is None:
        q = np.array([0.1, 0.3, 0.5, -0.3, -0.4, -0.7])
    if amp is None:
        amp = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    hmax = np.max(np.abs(h))
    h = h / hmax

    nt = int(np.floor(tmax / dt)) + 1
    nfft = 4 * (2 ** _nextpow2(nt))
    n_events = len(tau)
    nh = len(h)
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    D = np.zeros((nfft, nh), dtype=complex)

    delay = dt * (np.floor(nw / 2) + 1)

    for ifreq in range(1, nfft // 2 + 1):
        w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
        for k in range(n_events):
            Shift = np.exp(-1j * w * ((tau[k] + q[k] * h ** 2) - delay))
            D[ifreq, :] += amp[k] * W[ifreq] * Shift

    # Apply w-domain symmetries
    for ifreq in range(1, nfft // 2):
        D[nfft + 1 - ifreq, :] = np.conj(D[ifreq, :])

    d = np.fft.ifft(D, axis=0)
    d = np.real(d[:nt, :])
    t = np.arange(nt) * dt

    return d, h, t


def trapezoidal_wavelet(dt: float, f1: float, f2: float, f3: float, f4: float, c: float) -> tuple:
    """
    Compute a band-pass wavelet with trapezoidal amplitude spectrum and phase rotation.

    转换自 MATLAB: codes/synthetics/trapezoidal_wavelet.m

    Parameters
    ----------
    dt : float
        Sampling interval in sec
    f1 : float
        Low-cut frequency in Hz
    f2 : float
        Low-pass frequency in Hz
    f3 : float
        High-pass frequency in Hz
    f4 : float
        High-cut frequency in Hz
    c : float
        Phase rotation in degrees

    Returns
    -------
    w : np.ndarray
        Wavelet
    tw : np.ndarray
        Time axis in secs

    Notes
    -----
    Amplitude spectrum shape:
       ___________
      /           \\
     /             \\
    /               \\
    ------------------------>
       f1 f2        f3 f4

    Examples
    --------
    >>> w, tw = trapezoidal_wavelet(0.004, 5, 10, 40, 50, 0)
    """
    fc = (f3 - f2) / 2
    L = int(np.floor(1.5 / (fc * dt)))
    nt = 2 * L + 1
    k = _nextpow2(nt)
    nf = 4 * (2 ** k)

    i1 = int(np.floor(nf * f1 * dt)) + 1
    i2 = int(np.floor(nf * f2 * dt)) + 1
    i3 = int(np.floor(nf * f3 * dt)) + 1
    i4 = int(np.floor(nf * f4 * dt)) + 1

    up = np.arange(1, i2 - i1 + 1) / (i2 - i1)
    down = np.arange(i4 - i3, 0, -1) / (i4 - i3)

    aux = np.zeros(nf // 2 + 1)
    aux[i1:i2] = up
    aux[i2:i3] = 1.0
    aux[i3:i4] = down
    # aux[i4:] remains 0

    aux2 = np.flipud(aux[1:nf // 2])

    F = np.concatenate([aux, aux2])
    Phase = np.zeros(nf)
    Phase[1:nf // 2] = -c
    Phase[nf // 2 + 1:] = c

    Transfer = F * np.exp(-1j * Phase * np.pi / 180.0)
    temp = np.fft.fftshift(np.fft.ifft(Transfer))
    temp = np.real(temp)

    w = temp[nf // 2 + 1 - L : nf // 2 + 2 + L]
    nw = len(w)
    w = w * windows.hamming(nw)

    tw = np.arange(-L, L + 1) * dt

    return w.astype(float), tw.astype(float)


def rotated_wavelet(dt: float, fl: float, fh: float, c: float) -> tuple:
    """
    Band-limited wavelet with phase rotation (box-car amplitude spectrum).

    转换自 MATLAB: codes/synthetics/rotated_wavelet.m

    Parameters
    ----------
    dt : float
        Sampling interval in sec
    fl : float
        Minimum frequency in Hz
    fh : float
        Maximum frequency in Hz
    c : float
        Rotation in degrees

    Returns
    -------
    w : np.ndarray
        Wavelet
    t : np.ndarray
        Time axis in secs
    A : np.ndarray
        Amplitude spectrum
    P : np.ndarray
        Phase spectrum in degrees
    freq : np.ndarray
        Frequency axis in Hz

    Notes
    -----
    For a wavelet with trapezoidal amplitude spectrum, see trapezoidal_wavelet.

    Examples
    --------
    >>> w, t, A, P, freq = rotated_wavelet(0.004, 10, 40, 30)
    """
    # Define expected length of the wavelet
    fc = (fh - fl) / 2
    L = 4 * int(np.floor(2.5 / fc / dt))

    t = dt * np.arange(-L, L + 1)
    B = 2 * np.pi * fh
    b = 2 * np.pi * fl
    c_rad = c * np.pi / 180.0

    w = np.sin(c_rad + B * t) - np.sin(c_rad + b * t)

    # Avoid 0/0 (Use L'Hopital rule)
    I = t == 0
    t_old = t.copy()
    t[I] = 99999

    w = w / (t * np.pi)
    w[I] = (B / np.pi) * np.cos(c_rad) - (b / np.pi) * np.cos(c_rad)
    t[I] = 0.0

    # Normalize with dt to get unit amplitude spectrum
    # and smooth with a Hamming window
    w = dt * w * windows.hamming(len(w))

    nw = 2 * L + 1
    nh = L + 1

    nf = 4 * (2 ** _nextpow2(nw))

    # Take into account that the wavelet is non-causal
    # The following will remove linear phase shift and show the actual phase
    ww = np.concatenate([w[nh - 1 :], np.zeros(nf - nw), w[: nh - 1]])

    W = np.fft.fft(ww)
    M = len(W) // 2 + 1
    A = np.abs(W[:M])
    P = (180.0 / np.pi) * np.angle(W[:M])
    Kh = int(np.floor(fh * nf * dt)) + 1
    Kl = int(np.floor(fl * nf * dt)) + 1
    P[Kh:] = 0
    P[:Kl] = 0

    # Frequency axis in Hz
    freq = np.arange(M) / dt / nf

    return w.astype(float), t.astype(float), A.astype(float), P.astype(float), freq.astype(float)


def make_section(nt: int, nx: int, nJ: int, SNR: float, seed1: int, seed2: int, rho: float) -> tuple:
    """
    Make a synthetic data set with random reflectivity.

    转换自 MATLAB: codes/synthetics/make_section.m

    Parameters
    ----------
    nt : int
        Number of time samples
    nx : int
        Number of spatial positions (traces)
    nJ : int
        Number of reflecting events
    SNR : float
        Desired signal-to-noise ratio
    seed1 : int
        Random seed for noise generation
    seed2 : int
        Random seed for reflectivity generation
    rho : float
        Not used in current implementation (kept for API compatibility)

    Returns
    -------
    w : np.ndarray
        Wavelet
    r : np.ndarray
        Reflectivity model
    d0 : np.ndarray
        Clean data (no noise)
    d : np.ndarray
        Data corrupted by noise

    Examples
    --------
    >>> w, r, d0, d = make_section(500, 50, 20, 2.0, 42, 123, 0.1)
    """
    from scipy.signal import convolve2d, windows

    # Import phase_correction from dephasing module
    from .dephasing import phase_correction

    dt = 2.0 / 1000  # Sampling interval
    f0 = 40  # Central frequency of wavelet
    c = 50  # Wavelet constant phase rotation

    np.random.seed(seed1)
    # Note: MATLAB's rand('state', seed2) is equivalent to legacy random generator
    # For simplicity, we use numpy's default generator

    w = phase_correction(ricker(f0, dt)[0], c)

    r = np.zeros((nt, nx))

    J = 10 + np.floor(np.random.rand(nJ) * (nt - 10)).astype(int)
    a = np.random.rand(nJ) * 2 - 1

    r[J, 0] = a
    for k in range(1, nx):
        pert = np.round(np.random.rand(nJ) * 2 - 1).astype(int)
        J = J + pert
        r[J, k] = a + 0.06 * np.random.randn(nJ)

    d0 = convolve2d(r, w[:, None], mode="same")

    sed = np.std(d0)
    np.random.seed(seed2)
    noise = np.random.randn(*d0.shape)
    noise = convolve2d(noise, windows.hamming(3)[:, None], mode="same")
    sen = np.std(noise)

    num = (sed / sen) / SNR

    noise = noise * num

    d = d0 + noise

    return w.astype(float), r.astype(float), d0.astype(float), d.astype(float)


def bernoulli(N: int, lambda_: float, sigma: float) -> np.ndarray:
    """
    Random numbers with Bernoulli distribution (sparse series).

    转换自 MATLAB: codes/synthetics/bernoulli.m

    Parameters
    ----------
    N : int
        Length of series
    lambda_ : float
        Probability of zero sample (0, 1). Higher lambda = more zeros
    sigma : float
        Standard deviation for non-zero samples

    Returns
    -------
    r : np.ndarray
        Random sparse series

    Examples
    --------
    >>> import numpy as np
    >>> r = bernoulli(1000, 0.9, 0.1)
    """
    r = np.zeros(N, dtype=float)

    for k in range(N):
        if np.random.rand() > lambda_:
            r[k] = sigma * np.random.randn()

    return r


def hyperbolic_apex_shifted_events(
    dt=0.002,
    f0=20.0,
    tmax=1.2,
    h=None,
    h0=None,
    tau=None,
    v=None,
    amp=None,
):
    """
    Generate data containing hyperbolas with shifted apexes.

    转换自 MATLAB: codes/synthetics/hyperbolic_apex_shifted_events.m

    Parameters
    ----------
    dt : float, optional
        Sampling interval in secs (default: 0.002)
    f0 : float, optional
        Central frequency of Ricker wavelet in Hz (default: 20.0)
    tmax : float, optional
        Maximum time of the simulation in secs (default: 1.2)
    h : np.ndarray, optional
        Vector of offsets in meters
    h0 : np.ndarray, optional
        Vector of apex positions in meters
    tau : np.ndarray, optional
        Vector of intercept times in secs
    v : np.ndarray, optional
        Vector of RMS velocities in m/s
    amp : np.ndarray, optional
        Vector of amplitudes

    Returns
    -------
    d : np.ndarray
        Data with hyperbolic moveout and apex shifts
    h : np.ndarray
        Offset axis
    t : np.ndarray
        Time axis

    Examples
    --------
    >>> d, h, t = hyperbolic_apex_shifted_events()
    """
    if h is None:
        h = np.arange(-1500, 1200 + 0.1, 20.0)
    if tau is None:
        tau = np.array([0.1, 0.5, 0.8])
    if v is None:
        v = np.array([1500, 2400, 2300])
    if h0 is None:
        h0 = np.array([500, -300, 300])
    if amp is None:
        amp = np.array([1.0, -1.0, 1.0])

    nt = int(np.floor(tmax / dt)) + 1
    nfft = 4 * (2 ** _nextpow2(nt))
    n_events = len(tau)
    nh = len(h)
    wavelet, _ = ricker(f0, dt)
    nw = len(wavelet)
    W = np.fft.fft(wavelet, n=nfft)
    D = np.zeros((nfft, nh), dtype=complex)

    # Important: delay to have maximum of Ricker wavelet at right intercept time
    delay = dt * (np.floor(nw / 2) + 1)

    for ifreq in range(1, nfft // 2 + 1):
        w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
        for k in range(n_events):
            Shift = np.exp(
                -1j * w * (np.sqrt(tau[k] ** 2 + ((h - h0[k]) / v[k]) ** 2) - delay)
            )
            D[ifreq, :] += amp[k] * W[ifreq] * Shift

    # Apply w-domain symmetries
    for ifreq in range(1, nfft // 2):
        D[nfft + 1 - ifreq, :] = np.conj(D[ifreq, :])

    d = np.fft.ifft(D, axis=0)
    d = np.real(d[:nt, :])
    t = np.arange(nt) * dt

    return d, h, t


def make_traces(
    Nr: int,
    Ntraces: int,
    dt: float,
    w: np.ndarray,
    r_fun: str,
    arg,
    rho: float,
) -> tuple:
    """
    Make an ensemble of traces with correlated reflectivity.

    转换自 MATLAB: codes/synthetics/make_traces.m

    Parameters
    ----------
    Nr : int
        Number of samples of the reflectivity
    Ntraces : int
        Number of traces
    dt : float
        Sampling interval in secs
    w : np.ndarray
        Wavelet (column vector)
    r_fun : str
        Reflectivity function: 'bernoulli', 'laplace_mixture', 'gauss_mixture'
    arg : list or array
        Argument vector for r_fun
    rho : float
        Degree of similarity from trace to trace:
        - rho=0: no correlation between traces
        - rho=1: first trace is repeated Ntraces times

    Returns
    -------
    s : np.ndarray
        Seismic traces (length = Nr + Length of wavelet - 1)
    r : np.ndarray
        Reflectivity sequences (length = Nr)
    t : np.ndarray
        Time axis

    Examples
    --------
    >>> from seismiclab_py.synthetics import ricker
    >>> w, _ = ricker(40, 0.002)
    >>> s, r, t = make_traces(100, 20, 0.002, w, 'gauss_mixture', [1., 0.01, 0.1], 0.1)
    """
    from scipy.signal import convolve2d
    from .decon import taper as taper_func

    nw = len(w)

    r = np.zeros((Nr, Ntraces), dtype=float)

    # Get base reflectivity
    if r_fun == 'bernoulli':
        rr = bernoulli(Nr, arg[0], arg[1])
    elif r_fun == 'laplace_mixture':
        rr = laplace_mixture(Nr, arg)
    elif r_fun == 'gauss_mixture':
        rr = gauss_mixture(Nr, arg)
    else:
        raise ValueError(f"Unknown r_fun: {r_fun}")

    # Create correlated reflectivity sequences
    for k in range(Ntraces):
        if r_fun == 'bernoulli':
            r_new = bernoulli(Nr, arg[0], arg[1])
        elif r_fun == 'laplace_mixture':
            r_new = laplace_mixture(Nr, arg)
        elif r_fun == 'gauss_mixture':
            r_new = gauss_mixture(Nr, arg)

        r[:, k] = rho * rr + (1 - rho) * r_new

    # Convolve reflectivity with wavelet
    ss = convolve2d(r, w[:, None], mode="full")
    n2 = int(np.floor(nw / 2)) + 1
    s = ss[n2 : n2 + Nr - 1, :]

    Nt = s.shape[0]
    t = np.arange(Nt) * dt

    # Apply taper
    s, _ = taper_func(s, 10, 10)

    return s.astype(float), r.astype(float), t.astype(float)


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(np.ceil(np.log2(n)))

