from __future__ import annotations

import numpy as np


def _interp_velocity(tnmo, vnmo, nt, dt):
    tnmo = np.asarray(tnmo, dtype=float).flatten()
    vnmo = np.asarray(vnmo, dtype=float).flatten()
    if vnmo.size > 1:
        t1 = np.concatenate(([0.0], tnmo, [(nt - 1) * dt]))
        v1 = np.concatenate(([vnmo[0]], vnmo, [vnmo[-1]]))
        ti = np.arange(nt, dtype=float) * dt
        vi = np.interp(ti, t1, v1)
    else:
        ti = np.arange(nt, dtype=float) * dt
        vi = np.full(nt, float(vnmo[0]))
    return ti, vi


def nmo(d: np.ndarray, dt: float, h: np.ndarray, tnmo, vnmo, max_stretch: float):
    """
    Normal moveout correction.
    """

    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    nt, nh = d.shape
    ti, vi = _interp_velocity(tnmo, vnmo, nt, dt)
    dout = np.zeros_like(d)
    M = np.zeros(nt, dtype=int)

    for it in range(nt):
        for ih in range(nh):
            time = np.sqrt(ti[it] ** 2 + (h[ih] / vi[it]) ** 2)
            stretch = (time - ti[it]) / (ti[it] + 1e-10)
            if stretch < max_stretch / 100.0:
                M[it] += 1
                its = time / dt
                it1 = int(np.floor(its))
                it2 = it1 + 1
                if it2 < nt:
                    a = its - it1
                    dout[it, ih] = (1 - a) * d[it1, ih] + a * d[it2, ih]
    return dout, M, ti, vi


def inmo(d: np.ndarray, dt: float, h: np.ndarray, tnmo, vnmo, max_stretch: float):
    """
    Inverse NMO.
    """

    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    nt, nh = d.shape
    ti, vi = _interp_velocity(tnmo, vnmo, nt, dt)
    dout = np.zeros_like(d)
    M = np.zeros(nt, dtype=int)

    for it in range(nt):
        for ih in range(nh):
            time = np.sqrt(ti[it] ** 2 + (h[ih] / vi[it]) ** 2)
            stretch = (time - ti[it]) / (ti[it] + 1e-10)
            if stretch < max_stretch / 100.0:
                M[it] += 1
                its = time / dt
                it1 = int(np.floor(its))
                it2 = it1 + 1
                if it2 < nt:
                    a = its - it1
                    dout[it1, ih] += (1 - a) * d[it, ih]
                    dout[it2, ih] += a * d[it, ih]
    return dout, M, ti, vi


def velan(d: np.ndarray, dt: float, h: np.ndarray, vmin: float, vmax: float, nv: int, R: int, L: int):
    """
    Velocity analysis (unnormalized cross-correlation semblance).
    """

    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    nt, nh = d.shape
    v = np.linspace(vmin, vmax, nv)
    tau = np.arange(0, nt, R) * dt
    taper = np.hamming(2 * L + 1)
    H = np.outer(taper, np.ones(nh))
    S = np.zeros((tau.size, nv))

    for it, tau_val in enumerate(tau):
        for iv, vv in enumerate(v):
            time = np.sqrt(tau_val**2 + (h / vv) ** 2)
            s = np.zeros((2 * L + 1, nh))
            for ig in range(-L, L + 1):
                ts = time + ig * dt
                isamp = ts / dt
                i1 = np.floor(isamp).astype(int)
                i2 = i1 + 1
                a = isamp - i1
                valid = (i1 >= 0) & (i2 < nt)
                s[ig + L, valid] = (1 - a[valid]) * d[i1[valid], np.where(valid)[0]] + a[valid] * d[
                    i2[valid], np.where(valid)[0]
                ]
            ss = s * H
            s1 = np.sum(np.sum(ss, axis=1) ** 2)
            s2 = np.sum(s**2)
            S[it, iv] = s1 / (s2 + 0.001)

    S = S / np.max(S)
    return S, tau, v


def parabolic_moveout(d: np.ndarray, dt: float, h: np.ndarray, qmin: float, qmax: float, nq: int, R: int, L: int):
    """
    Parabolic moveout coherence display.
    """

    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    nt, nh = d.shape
    q = np.linspace(qmin, qmax, nq)
    tau = np.arange(0, nt, R) * dt
    taper = np.hamming(2 * L + 1)
    H = np.outer(taper, np.ones(nh))
    hmax = np.max(np.abs(h))
    S = np.zeros((tau.size, nq))

    for it, tau_val in enumerate(tau):
        for iq, qq in enumerate(q):
            time = tau_val + qq * (h / hmax) ** 2
            s = np.zeros((2 * L + 1, nh))
            for ig in range(-L, L + 1):
                ts = time + ig * dt
                isamp = ts / dt
                i1 = np.floor(isamp).astype(int)
                i2 = i1 + 1
                a = isamp - i1
                valid = (i1 >= 0) & (i2 < nt)
                s[ig + L, valid] = (1 - a[valid]) * d[i1[valid], np.where(valid)[0]] + a[valid] * d[
                    i2[valid], np.where(valid)[0]
                ]
            s = s * H
            s1 = np.sum(np.sum(s, axis=1) ** 2)
            s2 = np.sum(s**2)
            S[it, iq] = s1 / (s2 + 1e-8)

    S = S / np.max(S)
    return S, tau, q


def lmo(d: np.ndarray, dt: float, h: np.ndarray, v: float) -> np.ndarray:
    """
    Linear moveout correction.

    转换自 MATLAB: codes/velan_nmo/lmo.m

    Parameters
    ----------
    d : np.ndarray
        Data gather (nt, nh)
    dt : float
        Sampling interval in secs
    h : np.ndarray
        Vector of offsets in meters
    v : float
        Linear moveout velocity in m/s

    Returns
    -------
    dout : np.ndarray
        Data after LMO correction

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(1000, 50)
    >>> dt = 0.004
    >>> h = np.arange(0, 1000, 20)
    >>> dout = lmo(d, dt, h, 2000)
    """
    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    nt, nh = d.shape
    dout = np.zeros_like(d)

    for ih in range(nh):
        arg = h[ih] / v
        time = arg
        its = time / dt
        it1 = int(np.floor(its))
        it2 = it1 + 1
        a = its - it1

        # Note: The original MATLAB code applies the shift at all time samples
        for it_idx in range(nt):
            it1_t = it_idx + it1
            it2_t = it_idx + it2
            if it2_t < nt:
                dout[it_idx, ih] = (1 - a) * d[it1_t, ih] + a * d[it2_t, ih]

    return dout


def stackgather(d: np.ndarray, N: np.ndarray = None) -> np.ndarray:
    """
    Stack one gather with normalization.

    转换自 MATLAB: codes/velan_nmo/stackgather.m

    s(t) = [sum_x d(x,t)] / N(t)

    Parameters
    ----------
    d : np.ndarray
        Data gather (nt, nh)
    N : np.ndarray, optional
        Normalization factor for each time (nt, 1).
        If not provided, divide by number of traces

    Returns
    -------
    s : np.ndarray
        Stack (normalized spatial average of traces)

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(1000, 50)
    >>> s = stackgather(d)
    """
    d = np.asarray(d, dtype=float)
    nt, nh = d.shape

    if N is None:
        N = nh * np.ones((nt, 1), dtype=float)

    s = np.sum(d, axis=1, keepdims=True)

    for k in range(nt):
        if N[k, 0] > 0:
            s[k, 0] = s[k, 0] / N[k, 0]
        else:
            s[k, 0] = 0.0

    return s
