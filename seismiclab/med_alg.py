from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from .decon import _convmtx, delay


def funalpha(q, alpha):
    return (q**alpha - 1) / alpha


def dfunalpha(q, alpha):
    return q ** (alpha - 1)


def funln(q, _):
    return np.log(q)


def dfunln(q, _):
    return 1.0 / q


def non_linear_output(y: np.ndarray, fun, dfun, arg):
    y = np.asarray(y, dtype=float)
    N = len(y)
    c1 = np.sum(y**2) / N
    q = (y**2) / c1
    F = fun(q, arg)
    dF = dfun(q, arg)
    G = F + q * dF
    c2 = np.sum(G * q) / N
    b = G * y / c2
    c3 = N * fun(N, arg)
    h = np.sum(q * F) / c3
    return b, h


def med(wbp, s, dt, Nf, mu, Updates, fun, dfun, arg):
    s = np.asarray(s, dtype=float)
    Nt, Nc = s.shape
    Rs = np.zeros((Nf, Nf))
    for k in range(Nc):
        S = _convmtx(s[:, k], Nf)
        Rs += S.T @ S
    Rs = Rs / Nc
    mu = Rs[0, 0] * mu / 100.0

    f = np.zeros((Nf, 1))
    f[Nf // 2, 0] = 1.0
    x = convolve2d(s, f, mode="full")
    Q = np.eye(Nf)
    Matrix = np.linalg.inv(Rs + mu * Q)
    Med_Norm = []

    for _ in range(Updates):
        vs = 0.0
        g = np.zeros((Nf, 1))
        for k in range(Nc):
            b, v = non_linear_output(x[:, k], fun, dfun, arg)
            S = _convmtx(s[:, k], Nf)
            g += S.T @ b.reshape(-1, 1)
            vs += v
        g = g / Nc
        vs = vs / Nc
        f = Matrix @ g
        Med_Norm.append(vs)
        x = convolve2d(s, f, mode="full")

    x = convolve2d(x, wbp[:, None], mode="same")
    x = x[:Nt, :]
    x = delay(s, x, Nf)
    tx = np.arange(Nt) * dt
    tf = np.arange(Nf) * dt

    smax = np.max(s)
    xmax = np.max(x)
    c = smax / xmax if xmax != 0 else 1.0
    x = c * x
    f = f / c
    return f, tf, x, tx, np.array(Med_Norm)
