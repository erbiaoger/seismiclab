from __future__ import annotations

import numpy as np


def _nextpow2(n: int) -> int:
    return int(np.ceil(np.log2(n)))


def _th_schedule(option, x, perc_i, perc_f, N):
    nx1, nx2 = x.shape
    nk1 = 2 ** _nextpow2(nx1)
    nk2 = 2 ** _nextpow2(nx2)
    X = np.fft.fft2(x, s=(nk1, nk2))
    A = np.abs(X)
    Amax = np.max(A)
    th_i = perc_i * Amax / 100.0
    th_f = perc_f * Amax / 100.0
    k = np.arange(1, N + 1)
    if option == 1:
        th = th_i + (th_f - th_i) * (k - 1) / (N - 1)
    elif option == 2:
        b = -np.log(th_f / th_i)
        th = th_i * np.exp(-b * (k - 1) / (N - 1))
    else:
        avec = np.sort(A.ravel())[::-1]
        avec = avec[(avec < perc_i * Amax / 100.0) & (avec > perc_f * Amax / 100.0)]
        th = np.zeros(N)
        th[0] = avec[0]
        th[-1] = avec[-1]
        for j in range(1, N - 1):
            th[j] = avec[int(np.ceil(j * len(avec) / (N - 1)))]
    return th


def pocs(d, dtrue, T, dt, f_low, f_high, option, perc_i, perc_f, N, a, tol):
    d = np.asarray(d, dtype=complex)
    dtrue = np.asarray(dtrue, dtype=complex)
    T = np.asarray(T, dtype=float)
    nt, nx1, nx2 = d.shape

    nf = 2 ** _nextpow2(nt)
    nk1 = 2 ** _nextpow2(nx1)
    nk2 = 2 ** _nextpow2(nx2)
    S = 1.0 - T

    k_low = max(int(np.floor(f_low * dt * nf)) + 1, 1)
    k_high = min(int(np.floor(f_high * dt * nf)) + 1, nf // 2)

    Dout = np.zeros((nf, nx1, nx2), dtype=complex)
    D = np.fft.fft(d, n=nf, axis=0)
    Dtrue = np.fft.fft(dtrue, n=nf, axis=0)
    e1 = []
    e2 = []

    k_low_idx = k_low - 1
    k_high_idx = k_high - 1

    for k_idx in range(k_low_idx, k_high_idx + 1):
        freq_val = (k_idx + 1) / (nf * dt)
        x = D[k_idx, :, :]
        xtrue = Dtrue[k_idx, :, :]
        th = _th_schedule(option, x, perc_i, perc_f, N)
        y = x.copy()
        iter_idx = 0
        E1 = 100.0
        while iter_idx < N and E1 > tol:
            Y = np.fft.fft2(y, s=(nk1, nk2))
            A = np.abs(Y)
            Angle = np.angle(Y)
            mask = A < th[iter_idx]
            A[mask] = 0
            Y = A * np.exp(1j * Angle)
            y = np.fft.ifft2(Y)[:nx1, :nx2]
            yold = y.copy()
            y = a * x + (1 - a) * T * y + S * y
            dif1 = y - yold
            dif2 = y - xtrue
            c1 = np.sum(np.abs(dif1) ** 2)
            c2 = np.sum(np.abs(dif2) ** 2)
            c = np.sum(np.abs(y) ** 2)
            ct = np.sum(np.abs(xtrue) ** 2)
            E1 = c1 / c
            E2 = c2 / ct if ct != 0 else 0
            e1.append(E1)
            e2.append(E2)
            iter_idx += 1
        Dout[k_idx, :, :] = y
        Dout[nf - k_idx - 1, :, :] = np.conj(y)

    dout = np.fft.ifft(Dout, axis=0)[:nt, :, :].real
    freq = np.arange(k_low_idx, k_high_idx + 1) / (nf * dt)
    e1 = np.array(e1).reshape(-1, iter_idx) if e1 else np.array([])
    e2 = np.array(e2).reshape(-1, iter_idx) if e2 else np.array([])
    return dout, e1, e2, freq
