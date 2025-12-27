from __future__ import annotations

import numpy as np


def _nextpow2(n: int) -> int:
    return int(np.ceil(np.log2(n)))


def inverse_radon_freq(d, dt, h, q, N, flow, fhigh, mu, sol):
    d = np.asarray(d, dtype=complex)
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)
    nt, nh = d.shape
    if N == 2:
        h = h / np.max(np.abs(h))

    nfft = 4 * 2 ** (_nextpow2(nt) + 1)
    if1 = int(np.floor(flow * dt * nfft)) + 1
    if2 = int(np.floor(fhigh * dt * nfft)) + 1

    D = np.fft.fft(d, n=nfft, axis=0)
    M = np.zeros((nfft, len(q)), dtype=complex)

    for ifreq in range(if1, if2 + 1):
        # MATLAB is 1-based; w uses (ifreq-1)
        w = 2 * np.pi * (ifreq - 1) / (nfft * dt)
        L = np.exp(-1j * w * np.power(h[:, None], N) * q[None, :])
        y = D[ifreq, :].reshape(-1, 1)
        if sol == "hr":
            x = _solver_hr_radon(L, y, mu)
        else:
            x = _solver_ls_radon(L, y, mu)
        M[ifreq, :] = x.ravel()

    # Hermitian symmetry (match MATLAB loop nfft/2+2:nfft)
    for idx in range(nfft // 2 + 1, nfft):
        M[idx, :] = np.conj(M[nfft - idx, :])
    m = np.fft.ifft(M, axis=0)
    return m[:nt, :]


def _solver_ls_radon(L, y, mu):
    nh, np_ = L.shape
    if nh <= np_:
        A = L @ L.conj().T + mu * np.eye(nh)
        u = np.linalg.solve(A, y)
        x = L.conj().T @ u
    else:
        x = np.linalg.solve(L.conj().T @ L + mu * np.eye(np_), L.conj().T @ y)
    return x


def _solver_hr_radon(L, y, mu):
    nh, np_ = L.shape
    if nh <= np_:
        Q = np.eye(np_, dtype=complex)
        for _ in range(3):
            A = L @ Q @ L.conj().T + mu * np.eye(nh)
            u = np.linalg.solve(A, y)
            x = Q @ L.conj().T @ u
            q = (np.abs(x) ** 2 + 0.001).ravel()
            Q = np.diag(q)
    else:
        Q = np.eye(np_, dtype=complex)
        for _ in range(3):
            x = np.linalg.solve(L.conj().T @ L + mu * Q, L.conj().T @ y)
            q = 1.0 / (np.abs(x) ** 2 + 0.001).ravel()
            Q = np.diag(q)
    return x


def forward_radon_freq(m, dt, h, p, N, flow, fhigh, ntd):
    m = np.asarray(m, dtype=complex)
    h = np.asarray(h, dtype=float)
    nt, nq = m.shape
    if N == 2:
        h = (h / np.max(np.abs(h))) ** 2

    nfft = 4 * 2 ** _nextpow2(nt)
    M = np.fft.fft(m, n=nfft, axis=0)
    D = np.zeros((nfft, len(h)), dtype=complex)
    if1 = int(np.floor(flow * dt * nfft)) + 1
    if2 = int(np.floor(fhigh * dt * nfft)) + 1

    for ifreq in range(if1, if2 + 1):
        f = 2 * np.pi * (ifreq - 1) / (nfft * dt)
        L = np.exp(-1j * f * np.outer(h, p))
        x = M[ifreq, :].reshape(-1, 1)
        y = L @ x
        D[ifreq, :] = y.ravel()

    for idx in range(nfft // 2 + 1, nfft):
        D[idx, :] = np.conj(D[nfft - idx, :])
    d = np.fft.ifft(D, axis=0)
    return d[:ntd, :]


def pradon_demultiple(d, dt, h, qmin, qmax, nq, flow, fhigh, mu, q_cut, sol):
    d = np.asarray(d, dtype=float)
    nt, _ = d.shape
    dq = (qmax - qmin) / (nq - 1)
    q = qmin + dq * np.arange(nq)
    m = inverse_radon_freq(d, dt, h, q, 2, flow, fhigh, mu, sol)
    m = np.real(m)
    iq_cut = int(np.floor((q_cut - qmin) / dq)) + 1
    mc = m.copy()
    mc[:, :iq_cut] = 0.0
    dm = forward_radon_freq(mc, dt, h, q, 2, flow, fhigh, nt)
    dm = np.real(dm)
    prim = d - dm
    tau = np.arange(nt) * dt
    return prim, m, tau, q
