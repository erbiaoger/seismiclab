from __future__ import annotations

import numpy as np
from scipy.linalg import hankel, toeplitz
from scipy.signal import windows


def _nextpow2(n: int) -> int:
    return int(np.ceil(np.log2(n)))


def fx_decon(d: np.ndarray, dt: float, lf: int, mu: float, flow: float, fhigh: float):
    d = np.asarray(d, dtype=float)
    nt, ntraces = d.shape
    nf = 2 ** _nextpow2(nt)

    D_f = np.zeros((nf, ntraces), dtype=complex)
    D_b = np.zeros((nf, ntraces), dtype=complex)

    ilow = max(int(np.floor(flow * dt * nf)) + 1, 1)
    ihigh = min(int(np.floor(fhigh * dt * nf)) + 1, nf // 2 + 1)

    D = np.fft.fft(d, n=nf, axis=0)
    for k in range(ilow, ihigh + 1):
        aux = D[k, :].reshape(-1, 1)
        aux_out_f, aux_out_b = _ar_modeling(aux, lf, mu)
        D_f[k, :] = aux_out_f[:, 0]
        D_b[k, :] = aux_out_b[:, 0]

    for k in range(nf // 2 + 1, nf):
        D_f[k, :] = np.conj(D_f[nf - k, :])
        D_b[k, :] = np.conj(D_b[nf - k, :])

    d_f = np.fft.ifft(D_f, axis=0).real[:nt, :]
    d_b = np.fft.ifft(D_b, axis=0).real[:nt, :]
    df = d_f + d_b
    if ntraces > 2 * lf:
        df[:, lf : ntraces - lf] = df[:, lf : ntraces - lf] / 2.0
    return df


def _ar_modeling(x: np.ndarray, lf: int, mu: float):
    x = np.asarray(x, dtype=complex).flatten()
    nx = x.shape[0]

    # backward AR modelling
    y = x[: nx - lf]
    C = x[1 : nx - lf + 1]
    R = x[nx - lf :]
    M = hankel(C, R)
    B = M.conj().T @ M
    beta = B[0, 0] * mu / 100.0
    ab = np.linalg.solve(B + beta * np.eye(lf), M.conj().T @ y)
    temp = M @ ab
    yb = np.concatenate([temp, np.zeros(lf, dtype=complex)])

    # forward AR modelling
    y = x[lf:]
    C = x[lf - 1 : nx - 1]
    R = np.flipud(x[:lf])
    M = toeplitz(C, R)
    B = M.conj().T @ M
    beta = B[0, 0] * mu / 100.0
    af = np.linalg.solve(B + beta * np.eye(lf), M.conj().T @ y)
    temp = M @ af
    yf = np.concatenate([np.zeros(lf, dtype=complex), temp])
    return yf[:, None], yb[:, None]


# ---------- Spitz FX interpolation helpers ----------


def spitz_fx_interpolation(
    d: np.ndarray, dt: float, npf: int, pre1: float, pre2: float, flow: float, fhigh: float
):
    d = np.asarray(d, dtype=float)
    nt, nh = d.shape
    nfft = 2 ** _nextpow2(nt)

    DF1 = np.fft.fft(d, n=2 * nfft, axis=0)
    DF2 = np.fft.fft(d, n=nfft, axis=0)

    ilow = int(np.floor(flow * dt * nfft)) + 1
    ihigh = int(np.floor(fhigh * dt * nfft)) + 1

    INTDF = np.zeros((nfft, 2 * nh - 1), dtype=complex)
    for ia in range(ilow, ihigh + 1):
        x1 = DF1[ia, :]
        x2 = DF2[ia, :]
        PF = _prediction_filter(x1, npf, pre1)
        y = _interpolate_freq(x2, PF, pre2)
        INTDF[ia, :] = y

    INTDF[nfft // 2 + 1 :] = np.conj(np.flipud(INTDF[1 : nfft // 2]))
    d_interp = np.fft.ifft(INTDF, n=nfft, axis=0).real[:nt, :]
    return d_interp


def _convmtx(vec: np.ndarray, n: int) -> np.ndarray:
    vec = np.asarray(vec).flatten()
    m = vec.size
    col = np.concatenate([vec, np.zeros(n - 1)])
    row = np.concatenate([[vec[0]], np.zeros(n - 1)])
    return toeplitz(col, row)[: m + n - 1, :n]


def _interpolate_freq(x: np.ndarray, PF: np.ndarray, pre: float) -> np.ndarray:
    npf = len(PF)
    nx = len(x)
    ny = 2 * nx - 1

    TMPF1 = np.concatenate([PF[::-1].conj(), [-1]])
    W1 = _convmtx(TMPF1, ny)
    TMPF2 = np.conjugate(TMPF1[::-1])
    W2 = _convmtx(TMPF2, ny)
    WT = np.vstack([W1, W2])

    A = WT[:, 1::2]
    B = -WT[:, ::2] @ x.reshape(-1, 1)
    R = A.T @ A
    g = A.T @ B
    mu = (pre / 100.0) * np.trace(R) / (nx - 1)
    y1 = np.linalg.solve(R + mu * np.eye(nx - 1), g)
    y = np.zeros(ny, dtype=complex)
    y[::2] = x
    y[1::2] = y1.ravel()
    return y


def _prediction_filter(VEC: np.ndarray, npf: int, pre: float) -> np.ndarray:
    VEC = np.asarray(VEC, dtype=complex).flatten()
    ns = len(VEC)
    C = np.zeros((ns - npf, npf + 1), dtype=complex)
    for j in range(ns - npf):
        C[j, :] = VEC[j : j + npf + 1][::-1]
    A = np.vstack([C[:, 1 : npf + 1], np.conj(np.fliplr(C[:, :npf]))])
    B = np.concatenate([C[:, 0], np.conj(C[:, npf])])
    R = A.conj().T @ A
    g = A.conj().T @ B
    mu = (pre / 100.0) * np.trace(R) / npf
    PF = np.linalg.solve(R + mu * np.eye(npf), g)
    return PF
