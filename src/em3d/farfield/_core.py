"""Variant A: batched matmul computation of the scatter integral.

F[m] = dv * J_flat @ phase[:, m].T
where J_flat[i, n] = sum_j eta[i,j,...][n] * u[j,...][n]   (polarization current)
      phase[b, n]  = exp(-1j * k0 * (e_p[b] @ r[n]))
"""
from __future__ import annotations

import numpy as np

from ..problem import Problem


def scatter_integral_direct(
    u,
    problem: Problem,
    directions,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Return F of shape (M, 3) as complex128 numpy array.

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex — E-field
    problem    : Problem — provides eps_tensor, k0, grid
    directions : array (M, 3) float — unit vectors ê_p
    batch_size : int — directions processed per batch to cap memory
    """
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            f"directions must have shape (M, 3), got {directions.shape}"
        )
    expected_u_shape = (3,) + tuple(problem.grid.N)
    u_arr = np.asarray(u)
    if u_arr.shape != expected_u_shape:
        raise ValueError(f"u must have shape {expected_u_shape}, got {u_arr.shape}")
    grid = problem.grid
    be = grid.backend
    xp = be.xp
    k0 = problem.k0
    dv = grid.dv

    # polarization current J = eta @ u  →  (3, N1, N2, N3)
    eta = problem.eps_tensor                                 # (3, 3, N1, N2, N3)
    J = xp.einsum("ij...,j...->i...", eta, u)               # (3, N1, N2, N3)
    J_flat = J.reshape(3, -1)                               # (3, N)

    # grid coordinates (3, N)
    X, Y, Z = grid.coords()
    r_flat = xp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=0)  # (3, N)

    # move to numpy (works for both CPU and GPU backends)
    J_np = be.to_host(J_flat).astype(np.complex128)         # (3, N)
    r_np = be.to_host(r_flat).astype(np.float64)            # (3, N)
    dirs_np = np.asarray(directions, dtype=np.float64)      # (M, 3)
    M = len(dirs_np)
    F = np.zeros((M, 3), dtype=np.complex128)

    for start in range(0, M, batch_size):
        e_batch = dirs_np[start : start + batch_size]       # (b, 3)
        # dot(e_p, r) for every pair:  (b, 3) @ (3, N)  ->  (b, N)
        dot_er = e_batch @ r_np                             # (b, N)
        phase = np.exp(-1j * k0 * dot_er)                  # (b, N)
        # F_batch = dv * J_np @ phase.T  ->  (3, b), then transpose
        F[start : start + batch_size] = (dv * (J_np @ phase.T)).T   # (b, 3)

    return F
