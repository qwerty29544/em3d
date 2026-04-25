"""em3d.farfield — far-field scatter integral and RCS post-processing.

Public API
----------
scatter_integral(u, problem, directions, *, method="direct", batch_size=64) -> (M, 3) complex
rcs(u, problem, direction) -> float
rcs_plane(u, problem, n_phi=80, plane="xy", method="direct", batch_size=64) -> (phi, sigma)
"""
from __future__ import annotations

import numpy as np

from ..problem import Problem
from ._core import scatter_integral_direct

__all__ = ["scatter_integral", "rcs", "rcs_plane"]


def scatter_integral(
    u,
    problem: Problem,
    directions,
    *,
    method: str = "direct",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute the far-field scatter integral F(ê_p).

    F[m] = ΔV · Σ_q  η_q · u_q · exp(−ik₀ · ê_p[m] · r_q)

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex — E-field solution
    problem    : Problem — eps_tensor (η=ε−I), k0, grid
    directions : array (M, 3) float — unit observation vectors; a single (3,)
                 vector is automatically broadcast to (1, 3)
    method     : "direct"  — batched matmul, O(3NM) flops, O(N+batch·N) memory
                 "fft"     — 3D FFT + map_coordinates, O(9N log N + 9M) flops
    batch_size : directions per batch (used only for method="direct")

    Returns
    -------
    F : ndarray (M, 3) complex128 — always numpy, on CPU
    """
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[np.newaxis, :]
    if method == "direct":
        return scatter_integral_direct(u, problem, directions, batch_size=batch_size)
    elif method == "fft":
        from ._fft import scatter_integral_fft
        return scatter_integral_fft(u, problem, directions)
    else:
        raise ValueError(f"method must be 'direct' or 'fft', got {method!r}")


def rcs(u, problem: Problem, direction) -> float:
    """RCS for a single observation direction.

    σ(ê_p) = k₀⁴ / (16π²) · |ê_p × F(ê_p)|²

    Parameters
    ----------
    u         : array (3, N1, N2, N3) complex
    problem   : Problem
    direction : array (3,) float — unit vector (normalised internally)

    Returns
    -------
    float — non-negative RCS value
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    F = scatter_integral(u, problem, direction[np.newaxis, :])[0]  # (3,) complex128
    cross = np.cross(direction, F)                                  # (3,) complex128
    cross_norm_sq = float(np.real(np.dot(cross, cross.conj())))
    return problem.k0 ** 4 / (16.0 * np.pi ** 2) * cross_norm_sq


def rcs_plane(
    u,
    problem: Problem,
    n_phi: int = 80,
    plane: str = "xy",
    method: str = "direct",
    batch_size: int = 64,
) -> tuple:
    """RCS curve over n_phi equally-spaced directions in a coordinate plane.

    Parameters
    ----------
    u         : array (3, N1, N2, N3) complex
    problem   : Problem
    n_phi     : number of directions in [0, 2π)
    plane     : "xy" | "yz" | "xz"
    method    : "direct" | "fft"
    batch_size: used only for method="direct"

    Returns
    -------
    phi   : ndarray (n_phi,) — angles in radians
    sigma : ndarray (n_phi,) — RCS values ≥ 0
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    c, s, z = np.cos(phi), np.sin(phi), np.zeros_like(phi)
    if plane == "xy":
        dirs = np.stack([c, s, z], axis=-1)    # (n_phi, 3)
    elif plane == "yz":
        dirs = np.stack([z, c, s], axis=-1)
    elif plane == "xz":
        dirs = np.stack([c, z, s], axis=-1)
    else:
        raise ValueError(f"plane must be 'xy'|'yz'|'xz', got {plane!r}")

    F = scatter_integral(u, problem, dirs, method=method, batch_size=batch_size)  # (n_phi, 3)
    cross = np.cross(dirs, F)                                                      # (n_phi, 3)
    sigma = (problem.k0 ** 4 / (16.0 * np.pi ** 2)
             * np.real(np.einsum("mi,mi->m", cross, cross.conj())))               # (n_phi,)
    return phi, sigma
