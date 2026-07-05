"""Far-field diagnostics for scalar acoustic scattering."""
from __future__ import annotations

import numpy as np

from .problem import AcousticProblem


def _directions_array(directions) -> np.ndarray:
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError(f"directions must have shape (M, 3), got {dirs.shape}")
    norms = np.linalg.norm(dirs, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("directions must be non-zero")
    return dirs / norms[:, None]


def _plane_directions(plane: str, phi: np.ndarray) -> np.ndarray:
    if plane == "xy":
        return np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    if plane == "xz":
        return np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)])
    if plane == "yz":
        return np.column_stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)])
    raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")


def farfield_amplitude(
    u,
    problem: AcousticProblem,
    directions,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Return scalar acoustic far-field amplitude for observation directions."""
    dirs = _directions_array(directions)
    be = problem.grid.backend
    xp = be.xp
    u_arr = xp.asarray(u, dtype=be.complex_dtype)
    if u_arr.shape != problem.grid.N:
        raise ValueError(f"u.shape {u_arr.shape} != expected {problem.grid.N}")

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    source = (problem.chi * u_arr).reshape(-1)
    X, Y, Z = problem.grid.coords()
    r_flat = xp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=0)
    source_np = be.to_host(source).astype(np.complex128)
    r_np = be.to_host(r_flat).astype(np.float64)
    out = np.zeros(len(dirs), dtype=np.complex128)
    coeff = (float(problem.k0) ** 2) * problem.grid.dv / (4.0 * np.pi)

    for start in range(0, len(dirs), batch):
        d = dirs[start : start + batch]
        phase = np.exp(-1j * float(problem.k0) * (d @ r_np))
        out[start : start + len(d)] = coeff * (phase @ source_np)
    return out


def scattering_pattern(
    u,
    problem: AcousticProblem,
    directions,
    *,
    normalize: str | None = "max",
) -> np.ndarray:
    """Return ``|f(s)|^2`` with optional max normalization."""
    amp = farfield_amplitude(u, problem, directions)
    sigma = np.abs(amp) ** 2
    if normalize is None:
        return sigma.astype(np.float64, copy=False)
    if normalize != "max":
        raise ValueError(f"normalize must be None or 'max', got {normalize!r}")
    max_value = float(np.max(sigma)) if sigma.size else 0.0
    if max_value > 0.0:
        sigma = sigma / max_value
    return sigma.astype(np.float64, copy=False)


def pattern_plane(
    u,
    problem: AcousticProblem,
    *,
    plane: str = "xy",
    n_angles: int = 360,
    normalize: str | None = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Return angle grid and acoustic scattering pattern in a coordinate plane."""
    if int(n_angles) <= 0:
        raise ValueError(f"n_angles must be positive, got {n_angles!r}")
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    dirs = _plane_directions(plane, phi)
    return phi, scattering_pattern(u, problem, dirs, normalize=normalize)
