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
    """Evaluate the far-field integral on the problem's active backend.

    For a CUDA problem the polarization current, coordinates, phase matrix, and
    reductions remain on the GPU.  Only the final ``(M, 3)`` array is copied to
    the host.  The CPU path is numerically identical and uses NumPy through the
    same implementation.
    """

    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            f"directions must have shape (M, 3), got {directions.shape}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    expected_u_shape = (3,) + tuple(problem.grid.N)
    u_shape = getattr(u, "shape", None)
    if u_shape != expected_u_shape:
        raise ValueError(f"u must have shape {expected_u_shape}, got {u_shape}")

    grid = problem.grid
    backend = grid.backend
    xp = backend.xp
    k0 = float(problem.k0)
    dv = float(grid.dv)

    contrast = problem.eps_tensor
    polarization_current = xp.einsum(
        "ij...,j...->i...", contrast, u
    ).reshape(3, -1)

    X, Y, Z = grid.coords()
    coordinates = xp.stack(
        [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=0
    )

    directions_device = xp.asarray(directions, dtype=backend.real_dtype)
    result = xp.zeros(
        (len(directions), 3), dtype=backend.complex_dtype
    )
    phase_scale = backend.complex_dtype(-1j * k0)
    volume_scale = backend.real_dtype(dv)

    for start in range(0, len(directions), batch_size):
        stop = min(start + batch_size, len(directions))
        direction_batch = directions_device[start:stop]
        dot_products = direction_batch @ coordinates
        phase = xp.exp(phase_scale * dot_products).astype(
            backend.complex_dtype, copy=False
        )
        result[start:stop] = (
            volume_scale * (polarization_current @ phase.T)
        ).T

    backend.synchronize()
    return np.asarray(backend.to_host(result), dtype=np.complex128)
