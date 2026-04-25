"""Variant B: 3D FFT + map_coordinates scatter integral.

Algorithm
---------
1. Compute J_dv = eta @ u * dv  (shape 3xN1xN2xN3)
2. Zero-pad by factor PAD and apply fftn + fftshift on each of the 3 components.
   Zero-padding oversamples the DFT, reducing interpolation error from O(1) to
   O(1/PAD^2) so that bilinear (order=1) interpolation is accurate.
3. For each observation direction e_p = (ex, ey, ez), the sample
   coordinates in the zero-padded fftshifted array are:
       ix = k0 * ex * Lx / (2*pi) * PAD + (PAD*Nx)/2
       iy = k0 * ey * Ly / (2*pi) * PAD + (PAD*Ny)/2
       iz = k0 * ez * Lz / (2*pi) * PAD + (PAD*Nz)/2
4. Interpolate (bilinear, order=1) real and imaginary parts separately
5. Apply phase correction for grid corner r0:
       F *= exp(-1j * k0 * (e_p @ r0))
   where r0 = [cx - Lx/2 + dx/2,  cy - Ly/2 + dy/2,  cz - Lz/2 + dz/2]
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..problem import Problem

# Zero-padding factor: oversamples the DFT spectrum so that bilinear
# interpolation error is suppressed below ~1e-5 for k0*L/(2*pi) <= 1.
_PAD = 32


def scatter_integral_fft(
    u,
    problem: Problem,
    directions,
) -> np.ndarray:
    """Return F of shape (M, 3) as complex128 numpy array.

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex
    problem    : Problem
    directions : array (M, 3) float — unit vectors e_p

    Returns
    -------
    F : ndarray (M, 3) complex128
    """
    grid = problem.grid
    be = grid.backend
    xp = be.xp
    k0 = problem.k0
    dv = grid.dv
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    cx, cy, cz = grid.center

    # 1. Polarization current * dv  ->  (3, Nx, Ny, Nz)
    eta = problem.eps_tensor                             # (3, 3, Nx, Ny, Nz)
    J = xp.einsum("ij...,j...->i...", eta, u)           # (3, Nx, Ny, Nz)
    J_dv = J * dv

    # 2. Move to numpy (handles CuPy transparently), then zero-pad + fftn + fftshift
    J_dv_np = be.to_host(J_dv).astype(np.complex128)    # (3, Nx, Ny, Nz)
    PNx, PNy, PNz = _PAD * Nx, _PAD * Ny, _PAD * Nz
    J_hat = np.fft.fftshift(
        np.fft.fftn(J_dv_np, s=(PNx, PNy, PNz), axes=(-3, -2, -1)),
        axes=(-3, -2, -1),
    )                                                    # (3, PNx, PNy, PNz)

    # 3. Grid corner r0  (phase correction for non-zero centre)
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    r0 = np.array([cx - Lx / 2.0 + dx / 2.0,
                   cy - Ly / 2.0 + dy / 2.0,
                   cz - Lz / 2.0 + dz / 2.0])

    # 4. Fractional interpolation indices in the zero-padded fftshifted array
    dirs_np = np.asarray(directions, dtype=np.float64)  # (M, 3)
    ix = k0 * dirs_np[:, 0] * Lx / (2.0 * np.pi) * _PAD + PNx / 2.0  # (M,)
    iy = k0 * dirs_np[:, 1] * Ly / (2.0 * np.pi) * _PAD + PNy / 2.0
    iz = k0 * dirs_np[:, 2] * Lz / (2.0 * np.pi) * _PAD + PNz / 2.0
    coords = np.stack([ix, iy, iz], axis=0)             # (3, M)

    # 5. Bilinear interpolation component by component
    #    map_coordinates does not support complex arrays -> split real/imag
    M = len(dirs_np)
    F = np.zeros((M, 3), dtype=np.complex128)
    for i in range(3):
        F[:, i] = (
            map_coordinates(J_hat[i].real, coords, order=1, mode="nearest")
            + 1j * map_coordinates(J_hat[i].imag, coords, order=1, mode="nearest")
        )

    # 6. Phase correction: multiply by exp(-1j * k0 * (e_p @ r0))
    phase_corr = np.exp(-1j * k0 * (dirs_np @ r0))     # (M,)
    F *= phase_corr[:, np.newaxis]

    return F
