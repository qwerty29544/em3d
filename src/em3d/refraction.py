"""Refractive-index / permittivity profiles on the grid."""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .grid import Grid


def _to_eta_mat(eps_real, eps_imag) -> np.ndarray:
    """Convert eps_real/eps_imag to a (3,3) complex128 eta matrix η = ε − I.

    Parameters
    ----------
    eps_real : float or array-like (3,3) — real part of permittivity ε
    eps_imag : float or array-like (3,3) — imaginary part of permittivity ε

    Both arguments must be the same kind: both scalars or both (3,3) arrays.
    Scalar inputs are promoted to scalar * eye(3) (isotropic case).

    Returns
    -------
    np.ndarray shape (3,3), dtype complex128
    """
    real_is_scalar = np.ndim(eps_real) == 0
    imag_is_scalar = np.ndim(eps_imag) == 0
    if real_is_scalar != imag_is_scalar:
        raise TypeError(
            "eps_real and eps_imag must both be scalars or both be (3,3) arrays; "
            f"got ndim={np.ndim(eps_real)} and ndim={np.ndim(eps_imag)}"
        )
    if real_is_scalar:
        if np.iscomplex(eps_real):
            raise TypeError(
                f"eps_real must be a real scalar, got complex value {eps_real!r}"
            )
        if np.iscomplex(eps_imag):
            raise TypeError(
                f"eps_imag must be a real scalar, got complex value {eps_imag!r}"
            )
        return (float(eps_real) - 1.0 + 1j * float(eps_imag)) * np.eye(3, dtype=np.complex128)
    E = np.asarray(eps_real, dtype=np.float64)
    F = np.asarray(eps_imag, dtype=np.float64)
    if E.shape != (3, 3):
        raise ValueError(f"eps_real must have shape (3,3), got {E.shape}")
    if F.shape != (3, 3):
        raise ValueError(f"eps_imag must have shape (3,3), got {F.shape}")
    return (E - np.eye(3)) + 1j * F


def cylinder_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    radius: float,
    axis: Literal["x", "y", "z"] = "z",
) -> object:
    """Infinite cylinder along `axis`, radius in grid length units.

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    if axis == "z":
        r2 = X * X + Y * Y
    elif axis == "y":
        r2 = X * X + Z * Z
    elif axis == "x":
        r2 = Y * Y + Z * Z
    else:
        raise ValueError(f"axis must be 'x'|'y'|'z', got {axis!r}")
    mask = r2 <= radius * radius
    eta_mat = _to_eta_mat(eps_real, eps_imag)          # (3,3) complex128, numpy
    out = be.zeros((3, 3) + grid.N, kind="complex")    # pre-zeroed backend array
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def step_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    z_min: float,
    z_max: float,
) -> object:
    """Slab between z_min and z_max (inclusive).

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    _, _, Z = grid.coords()
    mask = (Z >= z_min) & (Z <= z_max)
    eta_mat = _to_eta_mat(eps_real, eps_imag)
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def ellipsis_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    center: Tuple[float, float, float],
    radius: Tuple[float, float, float],
) -> object:
    """Axis-aligned ellipsoid.

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    cx, cy, cz = center
    rx, ry, rz = radius
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    mask = metric <= 1.0
    eta_mat = _to_eta_mat(eps_real, eps_imag)
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def apply_refraction(grid: Grid, *, scalar_eta=None, tensor_eta=None) -> object:
    """Return a (3, 3, Nx, Ny, Nz) complex tensor.

    Parameters
    ----------
    scalar_eta : array (Nx, Ny, Nz) complex — isotropic η field; placed on diagonal
    tensor_eta : array (3, 3, Nx, Ny, Nz) complex — full anisotropic η tensor

    Exactly one of scalar_eta or tensor_eta must be provided.
    """
    be = grid.backend
    if (scalar_eta is None) == (tensor_eta is None):
        raise ValueError("apply_refraction requires exactly one of scalar_eta or tensor_eta")
    if scalar_eta is not None:
        if len(scalar_eta.shape) == 5 and scalar_eta.shape[:2] == (3, 3):
            raise ValueError(
                f"scalar_eta has shape {scalar_eta.shape} which looks like a full "
                f"(3,3,Nx,Ny,Nz) tensor — did you pass the result of a geometry "
                f"function (cylinder_refraction / step_refraction / ellipsis_refraction) "
                f"directly? These functions now return (3,3,Nx,Ny,Nz) tensors ready for "
                f"use as eps_tensor; apply_refraction is not needed:\n"
                f"    eta = cylinder_refraction(...)  # shape (3,3,Nx,Ny,Nz)\n"
                f"    problem = Problem(..., eps_tensor=eta, ...)"
            )
        if scalar_eta.shape != grid.N:
            raise ValueError(f"scalar_eta.shape {scalar_eta.shape} != grid.N {grid.N}")
        out = be.zeros((3, 3) + grid.N, kind="complex")
        for i in range(3):
            out[i, i] = scalar_eta
        return out
    if tensor_eta.shape != (3, 3) + grid.N:
        raise ValueError(
            f"tensor_eta.shape {tensor_eta.shape} != {(3, 3) + grid.N}"
        )
    return tensor_eta.astype(be.complex_dtype, copy=False)
