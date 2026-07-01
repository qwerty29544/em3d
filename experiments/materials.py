"""Material models for chapter 6 electrodynamic experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    """Serializable dielectric material specification for research experiments."""

    kind: str
    eps_real: Any = None
    eps_imag: Any = 0.0
    eps_inf: float | None = None
    omega_p: float | None = None
    gamma: float | None = None
    orientation: Any = None

    @classmethod
    def isotropic(cls, eps_r: float | complex) -> "MaterialSpec":
        eps = complex(eps_r)
        return cls(kind="isotropic", eps_real=float(eps.real), eps_imag=float(eps.imag))

    @classmethod
    def anisotropic(cls, eps_real, *, orientation=None) -> "MaterialSpec":
        return cls(
            kind="anisotropic",
            eps_real=np.array(eps_real, dtype=np.float64, copy=True),
            eps_imag=np.zeros((3, 3), dtype=np.float64),
            orientation=orientation,
        )

    @classmethod
    def anisotropic_lossy(cls, eps_real, eps_imag, *, orientation=None) -> "MaterialSpec":
        return cls(
            kind="anisotropic_lossy",
            eps_real=np.array(eps_real, dtype=np.float64, copy=True),
            eps_imag=np.array(eps_imag, dtype=np.float64, copy=True),
            orientation=orientation,
        )

    @classmethod
    def plasma_drude(
        cls,
        *,
        eps_inf: float = 1.0,
        omega_p: float = 1.0,
        gamma: float = 0.0,
    ) -> "MaterialSpec":
        return cls(
            kind="plasma_drude",
            eps_inf=float(eps_inf),
            omega_p=float(omega_p),
            gamma=float(gamma),
        )


def rotate_tensor(tensor, orientation):
    """Rotate a 3x3 tensor from principal axes into lab coordinates."""
    tensor = np.asarray(tensor, dtype=np.float64)
    R = np.asarray(orientation, dtype=np.float64)
    if tensor.shape != (3, 3):
        raise ValueError(f"tensor must have shape (3,3), got {tensor.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"orientation must have shape (3,3), got {R.shape}")
    return R @ tensor @ R.T


def _apply_orientation(eps_real, eps_imag, orientation):
    if orientation is None:
        return eps_real, eps_imag
    return rotate_tensor(eps_real, orientation), rotate_tensor(eps_imag, orientation)


def material_eps(material: MaterialSpec, *, k0: float) -> tuple:
    """Return ``(eps_real, eps_imag)`` for an em3d refraction builder."""
    if material.kind == "plasma_drude":
        if k0 <= 0:
            raise ValueError(f"k0 must be positive for Drude model, got {k0}")
        eps = material.eps_inf - material.omega_p**2 / (k0**2 + 1j * material.gamma * k0)
        return float(np.real(eps)), float(np.imag(eps))
    if material.kind in {"isotropic", "anisotropic", "anisotropic_lossy"}:
        return _apply_orientation(material.eps_real, material.eps_imag, material.orientation)
    raise ValueError(f"unknown material kind {material.kind!r}")
