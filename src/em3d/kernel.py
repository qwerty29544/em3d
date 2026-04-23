"""Helmholtz Green's function and discrete volume-integral kernel coefficients."""
from __future__ import annotations

import numpy as np


def green_helmholtz(R, k: float, eps: float = 1e-30):
    """G(R) = exp(i k R) / (4π R) with a scalar regularisation at R=0.

    Accepts scalar or ndarray. Mirrors the notebook `kernel` behaviour.
    """
    R_reg = np.where(np.asarray(R) < eps, eps, R)
    return np.exp(1j * k * R_reg) / (4.0 * np.pi * R_reg)


def b_coeff(x, y, *, k: float, dv: float):
    """Discrete b-coefficient for the volume integral operator.

    For a pair of cell centres x, y ∈ R^3 with cell volume dv,
    returns dv · G(|x-y|, k) per collocation. See notebook cell ~654.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.linalg.norm(x - y)
    return dv * green_helmholtz(r, k=k)
