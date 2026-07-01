"""Helmholtz Green's function and discrete volume-integral kernel coefficients."""
from __future__ import annotations

import numpy as np


def green_helmholtz(R, k: float, eps: float = 1e-30):
    """G(R) = exp(i k R) / (4π R) with a scalar regularisation at R=0.

    Accepts scalar or ndarray. Mirrors the notebook `kernel` behaviour.

    WARNING: Do NOT call this at R=0 expecting a physical self-interaction
    value — the `eps` regularisation is a numerical guard only and returns
    an unphysically large number (~1/(4π·eps)).  Use `b_coeff(x, x, ...)`
    for self-cell contributions; it applies the correct excluded-sphere
    integral instead.
    """
    R_reg = np.where(np.asarray(R) < eps, eps, R)
    return np.exp(1j * k * R_reg) / (4.0 * np.pi * R_reg)


def b_coeff(x, y, *, k: float, dv: float):
    """Discrete 3x3 dyadic b-coefficient for the volume integral operator.

    For a pair of cell centres x, y ∈ R^3 with cell volume dv,
    returns the notebook dyadic Green block for x≠y:

        dv * G(R) * [C1(R) * alpha alpha^T + C2(R) * I],

    where R = |x-y|, alpha = (x-y)/R,
    C1 = 3/R^2 - 3ik/R - k^2, and C2 = k^2 + ik/R - 1/R^2.
    For x=y, returns the singular self-term limit -I/3.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    r = float(np.linalg.norm(diff))
    identity = np.eye(3, dtype=np.complex128)
    if r < 1e-15:
        return (-1.0 / 3.0) * identity
    alpha = diff / r
    alpha_outer = alpha[:, np.newaxis] * alpha[np.newaxis, :]
    coef_1 = (3.0 / (r * r)) - (3.0j * k / r) - (k * k)
    coef_2 = (k * k) + (1.0j * k / r) - (1.0 / (r * r))
    return green_helmholtz(r, k=k) * dv * (coef_1 * alpha_outer + coef_2 * identity)
