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
    """Discrete b-coefficient for the volume integral operator.

    For a pair of cell centres x, y ∈ R^3 with cell volume dv,
    returns dv · G(|x-y|, k) for x≠y, or the excluded-sphere self-interaction
    integral_0^{r0} exp(ikr)·r dr for x=y, where r0 = (3·dv/(4π))^{1/3}.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = float(np.linalg.norm(x - y))
    if r < 1e-15:
        # Excluded-sphere self-interaction: ∫₀^r₀ exp(ikr)·r dr, r0=(3dv/4π)^{1/3}.
        # No dv factor: the spherical integral already represents the full cell
        # contribution; dv is implicit through r0 = (3dv/4π)^{1/3}.
        r0 = (3.0 * dv / (4.0 * np.pi)) ** (1.0 / 3.0)
        if abs(k) < 1e-15:
            return r0 * r0 / 2.0
        return np.exp(1j * k * r0) * (r0 / (1j * k) - 1.0 / (k * k)) + 1.0 / (k * k)
    return dv * green_helmholtz(r, k=k)
