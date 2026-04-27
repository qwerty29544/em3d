"""em3d.mie — Analytical Mie scattering solution for a dielectric sphere.

Public API
----------
mie_coefficients(a, eps_r, k0)    -> dict with a, b, c, d arrays + n_max
mie_cross_sections(a, eps_r, k0)  -> dict with scat, ext, abs
mie_rcs_plane(a, eps_r, k0, ...)  -> (phi, sigma)
mie_field_at(xyz, a, eps_r, k0, amplitude, orient) -> (M, 3) complex
mie_field(grid, a, eps_r, k0, amplitude, orient)   -> (3, Nx, Ny, Nz) complex
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import spherical_jn, spherical_yn

__all__ = [
    "mie_coefficients",
    "mie_cross_sections",
    "mie_rcs_plane",
    "mie_field_at",
    "mie_field",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _n_max(x: float) -> int:
    """Wiscombe truncation order: round(x + 4*x^(1/3) + 2), minimum 1."""
    return max(1, round(float(x) + 4.0 * float(x) ** (1.0 / 3.0) + 2.0))


def _riccati(n: int, z: complex) -> tuple:
    """Return (psi_n(z), psi_n'(z), xi_n(z), xi_n'(z)).

    psi_n(z)  = z * j_n(z)
    psi_n'(z) = j_n(z) + z * j_n'(z)
    xi_n(z)   = z * h_n^(1)(z)  = z * (j_n(z) + i*y_n(z))
    xi_n'(z)  = h_n^(1)(z) + z * dh_n^(1)/dz
    """
    jn = spherical_jn(n, z)
    djn = spherical_jn(n, z, derivative=True)
    yn = spherical_yn(n, z)
    dyn = spherical_yn(n, z, derivative=True)

    psi = z * jn
    dpsi = jn + z * djn
    hn = jn + 1j * yn
    dhn = djn + 1j * dyn
    xi = z * hn
    dxi = hn + z * dhn
    return psi, dpsi, xi, dxi


def _validate_inputs(a: float, eps_r: complex, k0: float) -> None:
    if a <= 0:
        raise ValueError(f"a must be > 0, got {a}")
    if k0 <= 0:
        raise ValueError(f"k0 must be > 0, got {k0}")
    if np.real(eps_r) <= 0 and np.imag(eps_r) == 0:
        raise ValueError(f"Re(eps_r) must be > 0 for lossless medium, got {eps_r}")


# ---------------------------------------------------------------------------
# Public: coefficients
# ---------------------------------------------------------------------------

def mie_coefficients(a: float, eps_r: complex, k0: float) -> dict:
    """Compute Mie scattering coefficients for a dielectric sphere.

    Parameters
    ----------
    a     : float   — sphere radius (> 0)
    eps_r : complex — relative permittivity of sphere
    k0    : float   — free-space wave number (> 0)

    Returns
    -------
    dict:
        "a"     : ndarray (n_max,) complex128 — external TM coefficients
        "b"     : ndarray (n_max,) complex128 — external TE coefficients
        "c"     : ndarray (n_max,) complex128 — internal TM coefficients
        "d"     : ndarray (n_max,) complex128 — internal TE coefficients
        "n_max" : int — truncation order
    """
    _validate_inputs(a, eps_r, k0)
    x = k0 * a
    m = np.sqrt(complex(eps_r))
    if x > 10:
        warnings.warn(
            f"mie_coefficients: size parameter x={x:.2f} > 10; "
            f"Riccati-Bessel series may lose accuracy — consider logarithmic recurrence",
            UserWarning,
            stacklevel=3,
        )
    nmax = _n_max(x)
    an = np.empty(nmax, dtype=np.complex128)
    bn = np.empty(nmax, dtype=np.complex128)
    cn = np.empty(nmax, dtype=np.complex128)
    dn = np.empty(nmax, dtype=np.complex128)

    for idx in range(nmax):
        n = idx + 1
        psi_mx, dpsi_mx, _, _ = _riccati(n, m * x)
        psi_x, dpsi_x, xi_x, dxi_x = _riccati(n, x)

        an[idx] = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (
            m * psi_mx * dxi_x - xi_x * dpsi_mx
        )
        bn[idx] = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / (
            psi_mx * dxi_x - m * xi_x * dpsi_mx
        )
        cn[idx] = 1j / (m * psi_mx * dxi_x - xi_x * dpsi_mx)
        dn[idx] = 1j * m / (psi_mx * dxi_x - m * xi_x * dpsi_mx)

    return {"a": an, "b": bn, "c": cn, "d": dn, "n_max": nmax}


# ---------------------------------------------------------------------------
# Public: cross sections
# ---------------------------------------------------------------------------

def mie_cross_sections(a: float, eps_r: complex, k0: float) -> dict:
    """Integral scattering, extinction, and absorption cross sections.

    Parameters
    ----------
    a     : float   — sphere radius (> 0)
    eps_r : complex — relative permittivity
    k0    : float   — free-space wave number (> 0)

    Returns
    -------
    dict: {"scat": float, "ext": float, "abs": float}
    """
    coeffs = mie_coefficients(a, eps_r, k0)
    an, bn, nmax = coeffs["a"], coeffs["b"], coeffs["n_max"]
    ns = np.arange(1, nmax + 1, dtype=np.float64)
    weights = 2 * ns + 1
    sigma_scat = float(np.real(
        (2 * np.pi / k0**2) * np.sum(weights * (np.abs(an)**2 + np.abs(bn)**2))
    ))
    sigma_ext = float(
        (2 * np.pi / k0**2) * np.sum(weights * np.real(an + bn))
    )
    sigma_abs = sigma_ext - sigma_scat
    return {"scat": sigma_scat, "ext": sigma_ext, "abs": sigma_abs}


# ---------------------------------------------------------------------------
# Private: angular functions
# ---------------------------------------------------------------------------

def _angle_functions(n_max: int, cos_theta: np.ndarray) -> tuple:
    """Compute pi_n(cos_theta) and tau_n(cos_theta) for n = 1..n_max.

    Recurrence:
        pi_1 = 1
        pi_n = ((2n-1)/(n-1)) * cos_theta * pi_{n-1} - (n/(n-1)) * pi_{n-2}
        tau_n = n * cos_theta * pi_n - (n+1) * pi_{n-1}

    Parameters
    ----------
    n_max     : int — truncation order
    cos_theta : ndarray (M,) float — cosines of scattering angles

    Returns
    -------
    pi_arr  : ndarray (n_max, M) float64
    tau_arr : ndarray (n_max, M) float64
    """
    M = len(cos_theta)
    pi_arr = np.zeros((n_max, M), dtype=np.float64)
    tau_arr = np.zeros((n_max, M), dtype=np.float64)

    pi_prev2 = np.zeros(M, dtype=np.float64)   # pi_{n-2}
    pi_prev1 = np.ones(M, dtype=np.float64)    # pi_1 = 1

    for idx in range(n_max):
        n = idx + 1
        if n == 1:
            pi_n = pi_prev1.copy()
        else:
            pi_n = ((2*n - 1) / (n - 1)) * cos_theta * pi_prev1 - (n / (n - 1)) * pi_prev2
        tau_n = n * cos_theta * pi_n - (n + 1) * pi_prev1
        pi_arr[idx] = pi_n
        tau_arr[idx] = tau_n
        pi_prev2 = pi_prev1
        pi_prev1 = pi_n

    return pi_arr, tau_arr


# ---------------------------------------------------------------------------
# Public: bistatic RCS in a coordinate plane
# ---------------------------------------------------------------------------

def mie_rcs_plane(
    a: float,
    eps_r: complex,
    k0: float,
    n_phi: int = 180,
    plane: str = "xy",
) -> tuple:
    """Bistatic RCS over n_phi equally-spaced directions in a coordinate plane.

    Mirrors the interface of em3d.farfield.rcs_plane.
    Incident wave: E ∥ x̂, propagation ∥ ẑ.

    Parameters
    ----------
    a     : float   — sphere radius (> 0)
    eps_r : complex — relative permittivity
    k0    : float   — free-space wave number (> 0)
    n_phi : int     — number of angles in [0, 2π)
    plane : str     — "xy" | "xz" | "yz"

    Returns
    -------
    (phi, sigma) : both ndarray (n_phi,) float64
    """
    _validate_inputs(a, eps_r, k0)
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")

    coeffs = mie_coefficients(a, eps_r, k0)
    an, bn, nmax = coeffs["a"], coeffs["b"], coeffs["n_max"]
    ns = np.arange(1, nmax + 1, dtype=np.float64)
    weights = (2 * ns + 1) / (ns * (ns + 1))  # (n_max,)

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    if plane == "xy":
        # theta_sph = pi/2 for all phi (transverse to propagation ẑ)
        cos_theta = np.zeros(n_phi, dtype=np.float64)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
        # S1 and S2 at theta=pi/2
        S1 = np.einsum("n,nm->m", weights * an, pi_arr) + np.einsum("n,nm->m", weights * bn, tau_arr)
        S2 = np.einsum("n,nm->m", weights * an, tau_arr) + np.einsum("n,nm->m", weights * bn, pi_arr)
        sigma = (np.abs(S2)**2 * np.cos(phi)**2 + np.abs(S1)**2 * np.sin(phi)**2) / k0**2

    elif plane == "xz":
        # direction (cos(phi), 0, sin(phi)): phi_sph=0, theta_sph = pi/2 - phi
        theta_sph = np.pi / 2.0 - phi
        cos_theta = np.cos(theta_sph)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
        # phi_sph=0 → cos²(phi_sph)=1, sin²(phi_sph)=0 → sigma = |S2|²/k0²
        S2 = np.einsum("n,nm->m", weights * an, tau_arr) + np.einsum("n,nm->m", weights * bn, pi_arr)
        sigma = np.abs(S2)**2 / k0**2

    else:  # "yz"
        # direction (0, cos(phi), sin(phi)): phi_sph=pi/2, theta_sph = pi/2 - phi
        theta_sph = np.pi / 2.0 - phi
        cos_theta = np.cos(theta_sph)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
        # phi_sph=pi/2 → cos²=0, sin²=1 → sigma = |S1|²/k0²
        S1 = np.einsum("n,nm->m", weights * an, pi_arr) + np.einsum("n,nm->m", weights * bn, tau_arr)
        sigma = np.abs(S1)**2 / k0**2

    return phi, np.real(sigma).astype(np.float64)
