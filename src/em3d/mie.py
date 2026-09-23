from __future__ import annotations
 
import warnings
 
import numpy as np
from scipy.special import spherical_jn, spherical_yn
 
__all__ = [
    "mie_coefficients",
    "mie_cross_sections",
    "mie_rcs_plane",
    "compare_rcs_plane",
    "mie_field_at",
    "mie_field",
    "mie_nearfield_farfield_consistency",
]
 
def _n_max(x: float) -> int:
 
    return max(1, round(float(x) + 4.0 * float(x) ** (1.0 / 3.0) + 2.0))
 
def _riccati(n: int, z: complex) -> tuple:
 
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
 
def mie_coefficients(a: float, eps_r: complex, k0: float) -> dict:
 
    _validate_inputs(a, eps_r, k0)
    x = k0 * a
    m = np.sqrt(complex(eps_r))
    if abs(complex(eps_r) - 1.0) <= 1e-14:
        nmax = _n_max(x)
        zeros = np.zeros(nmax, dtype=np.complex128)
        ones = np.ones(nmax, dtype=np.complex128)
        return {"a": zeros.copy(), "b": zeros.copy(), "c": ones.copy(), "d": ones.copy(), "n_max": nmax}
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
        den_a = m * psi_mx * dxi_x - xi_x * dpsi_mx
        den_b = psi_mx * dxi_x - m * xi_x * dpsi_mx
 
        an[idx] = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / den_a
        bn[idx] = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / den_b
        cn[idx] = 1j * m / den_b
        dn[idx] = 1j * m / den_a
 
    return {"a": an, "b": bn, "c": cn, "d": dn, "n_max": nmax}
 
def mie_cross_sections(a: float, eps_r: complex, k0: float) -> dict:
 
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
 
def _angle_functions(n_max: int, cos_theta: np.ndarray) -> tuple:
 
    M = len(cos_theta)
    pi_arr = np.zeros((n_max, M), dtype=np.float64)
    tau_arr = np.zeros((n_max, M), dtype=np.float64)
 
    pi_prev2 = np.zeros(M, dtype=np.float64)
    pi_prev1 = np.ones(M, dtype=np.float64)
 
    for idx in range(n_max):
        n = idx + 1
        if n == 1:
            pi_n = pi_prev1.copy()
            pi_n_minus_1 = pi_prev2
        else:
            pi_n = ((2*n - 1) / (n - 1)) * cos_theta * pi_prev1 - (n / (n - 1)) * pi_prev2
            pi_n_minus_1 = pi_prev1
        tau_n = n * cos_theta * pi_n - (n + 1) * pi_n_minus_1
        pi_arr[idx] = pi_n
        tau_arr[idx] = tau_n
        pi_prev2 = pi_n_minus_1
        pi_prev1 = pi_n
 
    return pi_arr, tau_arr
 
def mie_rcs_plane(
    a: float,
    eps_r: complex,
    k0: float,
    n_phi: int = 180,
    plane: str = "xy",
) -> tuple:
 
    _validate_inputs(a, eps_r, k0)
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")
 
    coeffs = mie_coefficients(a, eps_r, k0)
    an, bn, nmax = coeffs["a"], coeffs["b"], coeffs["n_max"]
    ns = np.arange(1, nmax + 1, dtype=np.float64)
    weights = (2 * ns + 1) / (ns * (ns + 1))
 
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
 
    if plane == "xy":
 
        cos_theta = np.zeros(n_phi, dtype=np.float64)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
 
        S1 = np.einsum("n,nm->m", weights * an, pi_arr) + np.einsum("n,nm->m", weights * bn, tau_arr)
        S2 = np.einsum("n,nm->m", weights * an, tau_arr) + np.einsum("n,nm->m", weights * bn, pi_arr)
        sigma = (np.abs(S2)**2 * np.cos(phi)**2 + np.abs(S1)**2 * np.sin(phi)**2) / k0**2
 
    elif plane == "xz":
 
        theta_sph = np.pi / 2.0 - phi
        cos_theta = np.cos(theta_sph)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
 
        S2 = np.einsum("n,nm->m", weights * an, tau_arr) + np.einsum("n,nm->m", weights * bn, pi_arr)
        sigma = np.abs(S2)**2 / k0**2
 
    else:
 
        theta_sph = np.pi / 2.0 - phi
        cos_theta = np.cos(theta_sph)
        pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
 
        S1 = np.einsum("n,nm->m", weights * an, pi_arr) + np.einsum("n,nm->m", weights * bn, tau_arr)
        sigma = np.abs(S1)**2 / k0**2
 
    return phi, np.real(sigma).astype(np.float64)
 
def _normalize_rcs_curve(sigma: np.ndarray) -> tuple[np.ndarray, float]:
 
    peak = float(np.max(sigma)) if sigma.size else 0.0
    if peak > 0.0:
        return sigma / peak, peak
    return np.zeros_like(sigma, dtype=np.float64), peak
 
def compare_rcs_plane(
    u,
    problem,
    *,
    a: float,
    eps_r: complex,
    n_phi: int = 180,
    plane: str = "xy",
    method: str = "direct",
    batch_size: int = 64,
    normalize: str = "max",
) -> dict:
 
    if normalize != "max":
        raise ValueError(f"normalize must be 'max', got {normalize!r}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")
 
    from . import farfield
 
    phi_num, sigma_num = farfield.rcs_plane(
        u,
        problem,
        n_phi=n_phi,
        plane=plane,
        method=method,
        batch_size=batch_size,
    )
    phi_mie, sigma_mie = mie_rcs_plane(a, eps_r, problem.k0, n_phi=n_phi, plane=plane)
 
    phi_num = np.asarray(phi_num, dtype=np.float64)
    sigma_num = np.asarray(sigma_num, dtype=np.float64)
    sigma_mie = np.asarray(sigma_mie, dtype=np.float64)
    if not np.allclose(phi_num, phi_mie, rtol=0.0, atol=1e-15):
        raise RuntimeError("numerical and Mie RCS angle grids differ")
 
    sigma_num_norm, sigma_num_peak = _normalize_rcs_curve(sigma_num)
    sigma_mie_norm, sigma_mie_peak = _normalize_rcs_curve(sigma_mie)
 
    shape_den = float(np.linalg.norm(sigma_mie_norm))
    if shape_den > 0.0:
        shape_err = float(np.linalg.norm(sigma_num_norm - sigma_mie_norm) / shape_den)
    else:
        shape_err = 0.0 if np.linalg.norm(sigma_num_norm) == 0.0 else float("inf")
 
    if sigma_mie_peak > 0.0:
        scale_ratio = float(sigma_num_peak / sigma_mie_peak)
        abs_rel_err = float(np.max(np.abs(sigma_num - sigma_mie)) / sigma_mie_peak)
    else:
        scale_ratio = float("nan")
        abs_rel_err = 0.0 if sigma_num_peak == 0.0 else float("inf")
 
    return {
        "phi": phi_num,
        "sigma_num": sigma_num,
        "sigma_mie": sigma_mie,
        "sigma_num_norm": sigma_num_norm,
        "sigma_mie_norm": sigma_mie_norm,
        "shape_err": shape_err,
        "scale_ratio": scale_ratio,
        "abs_rel_err": abs_rel_err,
    }
 
def _build_frame(orient, amplitude) -> tuple:
 
    orient = np.asarray(orient, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if orient.shape != (3,):
        raise ValueError(f"orient must have shape (3,), got shape={orient.shape}")
    if amplitude.shape != (3,):
        raise ValueError(f"amplitude must have shape (3,), got shape={amplitude.shape}")
    orient_norm = np.linalg.norm(orient)
    if orient_norm == 0.0:
        raise ValueError("orient must be non-zero")
    amplitude_norm = np.linalg.norm(amplitude)
    if amplitude_norm == 0.0:
        raise ValueError("amplitude must be non-zero")
    z_can = orient / orient_norm
    amp_perp = amplitude - np.dot(amplitude, z_can) * z_can
    amp_perp_norm = np.linalg.norm(amp_perp)
    if amp_perp_norm < 1e-12 * amplitude_norm:
        raise ValueError("amplitude must not be parallel to orient")
    if abs(np.dot(amplitude, z_can)) > 1e-12 * amplitude_norm:
        raise ValueError("amplitude must be transverse to orient")
    x_can = amplitude / amplitude_norm
    y_can = np.cross(z_can, x_can)
    R = np.column_stack([x_can, y_can, z_can])
    E0 = float(amplitude_norm)
    return R, E0
 
def _field_at_canonical(
    xyz_can: np.ndarray,
    a: float,
    k0: float,
    m: complex,
    coeffs: dict,
) -> np.ndarray:
 
    an, bn, cn, dn, nmax = (
        coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"], coeffs["n_max"]
    )
    k_int = m * k0
    M = xyz_can.shape[0]
    E = np.zeros((M, 3), dtype=np.complex128)
 
    x_can = xyz_can[:, 0]
    y_can = xyz_can[:, 1]
    z_can = xyz_can[:, 2]
    r = np.sqrt(x_can**2 + y_can**2 + z_can**2)
 
    zero_mask = (r == 0.0)
    E[zero_mask, 0] = dn[0]
 
    non_zero = ~zero_mask
    if not np.any(non_zero):
        return E
 
    r_nz = r[non_zero]
    x_nz = x_can[non_zero]
    y_nz = y_can[non_zero]
    z_nz = z_can[non_zero]
    rho_nz = np.sqrt(x_nz**2 + y_nz**2)
 
    cos_theta = z_nz / r_nz
    sin_theta = rho_nz / r_nz
    cos_phi = np.ones_like(rho_nz)
    sin_phi = np.zeros_like(rho_nz)
    rho_positive = rho_nz > 0
    cos_phi[rho_positive] = x_nz[rho_positive] / rho_nz[rho_positive]
    sin_phi[rho_positive] = y_nz[rho_positive] / rho_nz[rho_positive]
 
    pi_arr, tau_arr = _angle_functions(nmax, cos_theta)
 
    interior = r_nz < a
    exterior = ~interior
 
    if np.any(interior):
        ri = r_nz[interior]
        cth_i = cos_theta[interior]
        sth_i = sin_theta[interior]
        cph_i = cos_phi[interior]
        sph_i = sin_phi[interior]
        pi_i = pi_arr[:, interior]
        tau_i = tau_arr[:, interior]
        Mi = ri.shape[0]
        Er_i = np.zeros(Mi, dtype=np.complex128)
        Eth_i = np.zeros(Mi, dtype=np.complex128)
        Eph_i = np.zeros(Mi, dtype=np.complex128)
 
        for idx in range(nmax):
            n = idx + 1
            prefactor = (1j**n) * (2*n + 1) / (n * (n + 1))
            k_int_r = k_int * ri
            jn_val = spherical_jn(n, k_int_r)
            djn_val = spherical_jn(n, k_int_r, derivative=True)
            psi_prime_over_kr = (jn_val + k_int_r * djn_val) / k_int_r
 
            M_eth = cph_i * pi_i[idx] * jn_val
            M_eph = -sph_i * tau_i[idx] * jn_val
 
            N_er = sth_i * cph_i * pi_i[idx] * n * (n + 1) * jn_val / k_int_r
            N_eth = cph_i * tau_i[idx] * psi_prime_over_kr
            N_eph = -sph_i * pi_i[idx] * psi_prime_over_kr
 
            cn_val = cn[idx]
            dn_val = dn[idx]
 
            Er_i  += prefactor * (-1j * dn_val * N_er)
            Eth_i += prefactor * (cn_val * M_eth - 1j * dn_val * N_eth)
            Eph_i += prefactor * (cn_val * M_eph - 1j * dn_val * N_eph)
 
        Ex_i = sth_i*cph_i*Er_i + cth_i*cph_i*Eth_i - sph_i*Eph_i
        Ey_i = sth_i*sph_i*Er_i + cth_i*sph_i*Eth_i + cph_i*Eph_i
        Ez_i = cth_i*Er_i - sth_i*Eth_i
 
        idx_int = np.where(non_zero)[0][interior]
        E[idx_int, 0] = Ex_i
        E[idx_int, 1] = Ey_i
        E[idx_int, 2] = Ez_i
 
    if np.any(exterior):
        re = r_nz[exterior]
        cth_e = cos_theta[exterior]
        sth_e = sin_theta[exterior]
        cph_e = cos_phi[exterior]
        sph_e = sin_phi[exterior]
        z_e = z_nz[exterior]
        pi_e = pi_arr[:, exterior]
        tau_e = tau_arr[:, exterior]
        Me = re.shape[0]
        Er_sc = np.zeros(Me, dtype=np.complex128)
        Eth_sc = np.zeros(Me, dtype=np.complex128)
        Eph_sc = np.zeros(Me, dtype=np.complex128)
 
        for idx in range(nmax):
            n = idx + 1
            prefactor = (1j**n) * (2*n + 1) / (n * (n + 1))
            k0_r = k0 * re
            jn_val = spherical_jn(n, k0_r)
            djn_val = spherical_jn(n, k0_r, derivative=True)
            yn_val = spherical_yn(n, k0_r)
            dyn_val = spherical_yn(n, k0_r, derivative=True)
            hn_val = jn_val + 1j * yn_val
            dhn_val = djn_val + 1j * dyn_val
 
            xi_prime_over_kr = (hn_val + k0_r * dhn_val) / k0_r
 
            M_eth = cph_e * pi_e[idx] * hn_val
            M_eph = -sph_e * tau_e[idx] * hn_val
 
            N_er = sth_e * cph_e * pi_e[idx] * n * (n + 1) * hn_val / k0_r
            N_eth = cph_e * tau_e[idx] * xi_prime_over_kr
            N_eph = -sph_e * pi_e[idx] * xi_prime_over_kr
 
            an_val = an[idx]
            bn_val = bn[idx]
 
            Er_sc  += prefactor * (1j * an_val * N_er)
            Eth_sc += prefactor * (1j * an_val * N_eth - bn_val * M_eth)
            Eph_sc += prefactor * (1j * an_val * N_eph - bn_val * M_eph)
 
        Ex_sc = sth_e*cph_e*Er_sc + cth_e*cph_e*Eth_sc - sph_e*Eph_sc
        Ey_sc = sth_e*sph_e*Er_sc + cth_e*sph_e*Eth_sc + cph_e*Eph_sc
        Ez_sc = cth_e*Er_sc - sth_e*Eth_sc
 
        E_inc_x = np.exp(1j * k0 * z_e)
 
        idx_ext = np.where(non_zero)[0][exterior]
        E[idx_ext, 0] = E_inc_x + Ex_sc
        E[idx_ext, 1] = Ey_sc
        E[idx_ext, 2] = Ez_sc
 
    return E
 
def mie_field_at(
    xyz,
    a: float,
    eps_r: complex,
    k0: float,
    amplitude=(1, 0, 0),
    orient=(0, 0, 1),
) -> np.ndarray:
 
    _validate_inputs(a, eps_r, k0)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (M, 3), got shape={xyz.shape}")
    if abs(complex(eps_r) - 1.0) <= 1e-14:
        orient_array = np.asarray(orient, dtype=np.float64)
        amplitude_array = np.asarray(amplitude, dtype=np.complex128)
        phase = np.exp(1j * float(k0) * (xyz @ orient_array))
        return phase[:, None] * amplitude_array[None, :]
    R, E0 = _build_frame(orient, amplitude)
    m = np.sqrt(complex(eps_r))
    coeffs = mie_coefficients(a, eps_r, k0)
 
    xyz_can = xyz @ R
 
    E_can = _field_at_canonical(xyz_can, a, k0, m, coeffs)
    E_lab = E0 * (E_can @ R.T)
    return E_lab.astype(np.complex128)
 
def mie_field(
    grid,
    a: float,
    eps_r: complex,
    k0: float,
    amplitude=(1, 0, 0),
    orient=(0, 0, 1),
) -> np.ndarray:
 
    X, Y, Z = grid.coords()
    be = grid.backend
    X_np = np.asarray(be.to_host(X), dtype=np.float64)
    Y_np = np.asarray(be.to_host(Y), dtype=np.float64)
    Z_np = np.asarray(be.to_host(Z), dtype=np.float64)
    shape = X_np.shape
    xyz = np.stack([X_np.ravel(), Y_np.ravel(), Z_np.ravel()], axis=1)
    E_flat = mie_field_at(xyz, a, eps_r, k0, amplitude=amplitude, orient=orient)
    return E_flat.T.reshape(3, *shape)


def mie_nearfield_farfield_consistency(
    a: float,
    eps_r: complex,
    k0: float,
    *,
    plane: str = "xz",
    n_phi: int = 360,
    observation_radius: float | None = None,
) -> dict[str, object]:
    """Compare the large-radius near field with the analytical Mie RCS.

    The canonical incident wave propagates along ``+z`` and is polarized along
    ``+x``.  For the time convention used by :func:`mie_field_at`,
    ``r**2 * |E_sc|**2`` must tend to the plane RCS curve.  This diagnostic is
    intentionally independent of the VIE discretization and is used as a gate
    before numerical near-field figures are interpreted against Mie theory.
    """

    _validate_inputs(a, eps_r, k0)
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")
    if observation_radius is None:
        observation_radius = max(100.0 * a, 100.0 / k0)
    radius = float(observation_radius)
    if radius <= max(a, 0.0):
        raise ValueError("observation_radius must exceed the sphere radius")

    phi, sigma_reference = mie_rcs_plane(
        a, eps_r, k0, n_phi=n_phi, plane=plane
    )
    if plane == "xy":
        xyz = np.column_stack(
            [radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)]
        )
    elif plane == "xz":
        xyz = np.column_stack(
            [radius * np.cos(phi), np.zeros_like(phi), radius * np.sin(phi)]
        )
    else:
        xyz = np.column_stack(
            [np.zeros_like(phi), radius * np.cos(phi), radius * np.sin(phi)]
        )

    total = mie_field_at(xyz, a, eps_r, k0)
    incident = np.zeros_like(total)
    incident[:, 0] = np.exp(1j * k0 * xyz[:, 2])
    scattered = total - incident
    sigma_from_nearfield = radius**2 * np.sum(np.abs(scattered) ** 2, axis=1)

    denominator = float(np.linalg.norm(sigma_reference))
    relative_l2 = (
        float(np.linalg.norm(sigma_from_nearfield - sigma_reference)) / denominator
        if denominator > 0.0
        else float(np.linalg.norm(sigma_from_nearfield))
    )
    peak_reference = float(np.max(sigma_reference)) if sigma_reference.size else 0.0
    peak_ratio = (
        float(np.max(sigma_from_nearfield)) / peak_reference
        if peak_reference > 0.0
        else float("nan")
    )
    return {
        "plane": plane,
        "observation_radius": radius,
        "phi": np.asarray(phi, dtype=np.float64),
        "sigma_nearfield": np.asarray(sigma_from_nearfield, dtype=np.float64),
        "sigma_reference": np.asarray(sigma_reference, dtype=np.float64),
        "relative_l2": relative_l2,
        "peak_ratio": peak_ratio,
    }
