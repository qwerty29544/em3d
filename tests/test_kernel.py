import numpy as np
import pytest

from em3d.kernel import green_helmholtz, b_coeff


def _expected_dyadic_b_coeff(x, y, *, k: float, dv: float):
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    R = float(np.linalg.norm(diff))
    if R < 1e-15:
        return (-1.0 / 3.0) * np.eye(3, dtype=np.complex128)
    alpha = diff / R
    alpha_outer = alpha[:, np.newaxis] * alpha[np.newaxis, :]
    identity = np.eye(3, dtype=np.complex128)
    gr = np.exp(1j * k * R) / (4.0 * np.pi * R)
    coef_1 = (3.0 / (R * R) - 3.0j * k / R - k * k) * alpha_outer
    coef_2 = (k * k + 1.0j * k / R - 1.0 / (R * R)) * identity
    return gr * dv * (coef_1 + coef_2)


def test_green_values_at_known_points():
    # G(R) = exp(i k R) / (4π R)
    R = 2.0
    k = 1.0
    expected = np.exp(1j * k * R) / (4.0 * np.pi * R)
    got = green_helmholtz(R, k)
    assert abs(got - expected) < 1e-12


def test_green_array_broadcast():
    R = np.array([1.0, 2.0, 3.0])
    k = 0.5
    got = green_helmholtz(R, k)
    expected = np.exp(1j * k * R) / (4.0 * np.pi * R)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_green_regularises_zero():
    # We do NOT evaluate at R=0; kernel must either raise or return a well-defined
    # regularised value for scalar R=0. We test that finite output emerges.
    out = green_helmholtz(0.0, k=1.0)
    assert np.isfinite(out.real) and np.isfinite(out.imag)


def test_b_coeff_returns_dyadic_tensor_for_generic_offset():
    """b_coeff must match the original dyadic Green tensor, including off-diagonal blocks."""
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([0.2, 0.3, 0.0])
    dv = 0.01
    k = 1.0
    got = b_coeff(x, y, k=k, dv=dv)
    expected = _expected_dyadic_b_coeff(x, y, k=k, dv=dv)
    assert got.shape == (3, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert abs(got[0, 1]) > 0.0


def test_b_coeff_self_term_is_minus_one_third_identity():
    """The dyadic singular self-term is the notebook limit -I/3."""
    x = np.array([0.1, 0.2, 0.3])
    got = b_coeff(x, x, k=1.0, dv=0.01)
    np.testing.assert_allclose(got, (-1.0 / 3.0) * np.eye(3), atol=1e-14)


def test_b_coeff_symmetry():
    # b_coeff(x, y) should equal b_coeff(y, x) by dyadic translational symmetry.
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.6, 1.0])
    dv = 0.01
    k = 1.0
    np.testing.assert_allclose(b_coeff(x, y, k=k, dv=dv), b_coeff(y, x, k=k, dv=dv), atol=1e-12)
