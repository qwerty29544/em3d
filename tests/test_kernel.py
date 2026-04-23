import numpy as np
import pytest

from em3d.kernel import green_helmholtz, b_coeff


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


def test_b_coeff_symmetry():
    # b_coeff(x, y) should equal b_coeff(y, x) by translational symmetry
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.6, 1.0])
    dv = 0.01
    k = 1.0
    assert abs(b_coeff(x, y, k=k, dv=dv) - b_coeff(y, x, k=k, dv=dv)) < 1e-12
