import numpy as np
import pytest

from em3d.spectral import (
    LocalizationStatus,
    analyze_spectrum,
    circle_contains_points,
    spectral_factor,
)


def test_localization_returns_margin_and_factor():
    spectrum = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    result = analyze_spectrum(spectrum)
    assert result.status is LocalizationStatus.OK
    assert result.circle is not None
    assert result.circle.mu == pytest.approx(2 + 0j)
    assert result.circle.radius == pytest.approx(1.0)
    assert result.circle.margin == pytest.approx(1.0)
    assert result.circle.q == pytest.approx(0.5)
    assert spectral_factor(spectrum, result.circle.mu) == pytest.approx(0.5)
    assert circle_contains_points(
        result.circle.mu, result.circle.radius, result.hull
    )


def test_origin_in_hull_is_reported_without_parameter():
    spectrum = np.array([-1 + 0j, 1 + 0j, 0 + 1j])
    result = analyze_spectrum(spectrum)
    assert result.status is LocalizationStatus.ORIGIN_IN_HULL
    assert result.origin_in_hull
    assert result.origin_distance == 0.0
    assert result.circle is None


def test_degenerate_spectrum_is_reported():
    result = analyze_spectrum(np.array([1 + 1j, 1 + 1j]))
    assert result.status is LocalizationStatus.DEGENERATE_SPECTRUM
    assert result.circle is None
