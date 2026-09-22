import numpy as np
import pytest

from em3d.spectral import (
    analyze_spectrum,
    assess_transfer,
    build_ensemble_localization,
)


def test_transfer_bound_and_certificate():
    coarse = analyze_spectrum(np.array([2 + 0j, 3 + 0.3j, 4 + 0j]))
    target = analyze_spectrum(np.array([2.05 + 0j, 3 + 0.32j, 3.95 + 0j]))
    assessment = assess_transfer(coarse, target)
    assert assessment.target_factor <= assessment.target_factor_bound + 1e-12
    assert assessment.directed_error >= 0.0
    assert assessment.certified == (
        assessment.directed_error < assessment.coarse_margin
    )


def test_ensemble_contains_each_input_hull():
    first = analyze_spectrum(np.array([2 + 0j, 3 + 0.5j, 4 + 0j]))
    second = analyze_spectrum(np.array([2.2 - 0.1j, 3 + 0.7j, 4.1 + 0.1j]))
    ensemble = build_ensemble_localization({4: first, 6: second})
    assert ensemble.levels == (4, 6)
    assert ensemble.localization.circle is not None
    assert len(ensemble.adjacent_directed_errors) == 1
    for point in np.concatenate([first.hull, second.hull]):
        # The ensemble is the convex hull of all source vertices.
        from em3d.spectral import point_in_convex_polygon

        assert point_in_convex_polygon(point, ensemble.localization.hull)
