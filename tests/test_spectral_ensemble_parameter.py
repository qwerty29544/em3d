from em3d.backend import Backend
from em3d.experiments.spectral_transfer import (
    build_ensemble_parameter,
    build_spectral_case,
    compute_grid_spectrum,
    local_inclusion_case,
)


def test_ensemble_parameter_contains_selected_hulls_and_geometry_metadata():
    backend = Backend.numpy()
    definition = local_inclusion_case(0.5, k0=0.5)
    spectra = {
        level: compute_grid_spectrum(
            build_spectral_case(definition, grid_shape=level, backend=backend)
        )
        for level in (2, 3)
    }
    parameter = build_ensemble_parameter(spectra, (2, 3), control=spectra[3])
    assert parameter.levels == (2, 3)
    assert parameter.circle is not None
    assert parameter.maximum_resolution == max(parameter.resolution_by_level.values())
    assert parameter.control_assessment is not None
    assert parameter.control_assessment.target_factor <= 1.0 + 1e-8
