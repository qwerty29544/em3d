import numpy as np

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.experiments.spectral_transfer import (
    SpectralCaseDefinition,
    assess_control_transfer,
    build_spectral_case,
    check_operator_consistency,
    compute_grid_spectrum,
    diagnose_iteration_spectrum,
    run_parameter_transfer,
)
from em3d.geometry import FullDomain
from em3d.solvers import SolverConfig
from em3d.spectral import ArnoldiConfig, LocalizationStatus


def _case():
    return SpectralCaseDefinition(
        key="weak_homogeneous",
        title="weak homogeneous case",
        domain_lengths=(1.0, 1.0, 1.0),
        domain_center=(0.0, 0.0, 0.0),
        k0=0.5,
        background_eps_r=np.eye(3),
        feature_eps_r=np.diag([1.1, 1.08, 1.05]),
        feature_geometry=FullDomain(),
    )


def test_packaged_spectral_transfer_pipeline_on_small_grids():
    backend = Backend.numpy(Precision.DOUBLE)
    coarse_case = build_spectral_case(_case(), grid_shape=2, backend=backend)
    control_case = build_spectral_case(_case(), grid_shape=3, backend=backend)

    assert check_operator_consistency(coarse_case) < 1e-10
    coarse = compute_grid_spectrum(coarse_case)
    control = compute_grid_spectrum(control_case)
    assert coarse.localization.status is LocalizationStatus.OK
    assert control.localization.status is LocalizationStatus.OK

    transfer = assess_control_transfer(coarse, control)
    assert transfer.assessment.target_factor <= transfer.assessment.target_factor_bound + 1e-10

    circle = coarse.localization.circle
    assert circle is not None
    runs = run_parameter_transfer(
        control_case,
        {"N2": circle},
        solver_config=SolverConfig(max_iter=100, rtol=1e-8),
    )
    assert len(runs) == 1
    assert runs[0].solver_result.matvec_count > 0

    diagnostic = diagnose_iteration_spectrum(
        coarse_case,
        circle,
        config=ArnoldiConfig(krylov_dimension=6, milestones=(2, 4, 6)),
    )
    assert np.isfinite(diagnostic.spectral_radius)
