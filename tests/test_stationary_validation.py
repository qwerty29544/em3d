from __future__ import annotations

import numpy as np

import em3d
from em3d.experiments.spectral_transfer import build_spectral_case
from em3d.experiments.stationary_validation import (
    RCSCurve,
    compare_fields,
    compare_rcs_curves,
    compute_rcs_curve,
    estimate_parameter_circle,
    run_solver_comparison,
    select_reference_solver,
)
from em3d.geometry import SamplingMode
from experiments.em_validation.cases import stationary_case_catalog


def _small_case():
    return stationary_case_catalog()["anisotropic_ellipsoid"]


def test_solver_comparison_uses_true_residual_and_action_counts():
    definition = _small_case()
    built = build_spectral_case(
        definition,
        grid_shape=6,
        backend=em3d.Backend.numpy(),
        sampling_mode=SamplingMode.CELL_CENTER,
    )
    circle = estimate_parameter_circle(definition, coarse_grid_shape=3)
    executions = run_solver_comparison(
        built,
        solver_names=("SIM", "BiCGStab", "TwoStep"),
        solver_config=em3d.SolverConfig(max_iter=100, rtol=1e-5),
        circle=circle,
        retain_solutions=True,
    )

    assert {execution.solver_name for execution in executions} == {
        "SIM",
        "BiCGStab",
        "TwoStep",
    }
    for execution in executions:
        result = execution.result
        assert execution.qualified
        assert result.status == "converged"
        assert result.residual_action_counts[-1] == result.operator_action_count
        assert len(result.residual_action_counts) == len(result.residual_history)
        assert np.all(np.diff(result.residual_action_counts) >= 0)
        np.testing.assert_allclose(
            result.residual_history[-1],
            execution.true_relative_residual,
            rtol=1e-10,
            atol=1e-12,
        )

    reference = select_reference_solver(executions)
    for execution in executions:
        metric = compare_fields(execution, reference)
        assert metric.relative_l2 < 5e-5
        curve = compute_rcs_curve(execution, built, n_phi=24, plane="xz")
        reference_curve = compute_rcs_curve(reference, built, n_phi=24, plane="xz")
        rcs_metric = compare_rcs_curves(curve, reference_curve)
        assert rcs_metric.normalized_l2 < 5e-5
        assert rcs_metric.absolute_l2 < 5e-5


def test_nonconverged_sim_is_excluded_from_physical_postprocessing():
    definition = stationary_case_catalog()["local_inclusion_stress"]
    built = build_spectral_case(
        definition,
        grid_shape=12,
        backend=em3d.Backend.numpy(),
        sampling_mode=SamplingMode.CELL_CENTER,
    )
    circle = estimate_parameter_circle(definition, coarse_grid_shape=5)
    executions = run_solver_comparison(
        built,
        solver_names=("SIM", "BiCGStab", "TwoStep"),
        solver_config=em3d.SolverConfig(
            max_iter=80,
            rtol=1e-6,
            divergence_guard=1e6,
        ),
        circle=circle,
        retain_solutions=True,
    )

    by_name = {execution.solver_name: execution for execution in executions}
    assert not by_name["SIM"].qualified
    assert by_name["SIM"].result.status == "max_iter"
    assert by_name["BiCGStab"].qualified
    assert by_name["TwoStep"].qualified

    reference = select_reference_solver(executions)
    assert reference.solver_name == "BiCGStab"
    physically_admissible = [
        execution.solver_name for execution in executions if execution.qualified
    ]
    assert physically_admissible == ["BiCGStab", "TwoStep"]


def test_rcs_peak_angle_metric_handles_symmetric_degenerate_maxima():
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    reference_sigma = 1.0 + np.cos(2.0 * phi)
    candidate_sigma = reference_sigma.copy()
    # Make the opposite member of the symmetric peak pair infinitesimally larger.
    candidate_sigma[18] += 1e-12
    reference_sigma[0] += 1e-12

    def curve(name, sigma):
        peak = float(np.max(sigma))
        return RCSCurve(
            solver_name=name,
            plane="xy",
            phi=phi,
            sigma=sigma,
            sigma_normalized=sigma / peak,
            peak=peak,
            angular_integral=float(2.0 * np.pi * np.mean(sigma)),
        )

    metric = compare_rcs_curves(
        curve("candidate", candidate_sigma),
        curve("reference", reference_sigma),
    )
    assert metric.normalized_l2 < 1e-10
    assert metric.peak_angle_error_degrees == 0.0
