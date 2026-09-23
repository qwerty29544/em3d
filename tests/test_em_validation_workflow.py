from __future__ import annotations

import json

from experiments.em_validation import (
    MieStudyConfig,
    RuntimeConfig,
    SolverStudyConfig,
    ValidationStudyConfig,
    run_validation_suite,
)


def test_validation_workflow_writes_reproducible_artifacts(tmp_path):
    config = ValidationStudyConfig(
        mode="quick",
        runtime=RuntimeConfig(device="cpu", precision="double"),
        solver=SolverStudyConfig(
            case_keys=("anisotropic_ellipsoid",),
            grid_size=6,
            coarse_size=3,
            solver_names=("SIM", "BiCGStab", "TwoStep"),
            max_iter=100,
            rtol=1e-5,
            rcs_n_phi=24,
            rcs_planes=("xz",),
            save_solutions=False,
        ),
        mie=MieStudyConfig(
            grid_sizes=(6,),
            eps_values=(1.2 + 0.0j,),
            k0a_values=(0.5,),
            radius=0.25,
            domain_length=1.0,
            solver_names=("BiCGStab",),
            sim_coarse_size=3,
            max_iter=100,
            rtol=1e-5,
            rcs_n_phi=24,
            rcs_plane="xz",
            field_reference_grids=(6,),
            compare_farfield_backends_on_smallest_grid=True,
            save_solutions=False,
        ),
        output_root=tmp_path,
    )

    result = run_validation_suite(config, render_figures=False)

    assert len(result.stationary_cases) == 1
    assert len(result.mie_cases) == 1
    assert result.stationary_cases[0].reference_solver is not None
    assert result.mie_cases[0].validations
    assert result.mie_cases[0].rcs_curves
    assert result.mie_cases[0].field_slices
    # The publication workflow releases full Mie solutions after metrics and
    # compact curve data have been recorded.
    assert result.mie_cases[0].executions[0].result.u is None
    assert result.mie_cases[0].executions[0].solution_host is None

    for relative_path in (
        "tables/solver_runs.csv",
        "tables/field_comparisons.csv",
        "tables/rcs_solver_comparisons.csv",
        "tables/mie_solver_runs.csv",
        "tables/mie_validation.csv",
        "tables/mie_validation_selected.csv",
        "tables/FINAL_experimental_protocol.csv",
    ):
        assert (tmp_path / relative_path).is_file()

    mie_header = (tmp_path / "tables" / "mie_validation.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "nominal_rcs_peak_angle_error_degrees" in mie_header

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "em3d-stationary-validation-artifacts-v1"
    assert manifest["status"] == "complete"
    assert manifest["artifacts"]
