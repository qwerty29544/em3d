from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.chapter4_final.campaign import merge_chapter4_campaign
from experiments.em_spectral.artifacts import ArtifactStore
from experiments.em_validation.config import SolverSuiteConfig, StationaryGridJobSpec
from experiments.em_validation.planner import build_large_grid_config


def test_main64_contains_single_and_ensemble_stress_variants(tmp_path: Path):
    config = build_large_grid_config(
        "main64",
        output_root=tmp_path / "main64",
        device="cpu",
    )
    keys = {job.key for job in config.stationary_jobs}
    assert "local_inclusion_stress_single_N5_N64" in keys
    assert "local_inclusion_stress_ensemble_5_6_7_N64" in keys
    assert len(keys) == len(config.stationary_jobs)

    jobs = {job.key: job for job in config.stationary_jobs}
    strategy, levels = jobs[
        "local_inclusion_stress_single_N5_N64"
    ].resolved_parameter_strategy(config.solver_suite)
    assert strategy == "single_coarse"
    assert levels == (5,)
    strategy, levels = jobs[
        "local_inclusion_stress_ensemble_5_6_7_N64"
    ].resolved_parameter_strategy(config.solver_suite)
    assert strategy == "coarse_ensemble"
    assert levels == (5, 6, 7)


def test_audit_jobs_have_unique_keys(tmp_path: Path):
    config = build_large_grid_config(
        "audit",
        output_root=tmp_path / "audit",
        device="cpu",
    )
    keys = [job.key for job in config.mie_jobs]
    assert len(keys) == len(set(keys))
    assert all("audit_rtol1e-8" in key for key in keys)


def test_stationary_job_strategy_validation():
    with pytest.raises(ValueError):
        StationaryGridJobSpec(
            case_key="x",
            grid_size=8,
            parameter_strategy="single_coarse",
            coarse_sizes=(5, 6),
        )
    inherited = StationaryGridJobSpec(case_key="x", grid_size=8)
    assert inherited.resolved_parameter_strategy(SolverSuiteConfig()) == (
        "coarse_ensemble",
        (5, 6, 7),
    )


def _make_spectral_run(root: Path) -> None:
    store = ArtifactStore(root, schema="em3d-spectral-transfer-artifacts-v1")
    store.write_json("config.json", {"mode": "test"})
    store.write_rows(
        "tables/FINAL_experimental_protocol.csv",
        [{"experiment": "E0-E6", "executed": True}],
    )
    store.write_rows(
        "tables/E3_summary_phase.csv",
        [{"wave_number": 4.0, "stable": True}],
    )
    store.finalize(config={"mode": "test"})


def _make_validation_run(root: Path) -> None:
    store = ArtifactStore(
        root,
        schema="em3d-chapter4-all-solvers-large-grid-v1",
    )
    store.write_json("config.json", {"profile": "test"})
    common = [{
        "job_key": "mie_eps1p5_k0a1_N8",
        "eps_real": 1.5,
        "eps_imag": 0.0,
        "k0a": 1.0,
        "grid_size": 8,
        "sim_qualified": True,
        "bicgstab_qualified": True,
        "twostep_qualified": True,
        "all_solvers_qualified": True,
    }]
    runs = []
    for solver in ("SIM", "BiCGStab", "TwoStep"):
        runs.append({
            "job_key": "mie_eps1p5_k0a1_N8",
            "tier": "main",
            "eps_real": 1.5,
            "eps_imag": 0.0,
            "k0a": 1.0,
            "grid_size": 8,
            "solver_name": solver,
            "qualified": True,
            "true_final_residual": 1e-7,
            "operator_action_count": 10,
            "elapsed_seconds": 0.1,
        })
    store.write_rows("tables/mie_common_solvability.csv", common)
    store.write_rows("tables/mie_solver_runs.csv", runs)
    store.write_rows(
        "tables/mie_solver_vs_mie_rcs.csv",
        [{
            "tier": "main",
            "grid_size": 8,
            "solver_name": "BiCGStab",
            "radius_label": "effective",
            "normalized_l2": 0.01,
            "absolute_l2": 0.02,
            "absolute_linf": 0.03,
        }],
    )
    store.write_rows(
        "tables/mie_solver_vs_mie_fields.csv",
        [{
            "tier": "main",
            "grid_size": 8,
            "solver_name": "BiCGStab",
            "radius_label": "effective",
            "field_relative_l2": 0.04,
            "scattered_field_relative_l2": 0.05,
            "scattered_field_outside_relative_l2": 0.06,
        }],
    )
    store.write_rows(
        "tables/stationary_solver_runs.csv",
        [{
            "job_key": "local_inclusion_stress_single_N5_N64",
            "case_key": "local_inclusion_stress",
            "grid_size": 64,
            "variant_label": "single_N5",
            "parameter_strategy": "single_coarse",
            "coarse_sizes": "5",
            "solver_name": "SIM",
            "status": "divergence_guard",
            "qualified": False,
            "operator_action_count": 100,
            "true_final_residual": 1e6,
            "circle_q": 0.4,
            "circle_margin": 0.1,
            "elapsed_seconds": 1.0,
        }],
    )
    store.write_rows(
        "tables/stationary_sim_parameters.csv",
        [{
            "job_key": "local_inclusion_stress_single_N5_N64",
            "parameter_strategy": "single_coarse",
            "q": 0.4,
            "margin": 0.1,
        }],
    )
    store.write_rows("tables/job_status.csv", [{"job_key": "x", "status": "complete"}])
    store.finalize(config={"profile": "test"})


def test_campaign_merge_builds_compact_summary(tmp_path: Path):
    spectral = tmp_path / "spectral"
    validation = tmp_path / "validation"
    _make_spectral_run(spectral)
    _make_validation_run(validation)

    result = merge_chapter4_campaign(
        [spectral, validation],
        output_root=tmp_path / "final",
        create_core=True,
        require_complete_campaign=False,
    )
    final = Path(result.output_root)
    assert json.loads((final / "manifest.json").read_text())["status"] == "complete"
    assert (final / "tables" / "chapter4_solver_domain.csv").is_file()
    assert (final / "tables" / "chapter4_stationary_stress_summary.csv").is_file()
    assert Path(result.core_archive_path).is_file()
    domain = pd.read_csv(final / "tables" / "chapter4_solver_domain.csv")
    assert int(domain.loc[0, "all_solvers_max_grid"]) == 8
    log = final / "logs" / "events.jsonl"
    assert log.is_file()
    assert "campaign_merge_start" in log.read_text(encoding="utf-8")


def test_final_kaggle_notebook_contract():
    path = Path("notebooks/em-chapter4-final-kaggle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for token in (
        "experiments.chapter4_final.cli",
        "spectral",
        "main64",
        "audit",
        "control96",
        "control128",
        "merge",
        "create_core_archive",
        "EM3D_MERGE_INPUTS",
    ):
        assert token in source


def test_fixed_spectrum_section42_scan_is_reproducible(tmp_path: Path):
    from dataclasses import replace

    from experiments.em_spectral import (
        FixedSpectrumScanConfig,
        SpectralStudyConfig,
    )
    from experiments.em_spectral.chapter42_spectrum import (
        run_fixed_spectrum_study,
    )

    base = SpectralStudyConfig.quick(
        output_root=tmp_path / "spectral42",
        device="cpu",
    )
    config = replace(
        base,
        fixed_spectrum=FixedSpectrumScanConfig(
            grid_size=2,
            real_wave_numbers=(0.0,),
            lossy_wave_numbers=(0.0,),
            selected_real_wave_numbers=(0.0,),
            selected_lossy_wave_numbers=(0.0,),
            max_iter=40,
            rtol=1e-4,
        ),
    )
    store = ArtifactStore(config.output_root)
    result = run_fixed_spectrum_study(
        config,
        store=store,
        render_figures=False,
    )
    store.finalize(config=config)

    assert len(result.points) == 2
    table = pd.read_csv(
        Path(config.output_root) / "tables" / "S42_fixed_grid_spectrum_scan.csv"
    )
    assert set(table["material_key"]) == {
        "real_anisotropy",
        "lossy_anisotropy",
    }
    assert set(table["k0"]) == {0.0}
    assert (table["N"] == 2).all()
