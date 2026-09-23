from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import nbformat
import numpy as np

from experiments.em_validation import build_large_grid_config, run_large_grid_suite
from experiments.em_validation.config import MieJobSpec, VisualizationConfig
from experiments.em_validation.memory import estimate_large_grid_memory


def test_large_grid_planner_profiles_have_expected_jobs(tmp_path):
    main = build_large_grid_config(
        "main64", output_root=tmp_path / "main", device="cpu"
    )
    control96 = build_large_grid_config(
        "control96", output_root=tmp_path / "c96", device="cpu"
    )
    control128 = build_large_grid_config(
        "control128", output_root=tmp_path / "c128", device="cpu"
    )
    control128_optional = build_large_grid_config(
        "control128",
        output_root=tmp_path / "c128o",
        device="cpu",
        include_optional=True,
    )
    assert len(main.mie_jobs) == 60
    assert {job.grid_size for job in main.mie_jobs} == {24, 32, 48, 64}
    assert len(control96.mie_jobs) == 5
    assert all(job.grid_size == 96 for job in control96.mie_jobs)
    assert len(control128.mie_jobs) == 3
    assert len(control128_optional.mie_jobs) == 4
    assert all(job.grid_size == 128 for job in control128_optional.mie_jobs)


def test_memory_model_is_monotone_and_derived_adjoint_is_smaller():
    values = [
        estimate_large_grid_memory(
            n,
            precision="double",
            include_adjoint=True,
            adjoint_storage="derived",
            rcs_batch_size=4,
        )
        for n in (64, 96, 128)
    ]
    assert values[0].estimated_peak_bytes < values[1].estimated_peak_bytes < values[2].estimated_peak_bytes
    explicit = estimate_large_grid_memory(
        96,
        precision="double",
        include_adjoint=True,
        adjoint_storage="explicit",
        rcs_batch_size=4,
    )
    assert values[1].kernel_bytes < explicit.kernel_bytes
    assert values[1].estimated_peak_bytes < explicit.estimated_peak_bytes


def test_all_solver_smoke_workflow(tmp_path):
    config = build_large_grid_config(
        "smoke",
        output_root=tmp_path / "run",
        device="cpu",
        render_progress=False,
    )
    config = replace(
        config,
        mie_jobs=(
            MieJobSpec(
                eps_r=1.2 + 0.0j,
                k0a=0.5,
                grid_size=4,
                true_rtol=1e-4,
                compute_full_field_metrics=True,
                render_field_slices=True,
            ),
        ),
        stationary_jobs=(),
        visualization=VisualizationConfig(
            field_planes=("xz",),
            rcs_planes=("xz",),
            rcs_coordinate_systems=("cartesian", "polar"),
            save_formats=("png",),
        ),
        rcs_n_phi=36,
    )
    result = run_large_grid_suite(config)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    common = result.common_solvability_rows
    assert len(common) == 1
    assert common[0]["all_solvers_qualified"] is True
    assert {row["solver_name"] for row in result.solver_rows} == {
        "SIM",
        "BiCGStab",
        "TwoStep",
    }
    assert (tmp_path / "run" / "archives" / "mie_field_figures.zip").is_file()
    assert (tmp_path / "run" / "archives" / "mie_rcs_figures.zip").is_file()


def test_kaggle_notebook_uses_large_grid_package_api():
    path = Path("notebooks/em-chapter4-large-grid-all-solvers-kaggle.ipynb")
    nb = nbformat.read(path, as_version=4)
    source = "\n".join(
        "".join(cell.get("source", "")) for cell in nb["cells"]
    )
    for token in (
        "build_large_grid_config",
        "experiments.em_validation.large_cli",
        "RUN_PROFILE",
        "control96",
        "control128",
        "mie_common_solvability.csv",
    ):
        assert token in source
    forbidden = (
        "class Operator",
        "def mie_coefficients",
        "def convex_hull_complex",
        "class BiCGStab",
    )
    assert not any(token in source for token in forbidden)
