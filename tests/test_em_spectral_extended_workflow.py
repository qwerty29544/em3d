import json
from pathlib import Path

import pandas as pd

from experiments.em_spectral import (
    GeometryScanConfig,
    GridHierarchyConfig,
    IterationStudyConfig,
    RuntimeConfig,
    SpectralStudyConfig,
    WaveNumberScanConfig,
    run_spectral_experiment_suite,
)
from em3d.spectral import ArnoldiConfig


def _tiny_config(root: Path) -> SpectralStudyConfig:
    return SpectralStudyConfig(
        mode="quick",
        grids=GridHierarchyConfig(
            coarse_sizes=(2,),
            control_size=2,
            fine_size=4,
        ),
        iteration=IterationStudyConfig(
            max_iter=40,
            rtol=1e-6,
            divergence_guard=1e4,
        ),
        arnoldi=ArnoldiConfig(krylov_dimension=6, milestones=(3, 6)),
        geometry=GeometryScanConfig(
            inclusion_sides=(0.5,),
            ensembles=((2,),),
        ),
        wave_number=WaveNumberScanConfig(
            inclusion_side=0.5,
            wave_numbers=(0.5,),
            boundary_points=(),
        ),
        runtime=RuntimeConfig(device="cpu"),
        output_root=root,
    )


def test_extended_suite_writes_E3_to_E6_tables(tmp_path):
    config = _tiny_config(tmp_path)
    result = run_spectral_experiment_suite(
        config,
        include_core=False,
        include_geometry=True,
        include_volume_averaging=True,
        include_ensemble=True,
        include_wave_number=True,
        include_fine=True,
        include_core_arnoldi=False,
        include_boundary_arnoldi=False,
    )

    assert result.geometry is not None
    assert result.volume_averaging is not None
    assert result.ensemble is not None
    assert result.wave_number is not None

    required = [
        "E4_summary.csv",
        "E6_comparison.csv",
        "E5_comparison.csv",
        "E3_summary_phase.csv",
        "FINAL_experimental_protocol.csv",
    ]
    for name in required:
        path = tmp_path / "tables" / name
        assert path.is_file(), name
        assert len(pd.read_csv(path)) > 0

    phase = pd.read_csv(tmp_path / "tables" / "E3_summary_phase.csv")
    assert "phase_detailed" in phase
    assert "article_phase_code" in phase
    assert set(phase["phase_detailed"]).issubset(
        {
            "no_parameter",
            "converged",
            "contracting_unresolved",
            "unstable",
            "unresolved",
            "nonfinite",
        }
    )

    geometry = pd.read_csv(tmp_path / "tables" / "E6_geometry_validation.csv")
    assert geometry["avg_volume_abs_error"].max() < 1e-12

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert any(
        item["path"] == "tables/E3_summary_phase.csv"
        for item in manifest["artifacts"]
    )
