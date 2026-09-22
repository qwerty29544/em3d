import json
from pathlib import Path

import numpy as np

from em3d.experiments.spectral_transfer import SpectralCaseDefinition
from em3d.geometry import FullDomain
from experiments.em_spectral import SpectralStudyConfig, run_spectral_transfer_study


def _case():
    return SpectralCaseDefinition(
        key="workflow_case",
        title="workflow case",
        domain_lengths=(1.0, 1.0, 1.0),
        domain_center=(0.0, 0.0, 0.0),
        k0=0.5,
        background_eps_r=np.eye(3),
        feature_eps_r=np.diag([1.1, 1.08, 1.05]),
        feature_geometry=FullDomain(),
    )


def test_quick_workflow_writes_manifest_and_tables(tmp_path):
    config = SpectralStudyConfig.quick(output_root=tmp_path, device="cpu")
    result = run_spectral_transfer_study(
        config,
        cases=(_case(),),
        include_fine=False,
        include_arnoldi=False,
    )
    assert len(result.cases) == 1
    assert result.cases[0].operator_consistency_error < 1e-10
    assert (tmp_path / "tables" / "spectral_localizations.csv").is_file()
    assert (tmp_path / "tables" / "control_transfers.csv").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "em3d-spectral-transfer-artifacts-v1"
    assert manifest["status"] == "complete"
    assert manifest["artifacts"]
