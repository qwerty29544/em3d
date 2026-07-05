import json
import shutil
import uuid
from pathlib import Path

import em3d


def _fresh_output_root(name: str) -> Path:
    return Path("experiments") / "outputs" / f"test-acoustic-{name}-{uuid.uuid4().hex}"


def test_packaged_acoustic_experiment_imports_and_defaults():
    from em3d.experiments.acoustic_scattering import AcousticCase, make_sphere_case

    case = make_sphere_case()
    assert isinstance(case, AcousticCase)
    assert case.N == (64, 64, 64)
    assert case.coarse_N == (6, 6, 6)
    assert case.solver_names == ("SIM", "BiCGStab", "TwoStep")
    assert case.eta_background == 1.0 + 0.0j


def test_packaged_acoustic_small_run_writes_artifacts(monkeypatch):
    from em3d.experiments.acoustic_scattering import (
        build_acoustic_problem,
        make_sphere_case,
        run_acoustic_experiment,
    )

    root = _fresh_output_root("small-run")
    try:
        monkeypatch.setattr(em3d.acoustics.visualization, "plot_scalar_slices", lambda *args, **kwargs: [])
        monkeypatch.setattr(em3d.acoustics.visualization, "plot_pattern", lambda *args, **kwargs: (object(), object()))

        case = make_sphere_case(
            N=6,
            coarse_N=3,
            radius=0.25,
            eta_inside=1.3 + 0.05j,
            eta_background=1.0,
            k0=0.8,
            solver_names=("BiCGStab",),
        )
        problem, operator = build_acoustic_problem(case)
        assert problem.grid.N == (6, 6, 6)
        assert isinstance(operator, em3d.acoustics.AcousticOperator)

        summary = run_acoustic_experiment(
            case=case,
            output_root=root,
            max_iter=5,
            rtol=1e-4,
            n_angles=12,
        )
        assert summary["case_name"] == case.name
        assert summary["N"] == "6x6x6"
        assert summary["coarse_N"] == "3x3x3"
        assert summary["solver_names"] == ["BiCGStab"]
        assert (root / "tables" / "acoustic_solver_runs.csv").is_file()
        assert (root / "raw" / "acoustic_summary.json").is_file()
        assert (root / "raw" / "acoustic_residual_histories.json").is_file()
        assert '"event": "finish"' in (root / "raw" / "acoustic_scattering.jsonl").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_acoustic_kaggle_notebook_uses_packaged_api():
    path = Path("notebooks") / "acoustic-scattering-kaggle.ipynb"
    assert path.is_file()
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "pip install" in source
    assert "git+https://github.com/qwerty29544/em3d.git" in source
    assert "from em3d.experiments.acoustic_scattering import" in source
    assert "make_sphere_case" in source
    assert "run_acoustic_experiment" in source
    assert "zipfile" in source
