import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import em3d


def _fresh_output_root(name: str) -> Path:
    return Path("experiments") / "outputs" / f"test-packaged-{name}-{uuid.uuid4().hex}"


def test_packaged_structured_lattice_imports_and_defaults():
    from em3d.experiments.structured_lattice import (
        MaterialSpec,
        StructuredLatticeCase,
        make_structured_lattice_case,
    )

    case = make_structured_lattice_case()
    assert isinstance(case, StructuredLatticeCase)
    assert case.N == (100, 100, 100)
    assert case.coarse_N == (9, 9, 9)
    assert case.lattice_shape == (5, 5, 5)
    assert case.solver_names == ("SIM", "BiCGStab", "TwoStep")
    assert isinstance(case.inclusions[0].material, MaterialSpec)


def test_packaged_structured_lattice_small_run_writes_artifacts(monkeypatch):
    from em3d.experiments.structured_lattice import (
        MaterialSpec,
        build_structured_lattice_problem,
        make_structured_lattice_case,
        run_structured_lattice_experiment,
    )

    root = _fresh_output_root("small-run")
    try:
        monkeypatch.setattr(em3d.vis, "plot_field_scalar_slice", lambda *args, **kwargs: (object(), object()))
        monkeypatch.setattr(em3d.vis, "plot_field_vector_slice", lambda *args, **kwargs: (object(), object()))
        monkeypatch.setattr(em3d.vis, "plot_rcs", lambda *args, **kwargs: (object(), object()))
        monkeypatch.setattr(em3d.vis, "plot_rcs_polar", lambda *args, **kwargs: (object(), object()))

        class FakeFig:
            def savefig(self, *args, **kwargs):
                return None

            def tight_layout(self):
                return None

        class FakeAx:
            def semilogy(self, *args, **kwargs):
                return None

            def set_xlabel(self, value):
                return None

            def set_ylabel(self, value):
                return None

            def set_title(self, value):
                return None

            def grid(self, *args, **kwargs):
                return None

            def legend(self):
                return None

        class FakePlt:
            def subplots(self, **kwargs):
                return FakeFig(), FakeAx()

        monkeypatch.setattr(em3d.vis, "_require_matplotlib", lambda: FakePlt())

        case = make_structured_lattice_case(
            N=6,
            coarse_N=3,
            lattice_shape=(1, 1, 1),
            inclusion_radius=(0.18, 0.18, 0.18),
            material=MaterialSpec.isotropic(1.4 + 0.0j),
            k0=1.0,
            solver_names=("BiCGStab",),
        )
        problem, operator = build_structured_lattice_problem(case)
        assert problem.grid.N == (6, 6, 6)
        assert isinstance(operator, em3d.Operator)
        assert np.count_nonzero(np.asarray(problem.eps_tensor[0, 0])) > 0

        summary = run_structured_lattice_experiment(
            case=case,
            output_root=root,
            max_iter=5,
            rtol=1e-4,
            rcs_n_phi=12,
        )
        assert summary["case_name"] == case.name
        assert summary["N"] == "6x6x6"
        assert summary["coarse_N"] == "3x3x3"
        assert summary["solver_names"] == ["BiCGStab"]
        assert (root / "tables" / "structured_lattice_solver_runs.csv").is_file()
        assert (root / "raw" / "structured_lattice_summary.json").is_file()
        assert (root / "raw" / "structured_lattice_residual_histories.json").is_file()
        assert '"event": "finish"' in (root / "raw" / "structured_lattice.jsonl").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_kaggle_notebook_uses_packaged_experiment_api():
    path = Path("notebooks") / "structured-lattice-kaggle.ipynb"
    assert path.is_file()
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "pip install" in source
    assert "git+https://github.com/qwerty29544/em3d.git" in source
    assert "from em3d.experiments.structured_lattice import" in source
    assert "make_structured_lattice_case" in source
    assert "run_structured_lattice_experiment" in source
    assert "zipfile" in source
