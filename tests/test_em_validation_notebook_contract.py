from __future__ import annotations

import json
from pathlib import Path


def test_validation_notebook_uses_packaged_experiment_api_only():
    path = Path("notebooks") / "em-solver-farfield-validation.ipynb"
    assert path.is_file()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for token in (
        "ValidationStudyConfig.quick",
        "ValidationStudyConfig.publication",
        "run_validation_suite",
        "solver_runs.csv",
        "field_comparisons.csv",
        "rcs_solver_comparisons.csv",
        "mie_validation.csv",
        "mie_validation_selected.csv",
    ):
        assert token in source
    for forbidden in (
        "class Operator",
        "class BiCGStab",
        "class TwoStep",
        "def compute_rcs_curve",
        "def evaluate_mie_solution",
        "def dense_kernel",
        "def mie_coefficients",
    ):
        assert forbidden not in source


def test_validation_notebook_is_saved_with_executed_outputs():
    path = Path("notebooks") / "em-solver-farfield-validation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert any(cell.get("outputs") for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
