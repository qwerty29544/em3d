import json
from pathlib import Path


def test_spectral_notebook_uses_packaged_api_only():
    path = Path("notebooks") / "em-spectral-transfer.ipynb"
    assert path.is_file()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for token in [
        "SpectralStudyConfig.quick",
        "SpectralStudyConfig.publication",
        "run_spectral_experiment_suite",
        "E4_summary.csv",
        "E6_comparison.csv",
        "E5_comparison.csv",
        "E3_summary_phase.csv",
        "phase_detailed",
        "article_phase_code",
    ]:
        assert token in source
    for forbidden in [
        "def convex_hull_complex",
        "def arnoldi_iteration_operator",
        "class FFTOperator",
        "def dense_kernel_blocks",
        "def _run_simple_iteration_once",
        "def run_E3_fine_k0",
        "def run_E5_fine_side",
    ]:
        assert forbidden not in source


def test_spectral_notebook_is_saved_with_executed_outputs():
    path = Path("notebooks") / "em-spectral-transfer.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert any(cell.get("outputs") for cell in code_cells)
