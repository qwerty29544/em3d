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
        "run_spectral_transfer_study",
        "default_article_cases",
        "plot_spectral_localizations",
        "plot_transfer_bounds",
    ]:
        assert token in source
    for forbidden in [
        "def convex_hull_complex",
        "def arnoldi_iteration_operator",
        "class FFTOperator",
        "def dense_kernel_blocks",
        "def _run_simple_iteration_once",
    ]:
        assert forbidden not in source
