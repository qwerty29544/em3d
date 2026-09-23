import json
from pathlib import Path


def test_kaggle_notebook_uses_repository_and_packaged_api():
    path = Path("notebooks") / "em-solver-farfield-validation-kaggle.ipynb"
    assert path.is_file()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for token in (
        "https://github.com/qwerty29544/em3d.git",
        "git\", \"clone",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "publication_config_for_kaggle",
        "run_validation_suite",
        "archive_results",
        "REPOSITORY_COMMIT",
        "manifest[\"git_commit\"]",
    ):
        assert token in source

    forbidden = (
        "class Operator(",
        "class BiCGStab(",
        "class TwoStep(",
        "def scatter_integral_direct(",
        "def run_validation_suite(",
    )
    for token in forbidden:
        assert token not in source
