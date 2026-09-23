import ast
from pathlib import Path

from tools.export_notebook_code import export_notebook_code, render_code_export


def test_export_notebook_code_is_valid_python(tmp_path):
    notebook = Path("notebooks/em-spectral-transfer.ipynb")
    output = tmp_path / "export.py"
    export_notebook_code(notebook, output)
    text = output.read_text(encoding="utf-8")
    assert "SHA-256:" in text
    assert "# %% [cell" in text
    assert "run_spectral_experiment_suite" in text
    assert "# Устойчивость переноса" not in text
    ast.parse(text)


def test_render_rejects_magics(tmp_path):
    path = tmp_path / "bad.ipynb"
    path.write_text(
        '{"cells":[{"cell_type":"code","source":["%matplotlib inline"]}]}',
        encoding="utf-8",
    )
    try:
        render_code_export(path)
    except ValueError as error:
        assert "magic" in str(error)
    else:
        raise AssertionError("a notebook magic must be rejected")
