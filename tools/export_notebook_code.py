from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def notebook_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for line_number, line in enumerate(source.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                raise ValueError(
                    f"unsupported notebook magic/shell command in cell {index}, "
                    f"line {line_number}: {line!r}"
                )
        cells.append((index, source.rstrip()))
    return cells


def render_code_export(notebook_path: Path) -> str:
    digest = notebook_sha256(notebook_path)
    lines = [
        "# Generated from a Jupyter notebook; edit the notebook, not this file.",
        f"# Source: {notebook_path.as_posix()}",
        f"# SHA-256: {digest}",
        "",
    ]
    for ordinal, (cell_index, source) in enumerate(
        extract_code_cells(notebook_path), start=1
    ):
        lines.append(f"# %% [cell {ordinal:02d}; notebook index {cell_index}]")
        if source:
            lines.append(source)
        lines.append("")
    result = "\n".join(lines).rstrip() + "\n"
    ast.parse(result)
    return result


def export_notebook_code(notebook_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_code_export(notebook_path),
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export only executable code cells from an .ipynb file"
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    export_notebook_code(args.notebook, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
