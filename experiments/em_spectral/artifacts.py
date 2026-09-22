from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

import em3d


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        repository_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.raw = self.root / "raw"
        self.figures = self.root / "figures"
        self.logs = self.root / "logs"
        for directory in (self.root, self.tables, self.raw, self.figures, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[dict[str, Any]] = []

    def _record(self, path: Path, kind: str) -> Path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._artifacts.append(
            {
                "path": str(path.relative_to(self.root)),
                "kind": kind,
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
        return path

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return self._record(path, "json")

    def write_rows(
        self,
        relative_path: str | Path,
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        materialized = [dict(row) for row in rows]
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in materialized:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in materialized:
                serialized = {
                    key: json.dumps(value, default=_json_default, ensure_ascii=False)
                    if isinstance(value, (complex, dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
                writer.writerow(serialized)
        return self._record(path, "csv")

    def write_npz(self, relative_path: str | Path, **arrays) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return self._record(path, "npz")

    def finalize(self, *, config: Any, status: str = "complete") -> Path:
        config_digest = hashlib.sha256(_stable_json_bytes(config)).hexdigest()
        manifest = {
            "schema": "em3d-spectral-transfer-artifacts-v1",
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "em3d_version": em3d.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "scipy", "pandas", "matplotlib", "cupy")
            },
            "config_sha256": config_digest,
            "config": config,
            "artifacts": self._artifacts,
        }
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path
