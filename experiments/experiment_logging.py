"""External event logs for chapter 6 experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class ExperimentLogger:
    """Append JSONL and text event logs under an experiment output root."""

    output_root: str | Path
    experiment_name: str
    raw_dir: Path = field(init=False)
    jsonl_path: Path = field(init=False)
    text_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.output_root)
        self.raw_dir = root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.experiment_name.replace("/", "-").replace("\\", "-")
        self.jsonl_path = self.raw_dir / f"{safe_name}.jsonl"
        self.text_path = self.raw_dir / f"{safe_name}.log"

    def event(self, event: str, **payload: Any) -> dict[str, Any]:
        """Append one event to both log files and return the serialized row."""
        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False, default=_json_default)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        payload_text = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=_json_default)}"
            for key, value in payload.items()
        )
        text_line = f"{row['time']} {row['event']}"
        if payload_text:
            text_line += f" {payload_text}"
        with self.text_path.open("a", encoding="utf-8") as f:
            f.write(text_line + "\n")
        return row
