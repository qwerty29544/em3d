# Generated from a Jupyter notebook; edit the notebook, not this file.
# Source: notebooks/em-chapter4-final-kaggle.ipynb
# SHA-256: 449b4c608fbd78a785829309fd4bcb6cebdffa1002e14dec99a3d56f9e768757

# %% [cell 01; notebook index 2]
import json
import os
from pathlib import Path

REPOSITORY_URL = "https://github.com/qwerty29544/em3d.git"
REPOSITORY_REF = os.environ.get("EM3D_REPOSITORY_REF", "main")

# smoke | spectral | main64 | audit | control96 | control128 | merge
RUN_STAGE = os.environ.get("EM3D_CHAPTER4_STAGE", "smoke")
DEVICE = os.environ.get("EM3D_DEVICE", "cuda")
PRECISION = os.environ.get("EM3D_PRECISION", "double")
BATCH_INDEX = int(os.environ.get("EM3D_BATCH_INDEX", "0"))
BATCH_COUNT = int(os.environ.get("EM3D_BATCH_COUNT", "1"))
INCLUDE_OPTIONAL = os.environ.get("EM3D_INCLUDE_OPTIONAL", "0") == "1"
RESUME = os.environ.get("EM3D_RESUME", "1") == "1"
RUN_PRECHECK = os.environ.get("EM3D_RUN_PRECHECK", "1") == "1"
RENDER_FIGURES = os.environ.get("EM3D_RENDER_FIGURES", "1") == "1"
SMOKE_RENDER_FIGURES = os.environ.get("EM3D_SMOKE_RENDER_FIGURES", "0") == "1"

# Для merge можно задать JSON-массив или список путей через os.pathsep.
MERGE_INPUTS_RAW = os.environ.get("EM3D_MERGE_INPUTS", "")

KAGGLE_WORKING = Path("/kaggle/working")
WORK_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else Path.cwd() / "_kaggle_work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
REPO_ROOT = WORK_ROOT / "em3d"
RESULTS_PARENT = WORK_ROOT / "em3d_chapter4_final_campaign"
RESULTS_PARENT.mkdir(parents=True, exist_ok=True)

print({
    "stage": RUN_STAGE,
    "device": DEVICE,
    "precision": PRECISION,
    "batch": f"{BATCH_INDEX + 1}/{BATCH_COUNT}",
    "include_optional": INCLUDE_OPTIONAL,
    "resume": RESUME,
    "render_figures": RENDER_FIGURES,
})

# %% [cell 02; notebook index 4]
import importlib.metadata
import re
import shutil
import subprocess
import sys
from typing import Sequence


def run(command: Sequence[str], *, cwd: Path | None = None, capture: bool = False, env=None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def manifest_complete(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except Exception:
        return False


local_repository = os.environ.get("EM3D_LOCAL_REPOSITORY")
if local_repository:
    REPO_ROOT = Path(local_repository).resolve()
else:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    run(["git", "clone", "--branch", REPOSITORY_REF, "--depth", "1", REPOSITORY_URL, str(REPO_ROOT)])

REPOSITORY_COMMIT = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture=True)

if DEVICE == "cuda":
    try:
        nvcc = run(["nvcc", "--version"], capture=True)
        match = re.search(r"release\s+([0-9]+)", nvcc)
        cuda_major = int(match.group(1)) if match else 12
    except Exception:
        cuda_major = 12
    cupy_package = "cupy-cuda13x" if cuda_major >= 13 else "cupy-cuda12x"
    installed = {
        dist.metadata.get("Name", "").lower()
        for dist in importlib.metadata.distributions()
    }
    if not any(name == "cupy" or name.startswith("cupy-cuda") for name in installed):
        run([sys.executable, "-m", "pip", "install", "-q", cupy_package])

run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(REPO_ROOT)])
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

print({"repository": str(REPO_ROOT), "commit": REPOSITORY_COMMIT})

# %% [cell 03; notebook index 6]
import platform
import numpy as np
import pandas as pd
import em3d

precision_enum = em3d.Precision.DOUBLE if PRECISION == "double" else em3d.Precision.SINGLE
backend = em3d.Backend.cupy(precision_enum) if DEVICE == "cuda" else em3d.Backend.numpy(precision_enum)

if DEVICE == "cuda":
    import cupy as cp
    dtype = cp.complex128 if PRECISION == "double" else cp.complex64
    warmup = cp.zeros((32, 32, 32), dtype=dtype)
    _ = cp.fft.fftn(warmup)
    backend.synchronize()
    del warmup
    backend.clear_memory_pool()

print({
    "em3d_version": em3d.__version__,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "device": backend.device,
    "memory": backend.memory_info(),
})
assert em3d.__version__ >= "0.8.0", "Требуется финальная версия стенда em3d >= 0.8.0"

# %% [cell 04; notebook index 8]
PRECHECK_ROOT = RESULTS_PARENT / f"smoke-{REPOSITORY_COMMIT[:12]}"
if RUN_PRECHECK and RUN_STAGE != "smoke":
    spectral_ok = manifest_complete(PRECHECK_ROOT / "spectral")
    validation_ok = manifest_complete(PRECHECK_ROOT / "validation")
    if not (RESUME and spectral_ok and validation_ok):
        command = [
            sys.executable, "-m", "experiments.chapter4_final.cli",
            "--stage", "smoke",
            "--output-root", str(PRECHECK_ROOT),
            "--device", DEVICE,
            "--precision", PRECISION,
        ]
        if not SMOKE_RENDER_FIGURES:
            command.append("--no-figures")
        run(command, cwd=REPO_ROOT)
    assert manifest_complete(PRECHECK_ROOT / "spectral")
    assert manifest_complete(PRECHECK_ROOT / "validation")
    print("Chapter 4 smoke/precheck: PASS")
else:
    print("Separate precheck skipped for this stage")

# %% [cell 05; notebook index 10]
from experiments.em_validation import build_large_grid_config

RUN_LABEL = f"{RUN_STAGE}-batch-{BATCH_INDEX + 1:02d}-of-{BATCH_COUNT:02d}-{REPOSITORY_COMMIT[:12]}"
RUN_ROOT = RESULTS_PARENT / RUN_LABEL
FINAL_ROOT: Path | None = None

if RUN_STAGE == "smoke":
    if not (RESUME and manifest_complete(RUN_ROOT / "spectral") and manifest_complete(RUN_ROOT / "validation")):
        command = [
            sys.executable, "-m", "experiments.chapter4_final.cli",
            "--stage", "smoke",
            "--output-root", str(RUN_ROOT),
            "--device", DEVICE,
            "--precision", PRECISION,
        ]
        if not SMOKE_RENDER_FIGURES:
            command.append("--no-figures")
        run(command, cwd=REPO_ROOT)
    FINAL_ROOT = RUN_ROOT

elif RUN_STAGE == "spectral":
    if not (RESUME and manifest_complete(RUN_ROOT)):
        command = [
            sys.executable, "-m", "experiments.chapter4_final.cli",
            "--stage", "spectral",
            "--output-root", str(RUN_ROOT),
            "--device", DEVICE,
            "--precision", PRECISION,
        ]
        if not RENDER_FIGURES:
            command.append("--no-figures")
        run(command, cwd=REPO_ROOT)
    FINAL_ROOT = RUN_ROOT

elif RUN_STAGE in {"main64", "audit"}:
    if not (RESUME and manifest_complete(RUN_ROOT)):
        command = [
            sys.executable, "-m", "experiments.chapter4_final.cli",
            "--stage", RUN_STAGE,
            "--output-root", str(RUN_ROOT),
            "--device", DEVICE,
            "--precision", PRECISION,
            "--batch-index", str(BATCH_INDEX),
            "--batch-count", str(BATCH_COUNT),
        ]
        run(command, cwd=REPO_ROOT)
    FINAL_ROOT = RUN_ROOT

elif RUN_STAGE in {"control96", "control128"}:
    plan = build_large_grid_config(
        RUN_STAGE,
        output_root=RUN_ROOT / "plan",
        device=DEVICE,
        precision=PRECISION,
        batch_index=BATCH_INDEX,
        batch_count=BATCH_COUNT,
        include_optional=INCLUDE_OPTIONAL,
        resume=RESUME,
        render_progress=True,
    )
    job_keys = [job.key for job in plan.mie_jobs] + [job.key for job in plan.stationary_jobs]
    print({"isolated_jobs": len(job_keys), "keys": job_keys})
    completed_roots = []
    for job_key in job_keys:
        job_root = RUN_ROOT / "jobs" / job_key
        if not (RESUME and manifest_complete(job_root)):
            command = [
                sys.executable, "-m", "experiments.chapter4_final.cli",
                "--stage", RUN_STAGE,
                "--output-root", str(job_root),
                "--device", DEVICE,
                "--precision", PRECISION,
                "--batch-index", str(BATCH_INDEX),
                "--batch-count", str(BATCH_COUNT),
                "--job-key", job_key,
            ]
            if INCLUDE_OPTIONAL:
                command.append("--include-optional")
            run(command, cwd=REPO_ROOT)
        if manifest_complete(job_root):
            completed_roots.append(job_root)
    if not completed_roots:
        raise RuntimeError("No control jobs completed")
    merged_root = RUN_ROOT / "merged"
    merge_command = [
        sys.executable, "-m", "experiments.em_validation.large_cli",
        "--profile", "merge",
        "--output-root", str(merged_root),
    ]
    for root in completed_roots:
        merge_command.extend(["--merge-input", str(root)])
    run(merge_command, cwd=REPO_ROOT)
    FINAL_ROOT = merged_root

elif RUN_STAGE == "merge":
    def parse_merge_inputs(raw: str) -> list[Path]:
        if raw.strip():
            try:
                values = json.loads(raw)
                if isinstance(values, str):
                    values = [values]
            except json.JSONDecodeError:
                values = [value for value in raw.split(os.pathsep) if value]
            return [Path(value).resolve() for value in values]
        candidates = []
        search_roots = [RESULTS_PARENT, Path("/kaggle/input")]
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            for manifest_path in search_root.rglob("manifest.json"):
                root = manifest_path.parent
                if "jobs" in root.parts:
                    # Individual N=96/N=128 job manifests are already
                    # represented by the stage-level merged directory.
                    continue
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if payload.get("status") != "complete":
                    continue
                schema = str(payload.get("schema", ""))
                if "spectral" in schema or "all-solvers" in schema:
                    candidates.append(root)
        # Keep deepest merged control outputs and remove exact duplicates.
        unique = []
        seen = set()
        for path in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    merge_inputs = parse_merge_inputs(MERGE_INPUTS_RAW)
    print("Merge inputs:")
    for root in merge_inputs:
        print(" -", root)
    if len(merge_inputs) < 2:
        raise RuntimeError("Merge requires a spectral run and one or more validation runs")
    command = [
        sys.executable, "-m", "experiments.chapter4_final.cli",
        "--stage", "merge",
        "--output-root", str(RUN_ROOT),
        "--archive-core",
    ]
    for root in merge_inputs:
        command.extend(["--input", str(root)])
    run(command, cwd=REPO_ROOT)
    FINAL_ROOT = RUN_ROOT

else:
    raise ValueError(f"Unknown RUN_STAGE={RUN_STAGE!r}")

print("FINAL_ROOT =", FINAL_ROOT)

# %% [cell 06; notebook index 12]
from experiments.em_validation.kaggle import write_session_metadata

if RUN_STAGE == "smoke":
    manifest_paths = [FINAL_ROOT / "spectral" / "manifest.json", FINAL_ROOT / "validation" / "manifest.json"]
else:
    manifest_paths = [FINAL_ROOT / "manifest.json"]

for path in manifest_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete", payload
    print({
        "manifest": str(path),
        "schema": payload.get("schema"),
        "commit": payload.get("git_commit"),
        "artifacts": len(payload.get("artifacts", [])),
    })

session_root = FINAL_ROOT if RUN_STAGE != "smoke" else FINAL_ROOT / "validation"
write_session_metadata(session_root, {
    "repository_url": REPOSITORY_URL,
    "repository_ref": REPOSITORY_REF,
    "repository_commit": REPOSITORY_COMMIT,
    "run_stage": RUN_STAGE,
    "device": DEVICE,
    "precision": PRECISION,
    "batch_index": BATCH_INDEX,
    "batch_count": BATCH_COUNT,
    "include_optional": INCLUDE_OPTIONAL,
    "final_root": str(FINAL_ROOT),
    "cuda_after_run": backend.memory_info(),
})
print("Session metadata written")

# %% [cell 07; notebook index 14]
from IPython.display import display


def display_csv(path: Path, rows: int = 20):
    if not path.is_file():
        print("missing:", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    display(frame.head(rows))
    return frame

if RUN_STAGE == "spectral":
    display_csv(FINAL_ROOT / "tables" / "FINAL_experimental_protocol.csv")
    display_csv(FINAL_ROOT / "tables" / "S42_fixed_grid_spectrum_scan.csv")
    display_csv(FINAL_ROOT / "tables" / "fine_transfers.csv")
    display_csv(FINAL_ROOT / "tables" / "E3_summary_phase.csv")
elif RUN_STAGE == "merge":
    display_csv(FINAL_ROOT / "tables" / "chapter4_experiment_coverage.csv")
    display_csv(FINAL_ROOT / "tables" / "chapter4_solver_domain.csv")
    display_csv(FINAL_ROOT / "tables" / "chapter4_stationary_stress_summary.csv")
    display_csv(FINAL_ROOT / "tables" / "chapter4_mie_error_summary.csv")
elif RUN_STAGE != "smoke":
    display_csv(FINAL_ROOT / "tables" / "job_status.csv")
    display_csv(FINAL_ROOT / "tables" / "mie_common_solvability.csv")
    display_csv(FINAL_ROOT / "tables" / "stationary_solver_runs.csv")
    display_csv(FINAL_ROOT / "tables" / "gpu_memory_probe.csv")

# %% [cell 08; notebook index 16]
from IPython.display import Image, display as ipy_display

if RUN_STAGE == "smoke":
    preview_root = FINAL_ROOT / "validation"
else:
    preview_root = FINAL_ROOT

patterns = (
    "figures/selected/*.png",
    "figures/S42/*_spectra.png",
    "figures/E3_phase_article.png",
    "figures/stationary_residuals/*.png",
    "figures/mie_convergence/*.png",
    "figures/mie_rcs/*.png",
)
preview_paths = []
for pattern in patterns:
    preview_paths.extend(sorted(preview_root.glob(pattern)))
for path in preview_paths[:6]:
    print(path.relative_to(preview_root))
    ipy_display(Image(filename=str(path), width=900))

# %% [cell 09; notebook index 18]
from experiments.chapter4_final import create_core_archive

if RUN_STAGE == "smoke":
    archive_root = FINAL_ROOT / "validation"
else:
    archive_root = FINAL_ROOT

core_archive, core_digest = create_core_archive(archive_root)
print({"core_archive": str(core_archive), "sha256": core_digest})

figure_archives = sorted((archive_root / "archives").glob("*.zip")) if (archive_root / "archives").is_dir() else []
print("Figure archives:")
for path in figure_archives:
    print(" -", path)

print("Download the compact core archive first; figure archives can be downloaded separately.")
