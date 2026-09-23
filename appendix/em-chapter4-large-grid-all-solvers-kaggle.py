# Generated from a Jupyter notebook; edit the notebook, not this file.
# Source: notebooks/em-chapter4-large-grid-all-solvers-kaggle.ipynb
# SHA-256: ba50cc8ba37e23bf44871d680b8b0066213500069855abd379876213523c9eb6

# %% [cell 01; notebook index 2]
import os
from pathlib import Path

REPOSITORY_URL = "https://github.com/qwerty29544/em3d.git"
REPOSITORY_REF = os.environ.get("EM3D_REPOSITORY_REF", "main")

# smoke | main64 | audit | control96 | control128
RUN_PROFILE = os.environ.get("EM3D_RUN_PROFILE", "main64")
DEVICE = os.environ.get("EM3D_DEVICE", "cuda")
PRECISION = os.environ.get("EM3D_PRECISION", "double")
BATCH_INDEX = int(os.environ.get("EM3D_BATCH_INDEX", "0"))
BATCH_COUNT = int(os.environ.get("EM3D_BATCH_COUNT", "1"))
INCLUDE_OPTIONAL = os.environ.get("EM3D_INCLUDE_OPTIONAL", "0") == "1"
RESUME = os.environ.get("EM3D_RESUME", "1") == "1"
RUN_SMOKE_TEST = os.environ.get("EM3D_RUN_SMOKE", "1") == "1"

KAGGLE_WORKING = Path("/kaggle/working")
WORK_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else Path.cwd() / "_kaggle_work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
REPO_ROOT = WORK_ROOT / "em3d"
RESULTS_PARENT = WORK_ROOT / "em3d_chapter4_all_solvers"
RESULTS_PARENT.mkdir(parents=True, exist_ok=True)

print({
    "profile": RUN_PROFILE,
    "device": DEVICE,
    "precision": PRECISION,
    "batch": f"{BATCH_INDEX + 1}/{BATCH_COUNT}",
    "include_optional": INCLUDE_OPTIONAL,
    "resume": RESUME,
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

def complete_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    import json
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
import json
import platform
import numpy as np
import pandas as pd
import em3d

precision_enum = em3d.Precision.DOUBLE if PRECISION == "double" else em3d.Precision.SINGLE
backend = em3d.Backend.cupy(precision_enum) if DEVICE == "cuda" else em3d.Backend.numpy(precision_enum)

if DEVICE == "cuda":
    import cupy as cp
    warmup = cp.zeros((32, 32, 32), dtype=cp.complex128 if PRECISION == "double" else cp.complex64)
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

for plane in ("xy", "xz", "yz"):
    gate = em3d.mie.mie_nearfield_farfield_consistency(
        0.25, 1.5 + 0j, 2.0, plane=plane, n_phi=180, observation_radius=250.0
    )
    assert gate["relative_l2"] < 5e-4, gate
print("Mie near-field/far-field gate: PASS")

# %% [cell 04; notebook index 8]
from experiments.em_validation import build_large_grid_config

PLAN_ROOT = RESULTS_PARENT / f"plan-{RUN_PROFILE}"
plan = build_large_grid_config(
    RUN_PROFILE,
    output_root=PLAN_ROOT,
    device=DEVICE,
    precision=PRECISION,
    batch_index=BATCH_INDEX,
    batch_count=BATCH_COUNT,
    include_optional=INCLUDE_OPTIONAL,
    resume=RESUME,
    render_progress=True,
)
MIE_JOB_KEYS = [job.key for job in plan.mie_jobs]
STATIONARY_JOB_KEYS = [job.key for job in plan.stationary_jobs]
print({
    "mie_jobs": len(MIE_JOB_KEYS),
    "stationary_jobs": len(STATIONARY_JOB_KEYS),
    "first_jobs": (MIE_JOB_KEYS + STATIONARY_JOB_KEYS)[:10],
})

# %% [cell 05; notebook index 10]
if RUN_SMOKE_TEST:
    smoke_root = RESULTS_PARENT / f"smoke-{REPOSITORY_COMMIT[:12]}"
    if not (RESUME and complete_manifest(smoke_root / "manifest.json")):
        run([
            sys.executable, "-m", "experiments.em_validation.large_cli",
            "--profile", "smoke",
            "--output-root", str(smoke_root),
            "--device", DEVICE,
            "--precision", PRECISION,
        ], cwd=REPO_ROOT)
    smoke_manifest = json.loads((smoke_root / "manifest.json").read_text(encoding="utf-8"))
    assert smoke_manifest["status"] == "complete"
    smoke_common = pd.read_csv(smoke_root / "tables" / "mie_common_solvability.csv")
    assert bool(smoke_common["all_solvers_qualified"].all())
    display(smoke_common)
else:
    print("Smoke test disabled")

# %% [cell 06; notebook index 12]
RUN_LABEL = f"{RUN_PROFILE}-batch-{BATCH_INDEX + 1:02d}-of-{BATCH_COUNT:02d}-{REPOSITORY_COMMIT[:12]}"
RUN_ROOT = RESULTS_PARENT / RUN_LABEL

large_control = RUN_PROFILE in {"control96", "control128"}
if large_control:
    job_roots = []
    for job_key in MIE_JOB_KEYS + STATIONARY_JOB_KEYS:
        job_root = RUN_ROOT / "jobs" / job_key
        job_roots.append(job_root)
        if RESUME and complete_manifest(job_root / "manifest.json"):
            print("resume:", job_key)
            continue
        command = [
            sys.executable, "-m", "experiments.em_validation.large_cli",
            "--profile", RUN_PROFILE,
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
    merge_root = RUN_ROOT / "merged"
    merge_command = [
        sys.executable, "-m", "experiments.em_validation.large_cli",
        "--profile", "merge",
        "--output-root", str(merge_root),
    ]
    for root in job_roots:
        if complete_manifest(root / "manifest.json"):
            merge_command.extend(["--merge-input", str(root)])
    run(merge_command, cwd=REPO_ROOT)
    FINAL_ROOT = merge_root
else:
    FINAL_ROOT = RUN_ROOT
    if not (RESUME and complete_manifest(FINAL_ROOT / "manifest.json")):
        command = [
            sys.executable, "-m", "experiments.em_validation.large_cli",
            "--profile", RUN_PROFILE,
            "--output-root", str(FINAL_ROOT),
            "--device", DEVICE,
            "--precision", PRECISION,
            "--batch-index", str(BATCH_INDEX),
            "--batch-count", str(BATCH_COUNT),
        ]
        if INCLUDE_OPTIONAL:
            command.append("--include-optional")
        run(command, cwd=REPO_ROOT)

manifest = json.loads((FINAL_ROOT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete"
print({"final_root": str(FINAL_ROOT), "artifacts": len(manifest.get("artifacts", []))})

# %% [cell 07; notebook index 14]
def display_table(name: str, n=20):
    path = FINAL_ROOT / "tables" / name
    if path.is_file():
        frame = pd.read_csv(path)
        display(frame.head(n))
        return frame
    print("missing:", path)
    return pd.DataFrame()

solver_runs = display_table("mie_solver_runs.csv", 30)
common = display_table("mie_common_solvability.csv", 30)
rcs_errors = display_table("mie_solver_vs_mie_rcs.csv", 30)
memory = display_table("gpu_memory_probe.csv", 30)
jobs = display_table("job_status.csv", 30)

# %% [cell 08; notebook index 16]
from IPython.display import Image, display as ipy_display

preview_paths = (
    sorted((FINAL_ROOT / "figures" / "mie_rcs").glob("*.png"))[:2]
    + sorted((FINAL_ROOT / "figures" / "mie_fields").glob("*overview.png"))[:2]
    + sorted((FINAL_ROOT / "figures" / "stationary_rcs").glob("*.png"))[:1]
)
for path in preview_paths:
    print(path.relative_to(FINAL_ROOT))
    ipy_display(Image(filename=str(path), width=900))

# %% [cell 09; notebook index 18]
from experiments.em_validation.kaggle import archive_results, write_session_metadata

session = {
    "repository_url": REPOSITORY_URL,
    "repository_ref": REPOSITORY_REF,
    "repository_commit": REPOSITORY_COMMIT,
    "run_profile": RUN_PROFILE,
    "device": DEVICE,
    "precision": PRECISION,
    "batch_index": BATCH_INDEX,
    "batch_count": BATCH_COUNT,
    "include_optional": INCLUDE_OPTIONAL,
    "final_root": str(FINAL_ROOT),
    "manifest_status": manifest["status"],
    "cuda": backend.memory_info(),
}
write_session_metadata(FINAL_ROOT, session)
archive, digest = archive_results(FINAL_ROOT)
sha_path = archive.with_suffix(archive.suffix + ".sha256")
sha_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
print({"archive": str(archive), "sha256": digest, "sha_file": str(sha_path)})
