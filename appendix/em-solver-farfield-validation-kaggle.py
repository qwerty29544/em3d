# Generated from a Jupyter notebook; edit the notebook, not this file.
# Source: notebooks/em-solver-farfield-validation-kaggle.ipynb
# SHA-256: b9b010f6901ef72fb5507a24fe920432e2544b2ddf759d05913c96843a49655f

# %% [cell 01; notebook index 2]
import os
from pathlib import Path

# Репозиторий и ветка/тег, которые будут фактически использованы в расчёте.
REPOSITORY_URL = "https://github.com/qwerty29544/em3d.git"
REPOSITORY_REF = "main"

# По умолчанию Kaggle выполняет публикационный CUDA-расчёт.
# Переменные окружения используются только для локальной автоматической проверки блокнота.
MODE = os.environ.get("EM3D_RUN_MODE", "publication")       # "quick" | "publication"
DEVICE = os.environ.get("EM3D_DEVICE", "cuda")              # "cuda" | "cpu"
PRECISION = os.environ.get("EM3D_PRECISION", "double")      # "double" | "single"

RUN_SOLVER_STUDY = True
RUN_MIE_STUDY = True
RENDER_FIGURES = True
RUN_SMOKE_TEST = True

# Параметры публикационной серии V1–V3.
SOLVER_GRID_SIZE = 24

# Параметры публикационной серии V4.
MIE_GRID_SIZES = (24, 32, 48)
MIE_EPS_VALUES = (1.5 + 0.0j, 2.25 + 0.0j, 4.0 + 0.0j)
MIE_K0A_VALUES = (0.5, 1.0, 2.0, 4.0, 6.0)

# Разбиение серии Ми между несколькими Kaggle-сеансами.
# При MIE_BATCH_COUNT=1 выполняется вся серия.
MIE_BATCH_AXIS = "k0a"    # "k0a" | "eps" | "grid"
MIE_BATCH_INDEX = 0       # нумерация с нуля
MIE_BATCH_COUNT = 1

# Освобождать кэш CuPy после каждого физического случая Ми.
CLEAR_CUDA_CACHE_BETWEEN_CASES = True

# Число углов ЭПР задаётся публикационной конфигурацией пакета.
# Пути Kaggle.
KAGGLE_WORKING = Path("/kaggle/working")
WORK_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else Path.cwd() / "_kaggle_work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

REPO_ROOT = WORK_ROOT / "em3d"
RESULTS_PARENT = WORK_ROOT / "em3d_chapter4_results"
RESULTS_PARENT.mkdir(parents=True, exist_ok=True)

print({
    "mode": MODE,
    "device": DEVICE,
    "precision": PRECISION,
    "working_directory": str(WORK_ROOT),
    "mie_batch": f"{MIE_BATCH_INDEX + 1}/{MIE_BATCH_COUNT} by {MIE_BATCH_AXIS}",
})

# %% [cell 02; notebook index 4]
import importlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
from typing import Sequence

def run(command: Sequence[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""

def installed_cupy_distributions() -> list[str]:
    names = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower() == "cupy" or name.lower().startswith("cupy-cuda"):
            names.append(name)
    return sorted(set(names))

def detect_cuda_major() -> int:
    # Wheel CuPy выбирается по фактически установленному CUDA Toolkit.
    # nvcc надёжнее строки совместимости драйвера из nvidia-smi.
    try:
        nvcc = run(["nvcc", "--version"], capture=True)
        match = re.search(r"release\s+([0-9]+)", nvcc)
        if match:
            return int(match.group(1))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        output = run(["nvidia-smi"], capture=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "NVIDIA GPU не обнаружен. Включите Accelerator → GPU в настройках Kaggle."
        ) from exc
    match = re.search(r"CUDA Version:\s*([0-9]+)", output)
    if match:
        return int(match.group(1))
    raise RuntimeError("Не удалось определить основную версию CUDA")

LOCAL_REPO_OVERRIDE = os.environ.get("EM3D_KAGGLE_LOCAL_REPO")
if LOCAL_REPO_OVERRIDE:
    REPO_ROOT = Path(LOCAL_REPO_OVERRIDE).resolve()
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Некорректный локальный репозиторий: {REPO_ROOT}")
else:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    try:
        run([
            "git", "clone", "--depth", "1",
            "--branch", REPOSITORY_REF,
            REPOSITORY_URL, str(REPO_ROOT),
        ])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Не удалось получить GitHub-репозиторий. "
            "Включите Internet → On в настройках Kaggle."
        ) from exc

REPOSITORY_COMMIT = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture=True)
REPOSITORY_STATUS = run(["git", "status", "--short"], cwd=REPO_ROOT, capture=True)
if REPOSITORY_STATUS and not LOCAL_REPO_OVERRIDE:
    raise RuntimeError(f"После получения репозитория рабочее дерево не чисто:\n{REPOSITORY_STATUS}")
if REPOSITORY_STATUS and LOCAL_REPO_OVERRIDE:
    print("WARNING: локальная проверка выполняется на рабочем дереве с изменениями:")
    print(REPOSITORY_STATUS)

if DEVICE == "cuda":
    cuda_major = detect_cuda_major()
    try:
        import cupy as cp
        cupy_ready = bool(cp.cuda.is_available())
    except Exception:
        cupy_ready = False

    if not cupy_ready:
        existing = installed_cupy_distributions()
        if existing:
            raise RuntimeError(
                "Найдена неработоспособная или конфликтующая установка CuPy: "
                f"{existing}. Перезапустите Kaggle-сеанс с чистым образом."
            )
        cupy_package = "cupy-cuda13x" if cuda_major >= 13 else "cupy-cuda12x"
        print(f"Installing {cupy_package} for detected CUDA major={cuda_major}")
        run([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--upgrade", cupy_package,
        ])
        importlib.invalidate_caches()
        import cupy as cp
        if not cp.cuda.is_available():
            raise RuntimeError("CuPy установлен, но CUDA-устройство недоступно")

# Editable-установка нужна, поскольку сценарии experiments находятся в корне репозитория.
# При локальной автоматической проверке пакет уже доступен из рабочего дерева,
# поэтому сетевой pip-шаг намеренно пропускается.
if not LOCAL_REPO_OVERRIDE:
    run([
        sys.executable, "-m", "pip", "install",
        "--quiet", "--upgrade", "--no-build-isolation", "-e", str(REPO_ROOT),
    ])

for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

print({
    "repository": REPOSITORY_URL,
    "ref": REPOSITORY_REF,
    "commit": REPOSITORY_COMMIT,
    "repo_root": str(REPO_ROOT),
    "cupy_distributions": installed_cupy_distributions(),
})

# %% [cell 03; notebook index 6]
import json
import platform
import time
import numpy as np
import pandas as pd

import em3d

if DEVICE == "cuda":
    import cupy as cp

    backend = em3d.Backend.cupy(
        em3d.Precision.DOUBLE if PRECISION == "double" else em3d.Precision.SINGLE
    )
    device = cp.cuda.Device()
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")

    # Короткий прогрев cuFFT исключает стоимость инициализации из первого
    # измеряемого решения.
    warmup = cp.zeros(
        (32, 32, 32),
        dtype=cp.complex128 if PRECISION == "double" else cp.complex64,
    )
    _ = cp.fft.fftn(warmup)
    backend.synchronize()
    del warmup
    backend.clear_memory_pool()

    CUDA_INFO = {
        "device_id": int(device.id),
        "device_name": str(device_name),
        "compute_capability": str(device.compute_capability),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": cp.__version__,
        **(backend.memory_info() or {}),
    }
else:
    backend = em3d.Backend.numpy(
        em3d.Precision.DOUBLE if PRECISION == "double" else em3d.Precision.SINGLE
    )
    CUDA_INFO = None

print({
    "em3d_version": em3d.__version__,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "backend": backend.device,
    "precision": backend.precision.value,
    "cuda": CUDA_INFO,
})

# %% [cell 04; notebook index 8]
from dataclasses import replace

from experiments.em_validation import (
    ValidationStudyConfig,
    archive_results,
    publication_config_for_kaggle,
    run_validation_suite,
    write_session_metadata,
)

if MODE == "publication":
    batch_label = (
        f"{MIE_BATCH_AXIS}-batch-{MIE_BATCH_INDEX + 1:02d}-of-{MIE_BATCH_COUNT:02d}"
    )
    RUN_ID = f"chapter4-publication-{batch_label}-{REPOSITORY_COMMIT[:12]}"
    OUTPUT_ROOT = RESULTS_PARENT / RUN_ID
    config, batch_plan = publication_config_for_kaggle(
        output_root=OUTPUT_ROOT,
        precision=PRECISION,
        solver_grid_size=SOLVER_GRID_SIZE,
        mie_grid_sizes=MIE_GRID_SIZES,
        mie_eps_values=MIE_EPS_VALUES,
        mie_k0a_values=MIE_K0A_VALUES,
        mie_batch_axis=MIE_BATCH_AXIS,
        mie_batch_index=MIE_BATCH_INDEX,
        mie_batch_count=MIE_BATCH_COUNT,
        render_progress=True,
        clear_cuda_cache_between_cases=CLEAR_CUDA_CACHE_BETWEEN_CASES,
    )
else:
    RUN_ID = f"chapter4-quick-{REPOSITORY_COMMIT[:12]}"
    OUTPUT_ROOT = RESULTS_PARENT / RUN_ID
    config = ValidationStudyConfig.quick(
        output_root=OUTPUT_ROOT,
        device=DEVICE,
    )
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            precision=PRECISION,
            progress=True,
            clear_cuda_cache_between_cases=CLEAR_CUDA_CACHE_BETWEEN_CASES,
        ),
    )
    batch_plan = None

SESSION_METADATA = {
    "repository_url": REPOSITORY_URL,
    "repository_ref": REPOSITORY_REF,
    "repository_commit": REPOSITORY_COMMIT,
    "mode": MODE,
    "device": DEVICE,
    "precision": PRECISION,
    "run_solver_study": RUN_SOLVER_STUDY,
    "run_mie_study": RUN_MIE_STUDY,
    "batch_plan": None if batch_plan is None else {
        "label": batch_plan.label,
        "axis": batch_plan.axis,
        "index": batch_plan.index,
        "count": batch_plan.count,
        "selected_values": list(batch_plan.selected_values),
    },
    "cuda": CUDA_INFO,
}
write_session_metadata(OUTPUT_ROOT, SESSION_METADATA)

print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
print(config)

# %% [cell 05; notebook index 10]
SMOKE_ROOT = RESULTS_PARENT / f"smoke-{DEVICE}-{REPOSITORY_COMMIT[:12]}"
if RUN_SMOKE_TEST:
    smoke_config = ValidationStudyConfig.quick(
        output_root=SMOKE_ROOT,
        device=DEVICE,
    )
    smoke_config = replace(
        smoke_config,
        runtime=replace(
            smoke_config.runtime,
            precision=PRECISION,
            progress=True,
            clear_cuda_cache_between_cases=True,
        ),
    )
    smoke = run_validation_suite(
        smoke_config,
        include_solver_study=True,
        include_mie_study=True,
        render_figures=False,
    )
    smoke_manifest = json.loads((SMOKE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert smoke_manifest["status"] == "complete"
    if DEVICE == "cuda":
        backend.clear_memory_pool()
    print({
        "smoke_status": smoke_manifest["status"],
        "stationary_cases": len(smoke.stationary_cases),
        "mie_cases": len(smoke.mie_cases),
    })
else:
    print("Smoke test skipped")

# %% [cell 06; notebook index 12]
import traceback

started_at = time.time()
try:
    study = run_validation_suite(
        config,
        include_solver_study=RUN_SOLVER_STUDY,
        include_mie_study=RUN_MIE_STUDY,
        render_figures=RENDER_FIGURES,
    )
except Exception:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "kaggle_failure_traceback.txt").write_text(
        traceback.format_exc(),
        encoding="utf-8",
    )
    raise
finally:
    if DEVICE == "cuda":
        backend.synchronize()

elapsed_seconds = time.time() - started_at
manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete"
assert manifest["git_commit"] == REPOSITORY_COMMIT

SESSION_METADATA["elapsed_seconds"] = elapsed_seconds
SESSION_METADATA["manifest_status"] = manifest["status"]
SESSION_METADATA["artifact_count"] = len(manifest["artifacts"])
SESSION_METADATA["cuda_after_run"] = None if DEVICE != "cuda" else backend.memory_info()
write_session_metadata(OUTPUT_ROOT, SESSION_METADATA)

print({
    "status": manifest["status"],
    "git_commit": manifest["git_commit"],
    "elapsed_seconds": elapsed_seconds,
    "artifact_count": len(manifest["artifacts"]),
    "output_root": str(OUTPUT_ROOT),
})

# %% [cell 07; notebook index 14]
def read_table(name: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / "tables" / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()

protocol = read_table("FINAL_experimental_protocol.csv")
display(protocol)

solver_runs = read_table("solver_runs.csv")
if not solver_runs.empty:
    display(
        solver_runs[[
            "case_key", "solver_name", "status", "qualified",
            "iterations", "matvec_count", "rmatvec_count",
            "operator_action_count", "true_final_residual", "elapsed_seconds",
        ]]
    )

mie_runs = read_table("mie_solver_runs.csv")
if not mie_runs.empty:
    display(
        mie_runs[[
            "eps_real", "k0a", "grid_size", "solver_name",
            "status", "qualified", "operator_action_count",
            "true_final_residual", "elapsed_seconds",
        ]]
    )

mie_selected = read_table("mie_validation_selected.csv")
if not mie_selected.empty:
    display(mie_selected)

# %% [cell 08; notebook index 16]
from IPython.display import Image, display

figure_paths = sorted((OUTPUT_ROOT / "figures").rglob("*.png"))
print(f"PNG figures: {len(figure_paths)}")

# В выводе блокнота показывается ограниченное число рисунков; весь набор
# сохраняется в архиве в PNG/PDF/SVG.
for path in figure_paths[:12]:
    print(path.relative_to(OUTPUT_ROOT))
    display(Image(filename=str(path)))

# %% [cell 09; notebook index 18]
from IPython.display import FileLink, display

ARCHIVE_PATH, ARCHIVE_SHA256 = archive_results(OUTPUT_ROOT)
checksum_path = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".sha256")
checksum_path.write_text(
    f"{ARCHIVE_SHA256}  {ARCHIVE_PATH.name}\n",
    encoding="utf-8",
)

print({
    "archive": str(ARCHIVE_PATH),
    "bytes": ARCHIVE_PATH.stat().st_size,
    "sha256": ARCHIVE_SHA256,
})
display(FileLink(str(ARCHIVE_PATH)))
display(FileLink(str(checksum_path)))
