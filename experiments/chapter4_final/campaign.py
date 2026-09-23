from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from experiments.em_spectral.artifacts import ArtifactStore
from experiments.em_spectral.config import SpectralStudyConfig
from experiments.em_spectral.full_workflow import run_spectral_experiment_suite


CAMPAIGN_STAGES = (
    "spectral",
    "main64",
    "audit",
    "control96",
    "control128",
)


@dataclass(frozen=True)
class CampaignMergeResult:
    output_root: str
    manifest_path: str
    core_archive_path: str | None
    input_roots: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_dataframe(store: ArtifactStore, relative_path: str, frame: pd.DataFrame) -> Path:
    if frame.empty and len(frame.columns) > 0:
        path = store.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return store.record_existing(relative_path, kind="csv")
    records = frame.replace({np.nan: None}).to_dict(orient="records") if not frame.empty else []
    return store.write_rows(relative_path, records)


def _copy_file(store: ArtifactStore, source: Path, relative: Path, *, kind: str | None = None) -> Path:
    target = store.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    store.record_existing(relative, kind=kind)
    return target


def _manifest_kind(manifest: Mapping[str, Any], root: Path) -> str:
    schema = str(manifest.get("schema", ""))
    if "spectral" in schema or (root / "tables" / "E3_summary_phase.csv").is_file():
        return "spectral"
    if "all-solvers" in schema or (root / "tables" / "mie_solver_runs.csv").is_file():
        return "validation"
    return "unknown"


def _infer_stage(root: Path, manifest: Mapping[str, Any], kind: str) -> str:
    session_path = root / "kaggle_session.json"
    if session_path.is_file():
        try:
            session = _read_json(session_path)
            value = session.get("run_stage") or session.get("run_profile")
            if value:
                return str(value)
        except Exception:
            pass
    config = manifest.get("config")
    if isinstance(config, Mapping):
        value = config.get("profile") or config.get("mode")
        if value in CAMPAIGN_STAGES:
            return str(value)
    if kind == "spectral":
        return "spectral"
    lowered = str(root).lower()
    for stage in CAMPAIGN_STAGES:
        if stage in lowered:
            return stage
    return "unknown"


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _solver_domain(common: pd.DataFrame) -> pd.DataFrame:
    if common.empty:
        return pd.DataFrame()
    required = {"eps_real", "eps_imag", "k0a", "grid_size"}
    if not required.issubset(common.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in common.groupby(["eps_real", "eps_imag", "k0a"], dropna=False):
        eps_real, eps_imag, k0a = keys
        configured = sorted({int(value) for value in group["grid_size"]})
        row: dict[str, Any] = {
            "eps_real": eps_real,
            "eps_imag": eps_imag,
            "k0a": k0a,
            "configured_grids": "+".join(str(v) for v in configured),
        }
        common_grids: list[int] = []
        all_column = "all_solvers_qualified"
        if all_column in group:
            common_grids = sorted(
                int(value)
                for value in group.loc[_bool_series(group[all_column]), "grid_size"]
            )
        row["all_solvers_qualified_grids"] = "+".join(str(v) for v in common_grids)
        row["all_solvers_max_grid"] = max(common_grids) if common_grids else None
        for solver, prefix in (("SIM", "sim"), ("BiCGStab", "bicgstab"), ("TwoStep", "twostep")):
            column = f"{prefix}_qualified"
            grids = (
                sorted(int(value) for value in group.loc[_bool_series(group[column]), "grid_size"])
                if column in group
                else []
            )
            row[f"{prefix}_qualified_grids"] = "+".join(str(v) for v in grids)
            row[f"{prefix}_max_grid"] = max(grids) if grids else None
            row[f"{prefix}_qualified_count"] = len(grids)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["eps_real", "k0a"]).reset_index(drop=True)


def _mie_error_summary(rcs: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not rcs.empty:
        data = rcs.copy()
        if "radius_label" in data:
            data = data[data["radius_label"].astype(str) == "effective"]
        metrics = [name for name in ("normalized_l2", "absolute_l2", "absolute_linf") if name in data]
        if metrics:
            group = [name for name in ("tier", "grid_size", "solver_name") if name in data]
            summary = data.groupby(group, dropna=False)[metrics].agg(["median", "max"]).reset_index()
            summary.columns = [
                "_".join(str(part) for part in value if str(part)) if isinstance(value, tuple) else str(value)
                for value in summary.columns
            ]
            summary["metric_family"] = "rcs_effective_radius"
            frames.append(summary)
    if not fields.empty:
        data = fields.copy()
        if "radius_label" in data:
            data = data[data["radius_label"].astype(str) == "effective"]
        candidates = (
            "field_relative_l2",
            "scattered_field_relative_l2",
            "scattered_field_outside_relative_l2",
        )
        metrics = [name for name in candidates if name in data]
        if metrics:
            group = [name for name in ("tier", "grid_size", "solver_name") if name in data]
            summary = data.groupby(group, dropna=False)[metrics].agg(["median", "max"]).reset_index()
            summary.columns = [
                "_".join(str(part) for part in value if str(part)) if isinstance(value, tuple) else str(value)
                for value in summary.columns
            ]
            summary["metric_family"] = "near_field_effective_radius"
            frames.append(summary)
    if not frames:
        return pd.DataFrame()
    columns = sorted(set().union(*(frame.columns for frame in frames)))
    return pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)


def _stationary_stress_summary(runs: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "case_key" not in runs:
        return pd.DataFrame(
            columns=(
                "job_key", "grid_size", "variant_label", "parameter_strategy",
                "coarse_sizes", "solver_name", "status", "qualified",
                "operator_action_count", "true_final_residual", "circle_q",
                "circle_margin", "elapsed_seconds",
            )
        )
    data = runs[runs["case_key"].astype(str) == "local_inclusion_stress"].copy()
    if data.empty:
        return data
    keep = [
        name
        for name in (
            "job_key",
            "grid_size",
            "variant_label",
            "parameter_strategy",
            "coarse_sizes",
            "solver_name",
            "status",
            "qualified",
            "operator_action_count",
            "true_final_residual",
            "circle_q",
            "circle_margin",
            "elapsed_seconds",
        )
        if name in data
    ]
    data = data[keep]
    if not params.empty and "job_key" in params:
        p = params.copy()
        if "parameter_strategy" in p:
            p = p[p["parameter_strategy"].astype(str).isin({"single_coarse", "coarse_ensemble_result"})]
        pkeep = [
            name
            for name in (
                "job_key",
                "q",
                "margin",
                "delta_latest",
                "delta_max",
                "q_inflated_latest",
                "q_inflated_max",
                "inflated_latest_safe",
                "inflated_max_safe",
            )
            if name in p
        ]
        if pkeep:
            p = p[pkeep].drop_duplicates("job_key", keep="last")
            data = data.merge(p, how="left", on="job_key", suffixes=("", "_parameter"))
    return data.sort_values(["job_key", "solver_name"]).reset_index(drop=True)


def _audit_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "tier" not in runs:
        return pd.DataFrame(
            columns=(
                "eps_real", "eps_imag", "k0a", "grid_size", "solver_name",
                "true_final_residual_rtol1e-6",
                "true_final_residual_rtol1e-8",
                "operator_action_count_rtol1e-6",
                "operator_action_count_rtol1e-8",
            )
        )
    main = runs[runs["tier"].astype(str) == "main"].copy()
    audit = runs[runs["tier"].astype(str) == "audit"].copy()
    keys = [name for name in ("eps_real", "eps_imag", "k0a", "grid_size", "solver_name") if name in runs]
    if main.empty or audit.empty or not keys:
        return pd.DataFrame(
            columns=(
                "eps_real", "eps_imag", "k0a", "grid_size", "solver_name",
                "true_final_residual_rtol1e-6",
                "true_final_residual_rtol1e-8",
                "operator_action_count_rtol1e-6",
                "operator_action_count_rtol1e-8",
            )
        )
    main_cols = keys + [name for name in ("true_final_residual", "operator_action_count", "elapsed_seconds", "qualified") if name in main]
    audit_cols = keys + [name for name in ("true_final_residual", "operator_action_count", "elapsed_seconds", "qualified") if name in audit]
    return main[main_cols].merge(
        audit[audit_cols],
        on=keys,
        how="inner",
        suffixes=("_rtol1e-6", "_rtol1e-8"),
    )


def _coverage_rows() -> list[dict[str, Any]]:
    return [
        {
            "chapter_section": "4.2",
            "question": "форма спектра при изменении физической постановки и волнового числа",
            "required_stage": "spectral",
            "primary_tables": "S42_fixed_grid_spectrum_scan; spectral_localizations; E3_spectral_phase; E3_summary_phase",
            "status_rule": "spectral manifest complete",
        },
        {
            "chapter_section": "4.3",
            "question": "устойчивость переноса параметра между дискретизациями",
            "required_stage": "spectral",
            "primary_tables": "control_transfers; fine_transfers; E4; E5; E6",
            "status_rule": "E0--E6 protocol complete",
        },
        {
            "chapter_section": "4.4",
            "question": "корректность дальней зоны и расчёта ЭПР",
            "required_stage": "main64",
            "primary_tables": "mie_nearfield_gate; farfield_backend_consistency; mie_solver_vs_mie_rcs",
            "status_rule": "near-field gate passed and RCS tables present",
        },
        {
            "chapter_section": "4.5",
            "question": "сходимость методов, поля и ЭПР для задач без аналитического решения",
            "required_stage": "main64; control96; control128",
            "primary_tables": "stationary_solver_runs; stationary_solver_pairwise_fields; stationary_solver_pairwise_rcs",
            "status_rule": "single-N5 and ensemble stress variants present",
        },
        {
            "chapter_section": "4.6",
            "question": "аналитическая верификация по задаче Ми всеми тремя схемами",
            "required_stage": "main64; audit; control96; control128",
            "primary_tables": "mie_common_solvability; mie_solver_vs_mie_fields; mie_solver_vs_mie_rcs",
            "status_rule": "all planned grids recorded, including failures as scientific outcomes",
        },
        {
            "chapter_section": "4.7",
            "question": "границы применимости и сводные выводы",
            "required_stage": "merge",
            "primary_tables": "chapter4_solver_domain; chapter4_mie_error_summary; chapter4_stationary_stress_summary",
            "status_rule": "campaign merge complete",
        },
    ]


def _copy_selected_figures(store: ArtifactStore, roots: Sequence[tuple[str, Path]]) -> pd.DataFrame:
    rules = (
        ("spectral", "figures/S42/real_anisotropy_spectra.*"),
        ("spectral", "figures/S42/lossy_anisotropy_spectra.*"),
        ("spectral", "figures/S42/real_anisotropy_scan.*"),
        ("spectral", "figures/S42/lossy_anisotropy_scan.*"),
        ("spectral", "figures/E3_phase_article.*"),
        ("spectral", "figures/E4_geometry_ratio.*"),
        ("spectral", "figures/E5_ensemble_ratio.*"),
        ("spectral", "figures/E6_volume_averaging_improvement.*"),
        ("spectral", "figures/core/local_inclusion*control_transfer.*"),
        ("validation", "figures/stationary_residuals/local_inclusion_stress_single_N5_N64.*"),
        ("validation", "figures/stationary_residuals/local_inclusion_stress_ensemble_5_6_7_N64.*"),
        ("validation", "figures/stationary_rcs/local_inclusion_stress_ensemble_5_6_7_N64_xz.*"),
        ("validation", "figures/mie_convergence/*eps4*k0a4*.*"),
        ("validation", "figures/mie_rcs/mie_eps1p5_k0a0p5_N64_xz.*"),
        ("validation", "figures/mie_rcs/mie_eps2p25_k0a2_N64_xz.*"),
        ("validation", "figures/mie_rcs/mie_eps4_k0a4_N64_xz.*"),
        ("validation", "figures/mie_fields/mie_eps4_k0a4_N64_xz_overview.*"),
    )
    rows: list[dict[str, Any]] = []
    for label, root in roots:
        for required_label, pattern in rules:
            if label != required_label:
                continue
            for source in sorted(root.glob(pattern)):
                if not source.is_file():
                    continue
                relative = Path("figures") / "selected" / f"{root.name}__{source.name}"
                target = _copy_file(store, source, relative)
                rows.append(
                    {
                        "source_run": root.name,
                        "source_path": str(source.relative_to(root)),
                        "selected_path": str(target.relative_to(store.root)),
                        "format": target.suffix.lstrip("."),
                        "bytes": target.stat().st_size,
                    }
                )
    return pd.DataFrame(rows)


def _merge_validation_tables(roots: Sequence[Path]) -> dict[str, pd.DataFrame]:
    """Merge CSV tables without duplicating large raw/figure trees."""

    names: set[str] = set()
    for root in roots:
        names.update(path.stem for path in (root / "tables").glob("*.csv"))
    merged: dict[str, pd.DataFrame] = {}
    for name in sorted(names):
        frames = [_read_csv(root / "tables" / f"{name}.csv") for root in roots]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            merged[name] = pd.DataFrame()
            continue
        frame = pd.concat(frames, ignore_index=True, sort=False)
        # String-normalized duplicate detection remains stable when one batch
        # infers an integer column and another infers the same column as float.
        marker_frame = frame.astype(str)
        frame = frame.loc[~marker_frame.duplicated()].reset_index(drop=True)
        merged[name] = frame
    return merged


def run_spectral_stage(
    *,
    output_root: str | Path,
    device: str = "cuda",
    render_figures: bool = True,
    quick: bool = False,
) -> Path:
    config = (
        SpectralStudyConfig.quick(output_root=output_root, device=device)
        if quick
        else SpectralStudyConfig.publication(output_root=output_root, device=device)
    )
    run_spectral_experiment_suite(
        config,
        include_core=True,
        include_geometry=True,
        include_volume_averaging=True,
        include_ensemble=True,
        include_wave_number=True,
        include_fine=True,
        include_core_arnoldi=True,
        include_boundary_arnoldi=True,
        render_figures=render_figures,
    )
    return Path(output_root) / "manifest.json"


def merge_chapter4_campaign(
    inputs: Sequence[str | Path],
    *,
    output_root: str | Path,
    create_core: bool = True,
    require_complete_campaign: bool = True,
) -> CampaignMergeResult:
    roots = tuple(Path(value).resolve() for value in inputs)
    if not roots:
        raise ValueError("at least one input run is required")

    inventory: list[dict[str, Any]] = []
    spectral_roots: list[Path] = []
    validation_roots: list[Path] = []
    commits: set[str] = set()
    for root in roots:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"input run is not complete: {root}")
        kind = _manifest_kind(manifest, root)
        stage = _infer_stage(root, manifest, kind)
        commit = manifest.get("git_commit")
        if commit:
            commits.add(str(commit))
        inventory.append(
            {
                "run_name": root.name,
                "root": str(root),
                "kind": kind,
                "stage": stage,
                "schema": manifest.get("schema"),
                "status": manifest.get("status"),
                "git_commit": commit,
                "em3d_version": manifest.get("em3d_version"),
                "config_sha256": manifest.get("config_sha256"),
                "artifact_count": len(manifest.get("artifacts", [])),
            }
        )
        if kind == "spectral":
            spectral_roots.append(root)
        elif kind == "validation":
            validation_roots.append(root)
        else:
            raise ValueError(f"cannot classify campaign input {root}")
    if len(commits) > 1:
        raise ValueError(f"campaign inputs use different commits: {sorted(commits)}")
    if not spectral_roots:
        raise ValueError("the final campaign requires one spectral run")
    if not validation_roots:
        raise ValueError("the final campaign requires validation runs")
    present_stages = {str(row["stage"]) for row in inventory}
    missing_stages = [stage for stage in CAMPAIGN_STAGES if stage not in present_stages]
    if require_complete_campaign and missing_stages:
        raise ValueError(
            "incomplete Chapter 4 campaign; missing stages: "
            + ", ".join(missing_stages)
        )

    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    store = ArtifactStore(output, schema="em3d-chapter4-final-campaign-v1")
    store.log_event(
        "campaign_merge_start",
        inputs=len(roots),
        spectral_runs=len(spectral_roots),
        validation_runs=len(validation_roots),
    )

    # Keep every source manifest/config/session in the compact final tree.
    for root in roots:
        source_dir = Path("sources") / root.name
        for filename in ("manifest.json", "config.json", "kaggle_session.json"):
            source = root / filename
            if source.is_file():
                _copy_file(store, source, source_dir / filename)
        if (root / "logs").is_dir():
            for source in (root / "logs").rglob("*"):
                if source.is_file():
                    _copy_file(
                        store,
                        source,
                        source_dir / "logs" / source.relative_to(root / "logs"),
                    )

    # Merge validation tables only.  The source runs retain the very large
    # field-slice and figure trees; the final campaign copies only compact
    # evidence and a curated figure subset.
    merged_validation_tables = _merge_validation_tables(validation_roots)

    # Copy source tables into namespaced locations without flattening names.
    for root in spectral_roots:
        for source in sorted((root / "tables").glob("*.csv")):
            _copy_file(store, source, Path("tables/spectral") / source.name)
        for source in sorted((root / "raw" / "tables").glob("*.json")):
            _copy_file(store, source, Path("raw/source_tables/spectral") / source.name)
    for name, frame in merged_validation_tables.items():
        _write_dataframe(store, f"tables/validation/{name}.csv", frame)
        store.write_json(
            f"raw/source_tables/validation/{name}.json",
            frame.replace({np.nan: None}).to_dict(orient="records")
            if not frame.empty
            else [],
        )

    # Compact raw evidence needed for independent re-analysis.  Volumetric
    # field-slice arrays stay in the source run archives because they dominate
    # download size.
    for root in validation_roots:
        for subdir in ("raw/residual_histories", "raw/rcs", "raw/mie_gate"):
            source_root = root / subdir
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*"):
                if source.is_file():
                    _copy_file(
                        store,
                        source,
                        Path("raw/evidence") / root.name / source.relative_to(root / "raw"),
                    )

    common = merged_validation_tables.get("mie_common_solvability", pd.DataFrame())
    mie_runs = merged_validation_tables.get("mie_solver_runs", pd.DataFrame())
    rcs = merged_validation_tables.get("mie_solver_vs_mie_rcs", pd.DataFrame())
    fields = merged_validation_tables.get("mie_solver_vs_mie_fields", pd.DataFrame())
    stationary_runs = merged_validation_tables.get("stationary_solver_runs", pd.DataFrame())
    stationary_params = merged_validation_tables.get("stationary_sim_parameters", pd.DataFrame())

    inventory_frame = pd.DataFrame(inventory)
    _write_dataframe(store, "tables/chapter4_run_inventory.csv", inventory_frame)
    _write_dataframe(store, "tables/chapter4_experiment_coverage.csv", pd.DataFrame(_coverage_rows()))
    _write_dataframe(store, "tables/chapter4_solver_domain.csv", _solver_domain(common))
    _write_dataframe(store, "tables/chapter4_mie_error_summary.csv", _mie_error_summary(rcs, fields))
    _write_dataframe(
        store,
        "tables/chapter4_stationary_stress_summary.csv",
        _stationary_stress_summary(stationary_runs, stationary_params),
    )
    _write_dataframe(store, "tables/chapter4_tolerance_audit.csv", _audit_comparison(mie_runs))

    selected = _copy_selected_figures(
        store,
        [("spectral", root) for root in spectral_roots]
        + [("validation", root) for root in validation_roots],
    )
    _write_dataframe(store, "tables/chapter4_selected_figures.csv", selected)

    campaign_payload = {
        "schema": "em3d-chapter4-final-campaign-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_stages": list(CAMPAIGN_STAGES),
        "present_stages": sorted(present_stages),
        "missing_stages": missing_stages,
        "complete_campaign": not missing_stages,
        "inputs": inventory,
        "common_git_commit": next(iter(commits)) if commits else None,
        "excluded_from_core_archive": [
            "full figure trees",
            "volumetric field-slice npz arrays",
            "duplicated source archives",
        ],
    }
    store.write_json("campaign.json", campaign_payload)
    manifest = store.finalize(config=campaign_payload, status="complete")
    core_path: Path | None = None
    if create_core:
        core_path, _digest = create_core_archive(store.root)
    return CampaignMergeResult(
        output_root=str(store.root),
        manifest_path=str(manifest),
        core_archive_path=str(core_path) if core_path else None,
        input_roots=tuple(str(root) for root in roots),
    )


def create_core_archive(
    root: str | Path,
    *,
    destination: str | Path | None = None,
) -> tuple[Path, str]:
    """Create a compact archive suitable for sharing and dissertation analysis."""

    source = Path(root).resolve()
    if destination is None:
        destination_path = source.parent / f"{source.name}-core.zip"
    else:
        destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    include_roots = ("tables", "logs", "sources", "raw/source_tables", "raw/evidence", "figures/selected")
    top_files = ("manifest.json", "campaign.json", "config.json", "kaggle_session.json")
    with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in top_files:
            path = source / name
            if path.is_file():
                archive.write(path, arcname=f"{source.name}/{name}")
        for relative_root in include_roots:
            path_root = source / relative_root
            if not path_root.is_dir():
                continue
            for path in sorted(path_root.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=f"{source.name}/{path.relative_to(source)}")
    digest = hashlib.sha256(destination_path.read_bytes()).hexdigest()
    destination_path.with_suffix(destination_path.suffix + ".sha256").write_text(
        f"{digest}  {destination_path.name}\n", encoding="utf-8"
    )
    return destination_path, digest
