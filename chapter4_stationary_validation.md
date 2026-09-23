# Generated from a Jupyter notebook; edit the notebook, not this file.
# Source: notebooks/em-solver-farfield-validation.ipynb
# SHA-256: df012e81522bf5e281f931e09757b30f77d62c4285b006d54cca8e4491a7e2b3

# %% [cell 01; notebook index 2]
from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import Image, display

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "experiments").is_dir() and (REPO_ROOT.parent / "experiments").is_dir():
    REPO_ROOT = REPO_ROOT.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.em_validation import ValidationStudyConfig, run_validation_suite

# %% [cell 02; notebook index 3]
MODE = "quick"          # заменить на "publication" для итоговой серии
DEVICE = "cpu"          # "cpu", "cuda" или "auto"
RUN_SOLVER_STUDY = True
RUN_MIE_STUDY = True
RENDER_FIGURES = True

OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / "em_solver_farfield_validation_notebook"
factory = ValidationStudyConfig.quick if MODE == "quick" else ValidationStudyConfig.publication
config = factory(output_root=OUTPUT_ROOT, device=DEVICE)
config

# %% [cell 03; notebook index 5]
study = run_validation_suite(
    config,
    include_solver_study=RUN_SOLVER_STUDY,
    include_mie_study=RUN_MIE_STUDY,
    render_figures=RENDER_FIGURES,
)
study

# %% [cell 04; notebook index 6]
def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_ROOT / "tables" / name)

manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
display({
    "status": manifest["status"],
    "schema": manifest["schema"],
    "git_commit": manifest["git_commit"],
    "em3d_version": manifest["em3d_version"],
    "artifact_count": len(manifest["artifacts"]),
})
display(read_table("FINAL_experimental_protocol.csv"))

# %% [cell 05; notebook index 8]
solver_runs = read_table("solver_runs.csv")
display(solver_runs[[
    "case_key", "solver_name", "status", "qualified",
    "iterations", "matvec_count", "rmatvec_count",
    "operator_action_count", "true_final_residual", "elapsed_seconds",
]])

# %% [cell 06; notebook index 9]
for path in sorted((OUTPUT_ROOT / "figures" / "solver").glob("*.png")):
    display(Image(filename=str(path)))

# %% [cell 07; notebook index 11]
display(read_table("field_comparisons.csv"))
for path in sorted((OUTPUT_ROOT / "figures" / "field").glob("*.png")):
    display(Image(filename=str(path)))

# %% [cell 08; notebook index 13]
display(read_table("rcs_solver_comparisons.csv"))
for path in sorted((OUTPUT_ROOT / "figures" / "rcs").glob("*.png")):
    display(Image(filename=str(path)))

# %% [cell 09; notebook index 15]
mie_runs = read_table("mie_solver_runs.csv")
mie_validation = read_table("mie_validation.csv")
mie_selected = read_table("mie_validation_selected.csv")
display(mie_runs[[
    "eps_real", "k0a", "grid_size", "solver_name", "status",
    "qualified", "operator_action_count", "true_final_residual",
]])
display(mie_validation[[
    "eps_real", "k0a", "grid_size", "solver_name",
    "relative_volume_error", "nominal_field_relative_l2", "nominal_scattered_field_outside_relative_l2",
    "nominal_rcs_normalized_l2", "nominal_rcs_absolute_l2",
    "nominal_rcs_peak_angle_error_degrees",
    "effective_rcs_normalized_l2", "farfield_backend_relative_l2",
]])


display(mie_selected)

# %% [cell 10; notebook index 16]
for path in sorted((OUTPUT_ROOT / "figures" / "mie").glob("*.png")):
    display(Image(filename=str(path)))
for subdirectory in ("fields", "curves"):
    for path in sorted((OUTPUT_ROOT / "figures" / "mie" / subdirectory).glob("*.png")):
        display(Image(filename=str(path)))

# %% [cell 11; notebook index 19]
artifacts = pd.DataFrame(manifest["artifacts"])
display(artifacts.groupby("kind").size().rename("count").to_frame())
display(artifacts.head(20))
