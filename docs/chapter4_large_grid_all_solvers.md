# Chapter 4 large-grid all-solver validation

This workflow extends the stationary electrodynamic validation suite to a common
protocol for `SIM`, `BiCGStab`, and `TwoStep`.  Every solver is applied to the
same discretised Mie or non-analytic scattering problem and is qualified by an
independently recomputed true relative residual.

## Run profiles

```text
smoke       small CPU/CUDA end-to-end check
main64      full Mie tensor product on N=24,32,48,64 plus N=64 stationary cases
audit       selected N=48,64 cases at true residual tolerance 1e-8
control96   five Mie controls and three stationary controls on N=96
control128  three mandatory Mie controls and two stationary controls on N=128
merge       merge independent Kaggle job/profile outputs
```

The main Mie series contains

```text
eps_r = 1.5, 2.25, 4.0
k0 a  = 0.5, 1, 2, 4, 6
N     = 24, 32, 48, 64
solver = SIM, BiCGStab, TwoStep
```

`control96` and `control128` use memory probes and are intended to be run as
isolated Kaggle subprocesses.  `TwoStep` uses a derived adjoint FFT kernel, so a
second persistent nine-block spectral tensor is not stored.

## CLI

```bash
python -m experiments.em_validation.large_cli \
  --profile main64 \
  --device cuda \
  --precision double \
  --output-root /kaggle/working/chapter4-main64
```

Inspect deterministic job keys:

```bash
python -m experiments.em_validation.large_cli \
  --profile control128 \
  --device cuda \
  --output-root /tmp/unused \
  --list-jobs
```

Run one isolated control job:

```bash
python -m experiments.em_validation.large_cli \
  --profile control128 \
  --device cuda \
  --precision double \
  --job-key mie_eps4_k0a4_N128 \
  --output-root /kaggle/working/jobs/mie_eps4_k0a4_N128
```

Merge completed job roots:

```bash
python -m experiments.em_validation.large_cli \
  --profile merge \
  --output-root /kaggle/working/chapter4-control128-merged \
  --merge-input /kaggle/working/jobs/mie_eps1p5_k0a0p5_N128 \
  --merge-input /kaggle/working/jobs/mie_eps2p25_k0a6_N128 \
  --merge-input /kaggle/working/jobs/mie_eps4_k0a4_N128
```

## Numerical safeguards

1. The analytical Mie near field is checked against the no-scatterer limit and
   its own far-field RCS before numerical fields are interpreted.
2. `SIM` uses an ensemble localization from independently sampled coarse grids.
   An unavailable circle is recorded as `no_parameter`; it is never replaced by
   a parameter fitted on the fine grid.
3. Solver success requires both the solver flag and the true residual threshold.
4. Operator work is compared through `matvec_count + rmatvec_count`.
5. The N=96/128 profiles record analytical and measured memory checkpoints.
6. CUDA OOM is a job status, not a reason to lose previous completed jobs.

## Principal artifacts

```text
tables/mie_solver_runs.csv
tables/mie_common_solvability.csv
tables/mie_solver_vs_mie_fields.csv
tables/mie_solver_vs_mie_rcs.csv
tables/mie_observed_orders_by_solver.csv
tables/gpu_memory_probe.csv
tables/gpu_memory_checkpoints.csv
tables/figure_index.csv

archives/mie_field_figures.zip
archives/mie_rcs_figures.zip
archives/stationary_field_figures.zip
archives/stationary_rcs_figures.zip
archives/publication_selected_figures.zip
```

Each Mie RCS figure contains absolute and normalized curves in Cartesian and
polar coordinates.  Each Mie field archive contains total and scattered fields,
a selected scattered component, phase, and error against the effective-radius
Mie reference in the `xy`, `xz`, and `yz` central planes.
