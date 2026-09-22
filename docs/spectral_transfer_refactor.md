# Spectral-transfer refactor

This change moves the reusable numerical logic of `sim-spectral-stability.ipynb`
out of the notebook and into the `em3d` package.

## Package layers

- `em3d.layout` fixes one field-vector ordering for dense and matrix-free code.
- `em3d.geometry` builds cell-centred and volume-fraction material samples from
  continuous geometry definitions.
- `em3d.spectral` contains convex spectral geometry, localization circles,
  transfer bounds, convergence diagnostics and matrix-free Arnoldi.
- `PreparedEMKernel` allows several material cases with the same grid and wave
  number to reuse one FFT kernel.
- `em3d.experiments.spectral_transfer` is a pandas/matplotlib-free experiment
  API for dense/FFT consistency, full spectra on affordable grids, transfer to a
  work grid and Arnoldi diagnostics.
- `experiments.em_spectral` contains configuration, artifact persistence,
  reporting plots and the command-line workflow.
- `notebooks/em-spectral-transfer.ipynb` is a thin executable report. It does
  not define the operator, spectral algorithms, solver or Arnoldi process.

## Compatibility

The existing `gamma0` API, `Problem(eps_tensor=...)`, `Operator.to_dense()` and
solver constructors remain available. New code should use:

- `Problem.contrast_tensor` for the stored tensor `chi = eps_r - I`;
- `Operator.to_dense_kernel()` for `B`;
- `Operator.to_dense_operator()` for `A = I - B chi`.

The historical `gamma0.coarse_operator_matrix()` still uses nearest-cell
resampling for backward compatibility. New inter-grid experiments rebuild every
grid independently from `SpectralCaseDefinition`.

## Running

Quick CPU smoke run:

```bash
python -m experiments.em_spectral.cli \
  --mode quick --device cpu --no-fine --no-arnoldi
```

Publication configuration:

```bash
python -m experiments.em_spectral.cli \
  --mode publication --device auto
```

The run creates a versioned `manifest.json`, CSV tables, compressed spectra,
residual histories and Arnoldi Hessenberg matrices.

## Notebook code for the dissertation appendix

```bash
python tools/export_notebook_code.py \
  notebooks/em-spectral-transfer.ipynb \
  appendix/em-spectral-transfer.py
```

The export contains code cells only, stable cell boundary markers and the
SHA-256 hash of the source notebook.

## Scope of this commit

The commit provides the common implementation required by E0, E1, E2 and
E2A/E2B and the geometry/sampling and ensemble primitives needed by E3-E6.
The publication-scale parameter sweeps and their final dissertation figures
should be migrated as a separate experiment-only commit after regression
comparison with the saved version-8b notebook outputs.
