# Spectral-transfer experiment package

The spectral-transfer study is implemented as a reproducible package workflow.
The notebook is a report layer and does not contain an independent operator,
solver, convex-hull implementation or Arnoldi process.

## Package layers

- `em3d.layout` fixes one field-vector ordering for dense and matrix-free code.
- `em3d.geometry` builds cell-centred and volume-fraction material samples from
  continuous geometry definitions.
- `em3d.spectral` contains convex spectral geometry, localization circles,
  transfer bounds, convergence diagnostics and matrix-free Arnoldi.
- `PreparedEMKernel` allows material cases with the same grid and wave number to
  reuse one FFT kernel.
- `em3d.experiments.spectral_transfer` is a pandas/matplotlib-free API for
  dense/FFT consistency, full spectra, inter-grid transfer, sampling metrics,
  ensemble localization and single/multi-start Arnoldi diagnostics.
- `experiments.em_spectral` contains configuration, artifact persistence,
  chapter-level experiment workflows, reporting plots and the command-line API.
- `notebooks/em-spectral-transfer.ipynb` is a thin, executed report.

## Experiment map

| Block | Packaged workflow | Main outputs |
|---|---|---|
| E0--E2B | `run_spectral_transfer_study` | dense/FFT error, transfer bounds, fine-grid residuals, Arnoldi diagnostics |
| E4 | `run_geometry_resolution_study` | geometry resolution, control certificate and fine-grid classification |
| E6 | `run_volume_averaging_study` | exact volume diagnostics and centre-vs-average comparison |
| E5 | `run_ensemble_transfer_study` | ensemble localization, robust indicators and fine-grid comparison |
| E3/E3b | `run_wave_number_phase_study` | detailed phase map, article projection, PPW diagnostics and boundary Arnoldi |
| all | `run_spectral_experiment_suite` | one manifest and all E0--E6 artifacts |

E6 and E5 consume the spectra already computed by E4. They do not repeat the
same centre-sampled eigensolutions.

## Phase semantics

The raw wave-number results distinguish:

- `converged`: the requested residual tolerance was reached;
- `contracting_unresolved`: an asymptotic contraction was observed but the
  requested tolerance was not reached in the allotted iterations;
- `unstable`;
- `unresolved`;
- `nonfinite`;
- `no_parameter`.

The historical three-state diagram is exported separately in
`article_phase_code`. It is a declared projection, not the primary numerical
classification.

## Compatibility

The existing `gamma0` API, `Problem(eps_tensor=...)`, `Operator.to_dense()` and
solver constructors remain available. New code should use:

- `Problem.contrast_tensor` for the stored tensor `chi = eps_r - I`;
- `Operator.to_dense_kernel()` for `B`;
- `Operator.to_dense_operator()` for `A = I - B chi`.

The historical `gamma0.coarse_operator_matrix()` retains nearest-cell
resampling for backward compatibility. New inter-grid experiments rebuild each
grid independently from `SpectralCaseDefinition`.

## Running

Core quick CPU run:

```bash
python -m experiments.em_spectral.cli \
  --mode quick --device cpu --study core --no-arnoldi
```

Complete quick run with dissertation figures:

```bash
python -m experiments.em_spectral.cli \
  --mode quick --device cpu --study all --no-arnoldi --figures
```

Publication configuration:

```bash
python -m experiments.em_spectral.cli \
  --mode publication --device auto --study all --figures
```

The complete run creates one `manifest.json`, CSV tables, compressed spectra,
residual histories, Arnoldi Hessenberg matrices where requested, and PNG/PDF/SVG
figures.

## Notebook code for the dissertation appendix

```bash
python tools/export_notebook_code.py \
  notebooks/em-spectral-transfer.ipynb \
  appendix/em-spectral-transfer.py
```

The export contains code cells only, stable cell-boundary markers and the
SHA-256 hash of the source notebook.
