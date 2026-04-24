# em3d

**Volume-integral-equation solver for 3-D electrodynamics on structured Cartesian grids with FFT acceleration.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

`em3d` solves the volume integral equation (VIE) of 3-D electromagnetic scattering:

```
(I + B·η) u = f
```

| Symbol | Meaning |
|--------|---------|
| **u**(r) | scattered electric field inside the dielectric |
| **f**(r) | incident plane wave |
| η(r) = ε(r) − I | dielectric contrast tensor |
| **B** | volume-integral operator with Helmholtz Green's function G(R) = exp(ikR) / (4πR) |

The operator **B** is applied as an FFT convolution on the **doubled parallelepiped Π₂**, which eliminates wrap-around artefacts and gives **O(N log N)** cost per iteration.

---

## Features

- **FFT matvec** — doubled-grid Π₂ Toeplitz convolution; no dense matrix stored.
- **Three iterative solvers** — SIM/MSGD, BiCGStab, TwoStep gradient descent.
- **γ₀ parameter** — optimal iteration parameter via convex hull + smallest enclosing circle of the operator spectrum.
- **Dual backend** — NumPy (CPU) and CuPy (CUDA GPU) with identical API.
- **Dual precision** — `float64/complex128` (double) and `float32/complex64` (single).
- **Typed** — PEP 561 `py.typed` marker, fully annotated public API.

---

## Installation

### Stable release (v0.1.0) from GitHub

```bash
pip install git+https://github.com/qwerty29544/em3d.git@v0.1.0
```

### Latest development version

```bash
pip install git+https://github.com/qwerty29544/em3d.git
```

### With GPU support (requires CUDA 12 and CuPy)

```bash
pip install "git+https://github.com/qwerty29544/em3d.git@v0.1.0[gpu]"
```

### Local editable install (for development)

```bash
git clone https://github.com/qwerty29544/em3d.git
cd em3d
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
import em3d

# ── 1. Backend and grid ────────────────────────────────────────────────────
be   = em3d.Backend.numpy(em3d.Precision.DOUBLE)
grid = em3d.Grid(N=(16, 16, 16), L=(1.0, 1.0, 1.0),
                 center=(0.0, 0.0, 0.0), backend=be)

# ── 2. Dielectric cylinder along z (ε_r = 2, no absorption) ───────────────
scalar_eta = em3d.cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0,
                                       radius=0.3, axis="z")
eps_tensor = em3d.apply_refraction(grid, scalar_eta=scalar_eta)

# ── 3. Incident plane wave (E ∥ x̂, propagation along ẑ, k₀ = 1) ──────────
wave = em3d.flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))

# ── 4. Assemble problem and operator ──────────────────────────────────────
problem = em3d.Problem(grid=grid, eps_tensor=eps_tensor,
                       wave=wave, k0=1.0, volume=grid.dv * 16**3)
op = em3d.Operator(problem)

# ── 5. Solve with BiCGStab (no extra parameters needed) ───────────────────
cfg    = em3d.SolverConfig(max_iter=500, rtol=1e-8)
result = em3d.BiCGStab(cfg).solve(op, wave)

print(f"Converged: {result.converged}  iterations: {result.iterations}")
u = np.asarray(result.u)   # shape (3, 16, 16, 16), complex128
```

### Using SIM with optimal γ₀

```python
# Sample a few diagonal eigenvalue estimates of B·η
import em3d.gamma0 as g0

rng    = np.random.default_rng(0)
probes = [be.array(rng.standard_normal((3,) + grid.N).astype(np.complex128))]
samples = []
for v in probes:
    Av  = op.matvec(v)
    eta_v = np.einsum("ab...,b...->a...", np.asarray(problem.eps_tensor),
                      np.asarray(v))
    # Rayleigh-like quotient per component
    for idx in np.ndindex(*grid.N):
        s = (np.asarray(Av)[(slice(None),) + idx]
             - np.asarray(v)[(slice(None),) + idx])
        e = eta_v[(slice(None),) + idx]
        if np.linalg.norm(e) > 1e-12:
            samples.append(complex(np.vdot(e, s) / np.vdot(e, e)))
        if len(samples) >= 20:
            break
    if len(samples) >= 20:
        break

params  = g0.find_params(samples)
cfg_sim = em3d.SolverConfig(max_iter=500, rtol=1e-8,
                             mu=params["mu"], radius=params["radius"])
result  = em3d.SIM(cfg_sim).solve(op, wave)
```

### GPU backend

```python
be_gpu   = em3d.Backend.cupy(em3d.Precision.DOUBLE)   # requires CuPy
grid_gpu = em3d.Grid(N=(32, 32, 32), L=(1.0, 1.0, 1.0),
                     center=(0.0, 0.0, 0.0), backend=be_gpu)
# everything else is identical — the operator lives on the GPU
```

---

## Module reference

| Module | Public symbols | Purpose |
|--------|---------------|---------|
| `em3d` | see `__all__` | Top-level re-exports |
| `em3d.backend` | `Backend`, `Precision` | NumPy/CuPy abstraction, dtype pairs |
| `em3d.grid` | `Grid` | Structured Cartesian grid, cell volumes, coordinates |
| `em3d.refraction` | `cylinder_refraction`, `step_refraction`, `ellipsis_refraction`, `apply_refraction` | Build dielectric contrast tensor η |
| `em3d.wave` | `flat_wave_vec` | Sample a plane wave on the grid |
| `em3d.problem` | `Problem` | Container: grid + ε-tensor + wave + k₀ |
| `em3d.operator` | `Operator` | FFT VIE operator: `matvec` (A), `rmatvec` (A†), `to_dense` |
| `em3d.gamma0` | `find_params`, `sequential_chain`, `compute_circle_*` | γ₀ via convex hull + smallest enclosing circle |
| `em3d.solvers` | `SIM`, `BiCGStab`, `TwoStep`, `SolverConfig`, `SolverResult`, `BaseSolver` | Iterative solvers |

### `SolverConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `max_iter` | `200` | Maximum number of iterations |
| `rtol` | `1e-6` | Relative residual tolerance `‖Au−f‖/‖f‖` |
| `log` | `False` | Print residuals each iteration |
| `mu` | `None` | γ₀ centre (SIM only — from `gamma0.find_params`) |
| `radius` | `None` | γ₀ radius (SIM only) |

### Solver comparison

| Solver | γ₀ required | Uses `rmatvec` | Notes |
|--------|-------------|----------------|-------|
| `SIM` | ✅ | ✗ | Simple iteration; fastest per-iter when γ₀ ≪ spectral radius |
| `BiCGStab` | ✗ | ✗ | Generally fastest convergence; no extra parameters |
| `TwoStep` | ✗ | ✅ | Steepest descent; good for non-self-adjoint operators |

---

## Mathematical background

### Discretisation

The scattering domain Q ⊂ ℝ³ is covered by a uniform Cartesian grid with
N = Nx × Ny × Nz cells of volume dV = Lx Ly Lz / N.

The discrete VIE in collocation form is

```
u_i + Σ_j B_ij η_j u_j = f_i,    i = 1 … N
```

where **B**_ij = dV · G(|r_i − r_j|, k₀) for i ≠ j and
**B**_ii = ∫₀^{r₀} exp(ik₀r) r dr  (excluded-sphere self-interaction, r₀ = (3dV/4π)^{1/3}).

### FFT acceleration

The matrix **B** is Toeplitz on the periodic lattice.  Embedding in the 2× larger
lattice Π₂ makes the operator circulant; matvec becomes:

```
B η u = IFFT[ FFT(kernel) ⊙ FFT(zero-pad(η u)) ][ :N ]
```

Cost: O(N log N) per matvec vs. O(N²) dense.

### γ₀ parameter (SIM)

For convergence of SIM `u ← u − γ₀(Au − f)`, the optimal step satisfies
γ₀ = 1/μ where μ is the centre of the **smallest enclosing circle** of the
convex hull of sampled eigenvalues of **B**·**η**.
The `gamma0` module implements this via Andrew's monotone chain + Welzl-style enumeration.

---

## Development

```bash
# clone and install with dev extras
git clone https://github.com/qwerty29544/em3d.git
cd em3d
pip install -e ".[dev]"

# run CPU tests
pytest

# run GPU tests (requires CUDA device + CuPy)
pytest -m gpu

# run without GPU marker
pytest -m "not gpu"
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python  | ≥ 3.11  |
| NumPy   | ≥ 1.26  |
| SciPy   | ≥ 1.11  |
| CuPy    | ≥ 13 *(optional, GPU)* |

---

## License

MIT — see [LICENSE](LICENSE) file.
