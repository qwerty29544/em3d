# Gamma0 Research Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить исследовательский API для оценки `gamma0` по спектру исходного dense-оператора на грубой сетке и визуализации спектра, convex hull и оптимальной окружности.

**Architecture:** `gamma0.py` отвечает за геометрию, coarse dense matrix и анализ спектра. `vis.py` отвечает только за matplotlib-отрисовку объекта `Gamma0Analysis`. README и wiki фиксируют корректное ограничение: спектр FFT-embedding не используется как спектр исходного оператора.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib optional extra, pytest.

---

### Task 1: RED-тесты для gamma0 геометрии и анализа

**Files:**
- Modify: `tests/test_gamma0.py`

- [ ] **Step 1: Write failing tests**

Добавить тесты:

```python
def test_circle_two_points_uses_gamma0_visible_angle_formula():
    centre, radius = compute_circle_two_points(1.0 + 2.0j, 3.0 + 4.0j)
    assert abs(centre - (-0.5 + 3.0j)) < 1e-12
    assert abs(radius - np.sqrt(10.0)) < 1e-12


def test_analyze_spectrum_returns_hull_and_rho():
    samples = np.array([2.0 + 0.0j, 3.0 + 1.0j, 4.0 + 0.0j, 3.0 + 0.25j])
    analysis = analyze_spectrum(samples)
    assert analysis.spectrum.shape == samples.shape
    assert len(analysis.hull) == 3
    assert abs(analysis.rho - analysis.radius / abs(analysis.mu)) < 1e-12
    assert circle_contains_points(analysis.mu, analysis.radius, samples)
    assert not circle_contains_origin(analysis.mu, analysis.radius)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_gamma0.py::test_circle_two_points_uses_gamma0_visible_angle_formula tests/test_gamma0.py::test_analyze_spectrum_returns_hull_and_rho -q -p no:cacheprovider
```

Expected: FAIL because `compute_circle_two_points` still uses midpoint and `analyze_spectrum` does not exist.

### Task 2: Implement gamma0 analysis core

**Files:**
- Modify: `src/em3d/gamma0.py`

- [ ] **Step 1: Add `Gamma0Analysis`, `mu_2points`, `radius_2points`, and `analyze_spectrum`**

Implementation requirements:
- `compute_circle_two_points` delegates to notebook-compatible formulas.
- `analyze_spectrum` returns `Gamma0Analysis`.
- `find_params` delegates to `analyze_spectrum` and returns only `{"mu": mu, "radius": radius}`.

- [ ] **Step 2: Run Task 1 tests and verify GREEN**

Run same command from Task 1. Expected: PASS.

### Task 3: RED-тесты для coarse operator matrix по Problem

**Files:**
- Modify: `tests/test_gamma0.py`

- [ ] **Step 1: Write failing tests**

Добавить toy problem helper and tests:

```python
def _gamma0_problem(backend, N=(3, 3, 3)):
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend)
    eta = cylinder_refraction(grid, eps_real=1.5, eps_imag=0.1, radius=0.35, axis="z")
    wave = flat_wave_vec(grid, k=0.75, orient=(0, 0, 1), amplitude=(1, 0, 0))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.75, volume=grid.dv * int(np.prod(N)))


def test_coarse_operator_matrix_shape(backend_numpy_double):
    problem = _gamma0_problem(backend_numpy_double, N=(3, 3, 3))
    H = coarse_operator_matrix(problem, coarse_N=(2, 2, 2))
    assert H.shape == (24, 24)
    assert H.dtype == np.complex128


def test_estimate_from_problem_returns_solver_params(backend_numpy_double):
    problem = _gamma0_problem(backend_numpy_double, N=(3, 3, 3))
    analysis = estimate_from_problem(problem, coarse_N=(2, 2, 2))
    assert analysis.spectrum.shape == (24,)
    assert analysis.coarse_N == (2, 2, 2)
    assert analysis.matrix_shape == (24, 24)
    assert analysis.as_solver_config_kwargs() == {"mu": analysis.mu, "radius": analysis.radius}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_gamma0.py::test_coarse_operator_matrix_shape tests/test_gamma0.py::test_estimate_from_problem_returns_solver_params -q -p no:cacheprovider
```

Expected: FAIL because functions do not exist.

### Task 4: Implement coarse matrix and problem analysis

**Files:**
- Modify: `src/em3d/gamma0.py`

- [ ] **Step 1: Add coarse grid helpers**

Implementation requirements:
- `_normalize_grid_shape(coarse_N)` accepts int or 3-tuple.
- `_nearest_indices(source_axis, target_axis)` maps target coarse centers to nearest source centers.
- `_resample_eps_tensor(problem, coarse_grid)` returns `(3,3)+coarse_N` numpy tensor.

- [ ] **Step 2: Add `coarse_operator_matrix` and `estimate_from_problem`**

Implementation requirements:
- `coarse_operator_matrix` builds `B_dense` through `dense.B_operator_matrix`.
- Build block-diagonal `Eta_dense` in the same cell-major/component-major order used by tests.
- Return `H = I - B_dense @ Eta_dense`, matching `Operator.matvec`.
- `estimate_from_problem` computes `np.linalg.eigvals(H)` and calls `analyze_spectrum`.

- [ ] **Step 3: Run Task 3 tests and verify GREEN**

Run same command from Task 3. Expected: PASS.

### Task 5: RED-тест визуализации gamma0

**Files:**
- Modify: `tests/test_vis.py`

- [ ] **Step 1: Write failing test**

Добавить import `plot_gamma0_spectrum` and test:

```python
def test_plot_gamma0_spectrum_draws_spectrum_hull_and_circle():
    analysis = em3d.gamma0.analyze_spectrum(
        np.array([2.0 + 0.0j, 3.0 + 1.0j, 4.0 + 0.0j, 3.0 + 0.25j])
    )
    fig, ax = plot_gamma0_spectrum(analysis)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) >= 2
    assert len(ax.lines) >= 1
    assert len(ax.patches) == 1
    assert ax.get_aspect() in ("equal", 1.0)
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_vis.py::test_plot_gamma0_spectrum_draws_spectrum_hull_and_circle -q -p no:cacheprovider
```

Expected: FAIL because function is not exported.

### Task 6: Implement visualization

**Files:**
- Modify: `src/em3d/vis.py`

- [ ] **Step 1: Add `plot_gamma0_spectrum`**

Implementation requirements:
- Add function to `__all__`.
- Plot spectrum scatter, hull line closed back to first vertex, center scatter, origin marker, circle patch.
- Set `ax.set_aspect("equal", adjustable="box")`.
- Save to `filename` if provided.

- [ ] **Step 2: Run Task 5 test and verify GREEN**

Run same command from Task 5. Expected: PASS.

### Task 7: README and wiki updates

**Files:**
- Modify: `README.md`
- Modify external wiki pages: `concepts/optimal-iteration-parameter-gamma0.md`, `concepts/generalized-simple-iteration.md`, `code/em3d.md`, `index.md`, `log.md`

- [ ] **Step 1: Add README example**

Add a short section showing:

```python
analysis = em3d.gamma0.estimate_from_problem(problem, coarse_N=(4, 4, 4))
cfg = em3d.SolverConfig(max_iter=500, rtol=1e-6, **analysis.as_solver_config_kwargs())
fig, ax = em3d.vis.plot_gamma0_spectrum(analysis)
```

- [ ] **Step 2: Update wiki**

Mention dense coarse matrix and explicitly warn that FFT-embedding spectrum is not used as the operator spectrum.

### Task 8: Full verification

**Files:** all touched files.

- [ ] **Step 1: Run targeted tests**

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_gamma0.py tests/test_vis.py -q -p no:cacheprovider
```

- [ ] **Step 2: Run full non-file-saving suite**

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest -q -p no:cacheprovider -k "not saves_file"
```

- [ ] **Step 3: Run diff check**

```powershell
git diff --check
```
