# em3d Correctness Repair Plan

> **Назначение:** план починки после code review серии `em3d.mie` и сверки с wiki. Код реализации начинается только после прохождения этого плана сверху вниз. Шаги используют checkbox (`- [ ]`) для трекинга.

## Brainstorm: решения и приоритеты

### Цель

Довести проект до состояния, где `em3d.mie` можно использовать как верификационный эталон, а публичные API `farfield`, `vis`, `mie`, `Grid` и `Problem` имеют согласованные контракты.

### Главная развилка

Есть два способа чинить `mie`:

1. **Минимальная точечная починка**: исправить `_angle_functions`, special case `r=0`, тест absorption и GPU conversion.
2. **Контролируемая перепроверка Mie-модуля**: сначала зафиксировать независимые математические оракулы, затем чинить реализацию.

Выбор: **вариант 2**. Причина: review выявил, что ошибка `r=0` пришла из исходной wiki/spec-математики, поэтому простая правка кода без перепроверки ближнеполевых VSH-формул может оставить неверный эталон.

### Приоритеты

1. **P0: не сломать уже рабочее ядро.** Все изменения должны проходить существующие CPU-тесты.
2. **P1: Mie far-field correctness.** Сначала исправить угловые функции, RCS и cross sections, потому что это минимальный эталон для `farfield`.
3. **P1: near-field correctness gate.** Не считать `mie_field_at` стабильным, пока не пройдены независимые тесты на центр, пределы и граничные условия.
4. **P2: API/data movement.** Убрать неявные `np.asarray` на backend-массивах там, где поддерживается CuPy.
5. **P2: документация и wiki.** Исправить source-of-truth: README, docs/specs и wiki должны говорить одно и то же.

### Математические оракулы

- `_angle_functions`:
  - `pi_1(mu) = 1`
  - `tau_1(mu) = mu`
  - `pi_2(mu) = 3mu`
  - `tau_2(mu) = 6mu^2 - 3`
  - `pi_3(mu) = (15mu^2 - 3)/2`
  - `tau_3(mu) = (45mu^3 - 33mu)/2`

- Cross sections:
  - Rayleigh scattering for small `x`.
  - Lossless `abs` close to zero by absolute value.
  - Passive lossy `Im(eps_r) > 0` has non-negative absorption.

- Near field:
  - `eps_r == 1` gives exactly incident field outside and inside: no scatterer.
  - `r -> 0` is continuous and depends on sphere parameters; no hard-coded incident field.
  - Rayleigh/static limit at center tends to `3 / (eps_r + 2) * E0` for real positive `eps_r` and `k0*a << 1`.
  - Tangential `E` is continuous across `r=a` away from coordinate singularities.
  - Normal `D` is continuous across `r=a` for lossless isotropic sphere.

### Риски

- Ближнеполевые формулы в текущей spec могут иметь дополнительные ошибки, не только special case `r=0`. План поэтому требует тестов на граничные условия до объявления `mie_field_at` эталоном.
- Integration test `em3d vs Mie` может показать систематическую погрешность не из-за Mie, а из-за упрощенного скалярного ядра `G(R) * delta_ij` вместо полного тензора Грина. Это нужно фиксировать как ограничение, а не подгонять Mie.
- Полный pytest в текущей среде может падать на `tmp_path` из-за прав к temp/cache. Для verification использовать `py -m pytest ... -q`; при проблемах с temp запускать под пользовательским окружением вне sandbox.

---

## File Map

- **Modify:** `src/em3d/mie.py`
- **Modify:** `tests/test_mie.py`
- **Modify:** `src/em3d/farfield/_core.py`
- **Modify:** `src/em3d/vis.py`
- **Modify:** `src/em3d/problem.py`
- **Modify:** `src/em3d/wave.py`
- **Modify:** `README.md`
- **Modify:** `docs/superpowers/specs/2026-04-27-em3d-mie-design.md`
- **External wiki update:** `C:\Users\user\Documents\ClaudeProjects\ClaudeProject\wiki\concepts\mie-scattering-theory.md`
- **External wiki update:** `C:\Users\user\Documents\ClaudeProjects\ClaudeProject\wiki\code\em3d.md`
- **External wiki update:** `C:\Users\user\Documents\ClaudeProjects\ClaudeProject\wiki\concepts\scattering-cross-section.md`
- **External wiki update:** `C:\Users\user\Documents\ClaudeProjects\ClaudeProject\wiki\concepts\far-field-formula.md`

External wiki changes are outside the current worktree. Do them only when explicitly allowed for that path/session.

---

## Task 1: Add failing Mie mathematical tests

**Files:**
- Modify: `tests/test_mie.py`

- [ ] Add `test_angle_functions_closed_forms`.

Expected checks:

```python
mu = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
pi_arr, tau_arr = em3d.mie._angle_functions(3, mu)
np.testing.assert_allclose(pi_arr[0], 1.0)
np.testing.assert_allclose(tau_arr[0], mu)
np.testing.assert_allclose(pi_arr[1], 3.0 * mu)
np.testing.assert_allclose(tau_arr[1], 6.0 * mu**2 - 3.0)
np.testing.assert_allclose(pi_arr[2], 0.5 * (15.0 * mu**2 - 3.0))
np.testing.assert_allclose(tau_arr[2], 0.5 * (45.0 * mu**3 - 21.0 * mu))
```

- [ ] Replace `test_mie_cross_sections_lossless_zero_absorption` assertion with absolute error:

```python
assert abs(cross["abs"]) < 1e-10 * cross["scat"]
```

- [ ] Add `test_mie_cross_sections_passive_loss_nonnegative_absorption` for `eps_r=2.0+0.1j`.

- [ ] Replace `test_mie_field_center_equals_incident` with:
  - center continuity test against nearby interior points;
  - Rayleigh/static limit test at `k0*a = 0.01`, `eps_r=2.0`, expected `3/(eps_r+2)`.

- [ ] Add `test_mie_field_no_scatterer_matches_incident_wave` with `eps_r=1.0` for several points inside/outside.

- [ ] Add orientation tests:
  - transverse amplitude accepted;
  - non-transverse amplitude raises `ValueError` or uses documented projected norm. Pick one behavior before implementation.

Preferred behavior: **raise** for non-transverse amplitude unless the longitudinal component is below tolerance. This matches physical transverse plane wave semantics and avoids silent rescaling.

Verification command:

```powershell
py -m pytest tests/test_mie.py -q
```

Expected result before Task 2: new tests fail.

---

## Task 2: Fix `_angle_functions`

**Files:**
- Modify: `src/em3d/mie.py`

Implement recurrence with correct previous-state handling:

```python
pi_0 = zeros
pi_1 = ones

for n in 1..n_max:
    if n == 1:
        pi_n = pi_1
        pi_n_minus_1 = pi_0
    else:
        pi_n = ((2*n - 1)/(n - 1))*mu*pi_prev1 - (n/(n - 1))*pi_prev2
        pi_n_minus_1 = pi_prev1
    tau_n = n*mu*pi_n - (n + 1)*pi_n_minus_1
```

- [ ] Keep array shape `(n_max, M)`.
- [ ] Preserve dtype `float64`.
- [ ] Run `py -m pytest tests/test_mie.py -q`.

---

## Task 3: Fix Mie near-field center and validate VSH formulas

**Files:**
- Modify: `src/em3d/mie.py`
- Modify: `tests/test_mie.py`

Implementation rule:

- Do not return incident field at `r == 0`.
- Either:
  - derive and implement the analytic `r -> 0` limit from the corrected interior series; or
  - compute zero points through a stable dedicated limit helper based on the `n=1` internal term.

Do not use a blind epsilon displacement as the public behavior. It can be used in tests to verify continuity, not as the implementation.

- [ ] Audit the interior VSH formulas before coding the center limit. In particular verify whether the radial component of `N_e1n` should include a `sin(theta)` factor under the current definition of `pi_n`.
- [ ] Add a private helper if needed, e.g. `_field_center_canonical(k0, m, coeffs)`.
- [ ] Treat points exactly at `r=0` through that helper.
- [ ] Keep vectorized behavior for mixed zero/non-zero points.
- [ ] Run `py -m pytest tests/test_mie.py -q`.

Acceptance:

- Center field is continuous against points with radius `1e-8*a`, `1e-7*a`, `1e-6*a`.
- Rayleigh/static limit passes within a conservative tolerance, e.g. `rtol=2e-3` for `k0*a=0.01`.
- `eps_r=1.0` no-scatterer test passes for center, interior and exterior points.
- Add `test_mie_field_satisfies_sphere_boundary_conditions` as `xfail(strict=True)` until exterior near-field formulas are repaired. Run it with `--runxfail` to confirm it currently fails on tangential `E` / normal `D` continuity.

---

## Task 4: Fix amplitude/orientation contract

**Files:**
- Modify: `src/em3d/mie.py`
- Modify: `tests/test_mie.py`
- Modify: `docs/superpowers/specs/2026-04-27-em3d-mie-design.md`

Decision: `amplitude` must be transverse to `orient`.

- [ ] Validate `orient` shape `(3,)` and non-zero norm.
- [ ] Validate `amplitude` shape `(3,)` and non-zero norm.
- [ ] Reject `abs(dot(amplitude, z_can)) > tol * norm(amplitude)`.
- [ ] Set `E0 = norm(amplitude)` after validation.
- [ ] Build `x_can = amplitude / E0`.
- [ ] Keep current "parallel" error, but extend it to any materially non-transverse amplitude.

Tests:

- `amplitude=(1,0,0), orient=(0,0,1)` accepted.
- `amplitude=(1,0,1), orient=(0,0,1)` raises.
- `orient=(0,0,0)` raises.
- `amplitude=(0,0,0)` raises.

---

## Task 5: Fix backend data movement in public helpers

**Files:**
- Modify: `src/em3d/mie.py`
- Modify: `src/em3d/farfield/_core.py`
- Modify: `src/em3d/vis.py` if GPU plotting is intended to work.

Current risk:

- `np.asarray(cupy_array)` can fail because CuPy blocks implicit CPU transfer.
- Project convention already has `Backend.to_host`.

Implementation:

- [ ] In `mie_field`, convert grid coordinates through `grid.backend.to_host`.
- [ ] In `farfield/_core.py`, validate `u` shape without `np.asarray(u)` for CuPy, or use `be.to_host` only after shape validation.
- [ ] In `vis.py`, either document CPU-only plotting or convert `u` and `grid.coords()` through `grid.backend.to_host`.

Tests:

- Keep existing GPU test skipped when CuPy unavailable.
- Add CPU regression tests that do not require CuPy.
- If CuPy available, add `mie_field` GPU smoke test marked `@pytest.mark.gpu`.

---

## Task 6: Add normalized `em3d` vs Mie integration gate

**Files:**
- Modify: `src/em3d/mie.py`
- Modify: `src/em3d/vis.py`
- Modify: `tests/test_mie.py`
- Modify: `tests/test_vis.py`
- Modify: `docs/superpowers/specs/2026-04-27-em3d-mie-design.md`

Goal: implement the integration test promised by the Mie spec without turning the current absolute scale mismatch into a false negative, and expose the same comparison data for visual inspection. Compare the angular shape of the RCS curve after normalizing each curve by its own maximum; keep the absolute scale as a loose diagnostic.

Use the original spec cases:

- `(eps_r=2.0, k0*a=1.0)`
- `(eps_r=1.5, k0*a=0.5)`
- `a = 0.3`
- `N = 32`
- `plane = "xy"`
- Solver: `BiCGStab(SolverConfig(max_iter=500, rtol=1e-8))`

Acceptance:

- `sigma_num_peak = max(sigma_num)` and `sigma_mie_peak = max(sigma_mie)` are positive.
- `sigma_num_n = sigma_num / sigma_num_peak`.
- `sigma_mie_n = sigma_mie / sigma_mie_peak`.
- `shape_err = norm(sigma_num_n - sigma_mie_n) / norm(sigma_mie_n) <= 0.02`.
- `scale_ratio = sigma_num_peak / sigma_mie_peak` is recorded and must remain within loose diagnostic bounds `0.5 <= scale_ratio <= 2.0`.
- Do **not** normalize each far-field vector `F(phi)` independently; normalize only the final RCS curve. Per-direction normalization would destroy the angular diagram.
- Keep strict absolute RCS agreement, e.g. `<= 0.10`, as a future gate after effective-radius and dyadic-kernel issues are resolved.

Implementation:

- Add `em3d.mie.compare_rcs_plane(u, problem, *, a, eps_r, n_phi=180, plane="xy", method="direct", batch_size=64, normalize="max")`.
- Return a dictionary with keys: `phi`, `sigma_num`, `sigma_mie`, `sigma_num_norm`, `sigma_mie_norm`, `shape_err`, `scale_ratio`, `abs_rel_err`.
- Update `test_mie_verification_rcs_normalized_shape` to call this helper rather than duplicating metric code.
- Add `em3d.vis.plot_rcs_comparison(phi, sigma_num, sigma_mie, ...)` for Cartesian overlay.
- Add `em3d.vis.plot_rcs_comparison_polar(phi, sigma_num, sigma_mie, ...)` for polar overlay.
- The plot helpers must draw two lines and must not normalize data implicitly; callers pass raw or normalized curves intentionally.

Command:

```powershell
py -m pytest tests/test_mie.py -q
```

---

## Task 7: Repair documentation and wiki source-of-truth

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-27-em3d-mie-design.md`
- External wiki files listed in File Map

- [ ] README: resolve whether `u` is total field or scattered field. Current code solves `(I + B eta) u = f` as total field. Document that consistently.
- [ ] Mie spec: remove claim `E_int(0)=E0*x_hat`; replace with parameter-dependent center limit.
- [ ] Mie spec: document transverse amplitude requirement.
- [ ] Wiki `concepts/mie-scattering-theory.md`: same center correction.
- [ ] Wiki `code/em3d.md`: update from MVP-only state to include `farfield`, `vis`, `mie`, known limitations.
- [ ] Wiki `scattering-cross-section` / `far-field-formula`: replace "planned farfield" statements with implemented status.
- [ ] Wiki `log.md`: append a `code` or `write` entry for documentation correction.

Do not edit external wiki path without explicit approval if the active sandbox does not allow it.

---

## Task 8: Full verification

Run targeted tests:

```powershell
py -m pytest tests/test_mie.py -q
py -m pytest tests/test_farfield.py tests/test_refraction.py tests/test_operator_vs_dense.py tests/test_solvers.py -q
py -m pytest tests/test_vis.py -q -k "not saves_file"
```

Run full tests if temp/cache permissions are sane:

```powershell
py -m pytest -q
```

If full tests fail only because `tmp_path` cannot access the default temp directory, record that as environment limitation and run save-file tests manually in a permitted temp directory.

---

## Completion Criteria

- `tests/test_mie.py` contains independent mathematical checks, not only self-consistency.
- `mie_rcs_plane` uses correct angular functions.
- `mie_field_at` no longer hard-codes center field as incident field.
- Non-transverse plane-wave amplitudes fail loudly.
- CPU tests pass.
- Documentation and wiki no longer contradict code on `result.u`, `farfield` status, or Mie center behavior.
