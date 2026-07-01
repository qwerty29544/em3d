# em3d Project Implementation Roadmap

> **Назначение:** рабочий roadmap после текущей серии `mie`-коммитов. Этот документ не заменяет точечный repair plan; он задаёт порядок доведения проекта до проверяемого исследовательского инструмента.

## Brainstorm: текущее состояние проекта

### Что уже сделано хорошо

- Пакетная структура `src/em3d` отделена от wiki и docs.
- Есть единый backend layer (`numpy`/`cupy`) и параметризуемая точность.
- Основной FFT-оператор сверкивается с dense reference на малой сетке.
- Есть три solver-класса: `SIM`, `BiCGStab`, `TwoStep`.
- `farfield` реализует прямой и FFT backend для ЭПР.
- `vis` закрывает базовые сценарии RCS и field visualization.
- Начат аналитический эталон `mie`.

### Что ограничивает дальнейшую реализацию

1. **Mie пока не эталон.** Есть математические ошибки и слабые тесты.
2. **Модель оператора восстановлена до dyadic Green kernel.** После проверки raw notebook текущий оператор использует 3×3 ядро `G(R)[C1 alpha alpha^T + C2 I]` и self-term `-I/3`; прежняя формулировка про скалярное `G(R) * delta_ij` была ошибкой переноса.
3. **Документация рассинхронизирована.** README, docs/specs и wiki расходятся по статусу `farfield`, `mie`, и смыслу `u`.
4. **GPU-контракт неоднородный.** Ядро поддерживает CuPy, но часть post-processing helpers неаккуратно вызывает `np.asarray`.
5. **Тесты проверяют в основном внутреннюю согласованность.** Нужны внешние/математические gates: Mie, boundary conditions, known limits.

### Целевая версия следующего этапа

Следующий устойчивый milestone: **`em3d` как воспроизводимый CPU-first исследовательский пакет для векторной электродинамической VIE с dyadic Green kernel, проверенной ЭПР и Mie-ориентированным validation harness.**

---

## Milestone 1: Correctness Baseline

**Goal:** все текущие публичные API согласованы и покрыты тестами на известные математические инварианты.

Dependencies:

- Выполнить `2026-04-27-em3d-correctness-repair.md`.

Deliverables:

- [ ] `mie` far-field проходит closed-form angular tests.
- [ ] `mie` near-field проходит center/no-scatterer/boundary sanity tests.
- [ ] `farfield` использует safe backend conversion.
- [ ] `vis` корректно обрабатывает zero RCS in dB mode.
- [ ] README говорит, что `u` в `(I - B eta) u = f` — полное поле.
- [ ] Wiki обновлена или явно вынесена из текущего git state.

Verification:

```powershell
py -m pytest tests/test_mie.py tests/test_farfield.py tests/test_vis.py -q
```

Exit criteria:

- Нет известных P1/P2 correctness issues в текущем API.
- Все ограничения записаны явно.

---

## Milestone 2: Validation Harness

**Goal:** добавить воспроизводимые validation scenarios, которые можно запускать перед физическими изменениями.

File Map:

- **Add:** `tests/test_validation_mie.py` or extend `tests/test_mie.py`
- **Add:** `docs/superpowers/plans/2026-04-27-em3d-validation-harness.md` if work becomes large
- **Modify:** `README.md`

Tasks:

- [ ] Add small `em3d vs Mie RCS` integration scenario.
- [ ] Add near-field comparison scenario only after `mie_field_at` is trustworthy.
- [ ] Add `eps_r=1` transparent structure tests:
  - solver returns incident field;
  - farfield RCS is zero.
- [ ] Add passive lossy medium sanity:
  - no NaN/Inf;
  - positive absorption in Mie.
- [ ] Add benchmark note for expected runtime on `N=16`, `N=32`.

Acceptance:

- Validation tests run in under a few seconds for CPU default.
- Expensive tests are either skipped by default or explicitly marked.

---

## Milestone 3: Dyadic Kernel Baseline

**Goal:** keep the restored dyadic Green kernel covered by regression tests and use it as the baseline for subsequent validation experiments.

Status after repair:

- `kernel.b_coeff` returns a full 3×3 dyadic block.
- FFT kernel stores all 9 block components on the doubled grid.
- `Operator.matvec` uses `I - B eta`, matching the raw notebook.
- `Operator.rmatvec` is checked against the dense adjoint.
- `gamma0.coarse_operator_matrix` analyzes `H = I - B_dense @ Eta_dense`.

Remaining validation questions:

- Effective-radius and voxelization effects in Mie absolute RCS.
- Boundary-condition quality of the analytical near-field reference.
- Stronger anisotropic cases with off-diagonal material tensors.

---

## Milestone 4: Solver Robustness

**Goal:** make solver behavior reliable beyond toy weak-contrast tests.

Tasks:

- [ ] Add residual normalization helper shared across solvers.
- [ ] Add breakdown diagnostics to `BiCGStab`:
  - zero denominator in `alpha`;
  - zero/near-zero `omega`;
  - finite residual checks.
- [ ] Revisit `TwoStep`: current implementation is steepest descent on normal equations, not necessarily the full two-parameter method from wiki/literature.
- [ ] Add tests on moderately stronger contrast.
- [ ] Add deterministic spectrum sample helper for `gamma0.find_params`.

Acceptance:

- Solvers fail explicitly with `converged=False` and useful residual history rather than division warnings/NaN.
- Documentation distinguishes implemented `TwoStep` from published full TwoSGD if they differ.

---

## Milestone 5: Documentation and Wiki Governance

**Goal:** docs and wiki become reproducible project assets, not parallel divergent sources.

Tasks:

- [ ] Decide git policy for wiki:
  - track wiki as submodule-like gitlink and commit wiki repo changes; or
  - treat wiki as external local knowledge base and do not rely on it for repo reproduction.
- [ ] If wiki is part of project state, commit its ingest pages and update parent gitlink.
- [ ] Update `code/em3d.md` after each completed feature milestone.
- [ ] Add "Known limitations" to README:
  - voxelized sphere/effective-radius limits for absolute Mie RCS;
  - Mie large `x > 10`;
  - optional GPU support boundaries;
  - plotting CPU transfer behavior.
- [ ] Keep docs/specs immutable enough to explain decisions, but add correction specs for known mistakes.

Acceptance:

- A fresh checkout can answer: what is implemented, what is planned, what is intentionally approximate.

---

## Milestone 6: Performance and Usability

**Goal:** make the package usable for repeated experiments, not only tests.

Tasks:

- [ ] Add small examples under `examples/`:
  - cylinder solve + RCS;
  - sphere solve + Mie comparison;
  - anisotropic ellipsoid solve + field slice.
- [ ] Add optional benchmark script:
  - operator matvec timing;
  - solver iteration timing;
  - farfield direct vs FFT.
- [ ] Add CI-friendly markers:
  - `slow`
  - `gpu`
  - `validation`
- [ ] Add better error messages for shape/dtype mismatches in `Problem`, `Grid`, `farfield`, `mie`.

Acceptance:

- New user can run one example and one validation test without reading internals.

---

## Recommended Execution Order

1. Run `2026-04-27-em3d-correctness-repair.md`.
2. Add validation harness.
3. Update docs/wiki to match actual code state.
4. Validate dyadic-kernel Mie scale and anisotropic cases.
5. Improve solver robustness.
6. Add examples and benchmarks.

Do not mix further solver refactors with Mie validation unless the failing metric is already isolated. Otherwise validation target and numerical method change at the same time, making failures hard to attribute.
