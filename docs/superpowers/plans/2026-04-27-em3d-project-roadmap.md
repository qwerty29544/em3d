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
2. **Модель оператора упрощена.** В wiki зафиксировано, что полноценная электродинамическая ОСИУ использует тензор Грина `Gamma`, а текущий оператор использует скалярное ядро `G(R) * delta_ij`.
3. **Документация рассинхронизирована.** README, docs/specs и wiki расходятся по статусу `farfield`, `mie`, и смыслу `u`.
4. **GPU-контракт неоднородный.** Ядро поддерживает CuPy, но часть post-processing helpers неаккуратно вызывает `np.asarray`.
5. **Тесты проверяют в основном внутреннюю согласованность.** Нужны внешние/математические gates: Mie, boundary conditions, known limits.

### Целевая версия следующего этапа

Следующий устойчивый milestone: **`em3d` как воспроизводимый CPU-first исследовательский пакет для скалярно-ядровой электродинамической VIE с проверенной ЭПР и Mie-ориентированным validation harness.**

Полный dyadic Green tensor — следующий milestone, не смешивать с repair `mie`.

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
- [ ] README говорит, что `u` в `(I + B eta) u = f` — полное поле.
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

## Milestone 3: Full Electrodynamic Kernel Decision

**Goal:** decide whether to keep scalar kernel as MVP limitation or implement full dyadic Green tensor.

Brainstorm options:

1. **Keep scalar kernel.**
   - Pros: current implementation remains simple and tested.
   - Cons: limited physical fidelity for full EM scattering, especially anisotropic cases.

2. **Implement full dyadic tensor kernel.**
   - Pros: matches wiki/literature formulation of ОСИУ and improves physical correctness.
   - Cons: larger refactor; self-term and singular integral treatment become more delicate.

Recommendation:

- Do not start dyadic kernel until Mie baseline is fixed.
- After baseline, create a dedicated spec:
  - `docs/superpowers/specs/YYYY-MM-DD-em3d-dyadic-green-design.md`
  - `docs/superpowers/plans/YYYY-MM-DD-em3d-dyadic-green.md`

Minimum design questions:

- Exact discrete self-term for dyadic `Gamma`.
- Shape and storage of 3x3 kernel blocks on doubled grid.
- Compatibility with anisotropic `eta`.
- Dense reference implementation for tiny grids.
- Adjoint correctness for `rmatvec`.

Acceptance for dyadic milestone:

- FFT matvec equals dense dyadic reference on `N=3^3` or `N=4^3`.
- Mie RCS agreement improves or is at least physically interpretable.

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
  - scalar Green kernel;
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
4. Decide scalar vs dyadic kernel milestone.
5. Improve solver robustness.
6. Add examples and benchmarks.

Do not start full dyadic kernel or solver refactors before the Mie correctness baseline is stable. Otherwise validation target and numerical operator change at the same time, making failures hard to attribute.
