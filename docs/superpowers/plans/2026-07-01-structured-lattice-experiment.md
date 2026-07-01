# Structured Lattice Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать эксперимент решётки неоднородных включений для сетки `100x100x100` с оценкой `gamma0` на `9x9x9`, тремя итерационными методами, визуализацией поля/ЭПР/невязок и логами метрик.

**Architecture:** Основная реализация живёт в `experiments/structured_lattice.py`; `experiments/chapter06_em.py` остаётся фасадом и реэкспортирует новый API. Unit-тесты используют малые сетки и monkeypatch для дорогих графиков/солверов.

**Tech Stack:** Python, NumPy, pytest, локальный `em3d`, `ExperimentLogger`, `em3d.vis`, `em3d.farfield`.

## Global Constraints

- Все ответы и wiki-документация пишутся на русском языке.
- Рабочий default case: `N=(100,100,100)`, `coarse_N=(9,9,9)`.
- Unit-тесты не должны запускать `100^3` расчёт.
- Новые файлы создаются ASCII-compatible, код — с краткими комментариями только там, где это оправдано.

---

### Task 1: Case Builder and Problem Construction

**Files:**
- Create: `experiments/structured_lattice.py`
- Modify: `experiments/chapter06_em.py`
- Test: `tests/test_chapter06_em.py`

**Interfaces:**
- Produces: `InclusionSpec`, `StructuredLatticeCase`, `make_structured_lattice_case`, `build_structured_lattice_problem`.
- Consumes: `MaterialSpec`, `material_eps`, `ExperimentLogger`, `em3d.Grid`, `em3d.Problem`, `em3d.Operator`.

- [ ] **Step 1: Write failing tests** for default parameters and small-grid `eps_tensor` occupancy.
- [ ] **Step 2: Run RED** with the new tests and confirm missing symbols fail.
- [ ] **Step 3: Implement minimal dataclasses and problem builder**.
- [ ] **Step 4: Run GREEN** for builder/problem tests.

### Task 2: Full Runner and Artifacts

**Files:**
- Modify: `experiments/structured_lattice.py`
- Modify: `experiments/chapter06_em.py`
- Test: `tests/test_chapter06_em.py`

**Interfaces:**
- Produces: `run_structured_lattice_experiment`.
- Consumes: `build_structured_lattice_problem`, `estimate_gamma0`, `run_solver`, `save_runs_csv`, `save_json`, `plot_three_field_slices`, `plot_residual_histories`, `ExperimentLogger`.

- [ ] **Step 1: Write failing smoke test** for a small case with one/few iterations and monkeypatched plot functions.
- [ ] **Step 2: Run RED** and confirm missing runner fails.
- [ ] **Step 3: Implement runner** with logs, CSV/JSON artifacts, residual histories, field plots and RCS plots.
- [ ] **Step 4: Run GREEN** for runner smoke test.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify/Create: `wiki/code/structured-lattice-experiment.md`
- Modify: `wiki/index.md`, `wiki/log.md`

**Interfaces:**
- Documents default invocation and generated artifacts.

- [ ] **Step 1: Update README/wiki** with example code.
- [ ] **Step 2: Run verification**:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
py -m pytest -q -p no:cacheprovider -k "not saves_file"
git diff --check
```

## Self-Review Checklist

- Spec coverage: все требования пользователя покрыты задачами 1-3.
- Placeholder scan: нет `TBD`, `TODO`, `implement later`.
- Type consistency: имена публичных функций совпадают между plan и spec.
