# Packaged Structured Lattice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** включить эксперимент решётки неоднородных включений в устанавливаемый пакет `em3d` и добавить Kaggle notebook для запуска через GitHub installation.

**Architecture:** новый подпакет `src/em3d/experiments` содержит самодостаточный модуль `structured_lattice.py`; notebook импортирует только `em3d.experiments.structured_lattice`.

**Tech Stack:** Python, NumPy, pytest, setuptools package discovery, Jupyter notebook JSON.

## Global Constraints

- Не импортировать top-level research папку `experiments/` из пакетного кода.
- Unit-тесты используют малые сетки; full `100^3` не запускается автоматически.
- Notebook должен работать после `pip install "em3d[vis] @ git+https://github.com/qwerty29544/em3d.git"`.

---

### Task 1: Packaged API

**Files:**
- Create: `src/em3d/experiments/__init__.py`
- Create: `src/em3d/experiments/structured_lattice.py`
- Test: `tests/test_packaged_experiments.py`

**Interfaces:**
- Produces: `MaterialSpec`, `ExperimentLogger`, `InclusionSpec`, `StructuredLatticeCase`, `make_structured_lattice_case`, `build_structured_lattice_problem`, `run_structured_lattice_experiment`.

- [ ] Write failing import/default tests.
- [ ] Implement package module without imports from local `experiments/`.
- [ ] Run package tests.

### Task 2: Kaggle Notebook and Docs

**Files:**
- Create: `notebooks/structured-lattice-kaggle.ipynb`
- Modify: `README.md`
- Modify: `wiki/code/structured-lattice-experiment.md`

**Interfaces:**
- Notebook installs package from GitHub and runs package functions.

- [ ] Add notebook structure test.
- [ ] Add notebook and docs.
- [ ] Run verification.

## Self-Review Checklist

- Package discovery works because `src/em3d/experiments/__init__.py` is under `tool.setuptools.packages.find`.
- Notebook does not use `sys.path` or local `experiments`.
- Tests cover import, small runner artifacts, and notebook source.
