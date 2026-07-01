# Field Visualization Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разделить визуализации поля на отдельные scalar/vector функции в 2D и 3D без поломки старого API.

**Architecture:** `src/em3d/vis.py` получает четыре новые публичные функции и небольшие private helpers для выбора среза, компоненты и скаляра. Старые `plot_field_slice`/`plot_field_volume` остаются совместимыми.

**Tech Stack:** Python, NumPy, Matplotlib, pytest.

---

### Task 1: RED tests

**Files:**
- Modify: `tests/test_vis.py`

- [ ] Add imports for the four new functions.
- [ ] Add tests proving scalar 2D draws only scalar pcolormesh, vector 2D draws only quiver, scalar 3D draws scatter, vector 3D draws quiver.
- [ ] Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_vis.py::test_plot_field_scalar_slice_returns_heatmap_without_quiver tests/test_vis.py::test_plot_field_vector_slice_returns_quiver_without_scalar_background tests/test_vis.py::test_plot_field_scalar_volume_returns_scatter3d tests/test_vis.py::test_plot_field_vector_volume_returns_quiver3d -q -p no:cacheprovider
```

Expected: import failure because functions do not exist.

### Task 2: Implement functions

**Files:**
- Modify: `src/em3d/vis.py`

- [ ] Add names to `__all__` and Public API docstring.
- [ ] Add helpers `_validate_component`, `_field_scalar`, `_select_slice`.
- [ ] Implement `plot_field_scalar_slice`.
- [ ] Implement `plot_field_vector_slice`.
- [ ] Implement `plot_field_scalar_volume`.
- [ ] Implement `plot_field_vector_volume`.
- [ ] Keep `plot_field_slice` combined and make `plot_field_volume` call `plot_field_vector_volume`.

### Task 3: Docs and verification

**Files:**
- Modify: `README.md`
- Modify external wiki `code/em3d.md`, `index.md`, `log.md`

- [ ] Update README examples to show scalar/vector split.
- [ ] Update wiki code page and log.
- [ ] Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_vis.py -q -p no:cacheprovider -k "not saves_file"
py -m pytest -q -p no:cacheprovider -k "not saves_file"
git diff --check
```

