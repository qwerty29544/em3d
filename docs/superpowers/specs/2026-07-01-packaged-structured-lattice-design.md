# Packaged Structured Lattice Design

## Цель

Сделать эксперимент решётки неоднородных включений доступным после установки пакета из GitHub, без зависимости от локальной research-папки `experiments/`.

## Решение

Добавляется подпакет `em3d.experiments`:

- `src/em3d/experiments/__init__.py`
- `src/em3d/experiments/structured_lattice.py`

Публичный импорт:

```python
from em3d.experiments.structured_lattice import (
    make_structured_lattice_case,
    run_structured_lattice_experiment,
)
```

Модуль содержит самодостаточные `MaterialSpec`, `ExperimentLogger`, `InclusionSpec`, `StructuredLatticeCase`, builder, problem builder и runner. Он использует только пакетный `em3d`, NumPy и стандартную библиотеку.

## Kaggle Notebook

Добавляется `notebooks/structured-lattice-kaggle.ipynb`. Он устанавливает пакет:

```bash
pip install "em3d[vis] @ git+https://github.com/qwerty29544/em3d.git"
```

и запускает эксперимент через `em3d.experiments.structured_lattice`. Результаты упаковываются в zip в `/kaggle/working`.

## Ограничения

- Full-run `N=(100,100,100)` не запускается в unit-тестах.
- Визуализации требуют extra `em3d[vis]`.
- Research harness `experiments/` остаётся совместимым, но пакетный API живёт в `src/em3d/experiments`.

## Самопроверка спеки

- Placeholder-ов нет.
- Требование установки с GitHub покрыто через package namespace.
- Notebook не использует локальные project-relative imports.
