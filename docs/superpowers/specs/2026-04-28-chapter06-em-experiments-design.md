# Спецификация: модульный notebook для экспериментов главы 6

- **Дата:** 2026-04-28
- **Статус:** draft, дизайн подтверждён автором
- **Область:** численные эксперименты для объёмной электродинамической постановки `em3d`
- **Вне области:** акустика, GPU-серия, ремонт `mie_field_at` near-field

---

## Цель

Создать воспроизводимый экспериментальный слой для главы 6 диссертации:

> Численные эксперименты для объёмной электродинамической постановки.

Эксперименты должны демонстрировать:

- векторную электродинамическую ОИУ-постановку `(I - B eta) u = f`;
- работу dyadic Green kernel и FFT-ускоренного matvec;
- влияние дискретного спектра на сходимость SIM;
- применение `gamma0`;
- сравнение SIM, BiCGStab и TwoStep;
- расчёт поля внутри неоднородности;
- расчёт диаграммы направленности и ЭПР;
- вычислительную эффективность по серии сеток.

Notebook должен быть пригоден для диссертационного narrative: текст, таблицы, графики и контролируемые выводы. Повторяемая вычислительная логика должна быть вынесена из notebook в отдельный исследовательский модуль вне публичного API пакета.

---

## Структура файлов

Добавить:

```text
experiments/
  chapter06_em.py
  outputs/
    chapter06/
      raw/
      tables/
      figures/

notebooks/
  chapter-06-em.ipynb
```

`experiments/chapter06_em.py` не входит в публичный API `em3d`. Это исследовательский harness поверх установленного локального пакета.

`notebooks/chapter-06-em.ipynb` должен быть тонким notebook: параметры запуска, вызовы helper-функций, визуализации, таблицы и текстовые комментарии.

---

## Режимы запуска

Эксперименты CPU-only.

Базовая серия:

```python
N_SERIES = [8, 16, 24, 32, 40, 48, 56, 64]
```

Режимы:

```python
RUN_MODE = "quick"  # N = [8, 16, 24]
RUN_MODE = "full"   # N = [8, 16, 24, 32, 40, 48, 56, 64]
```

`quick` нужен для проверки notebook и быстрых правок. `full` нужен для финальных таблиц главы.

Для `N=64` запрещено строить dense-матрицу полного оператора. Разрешены только FFT-operator solve, поле, ЭПР и timing.

Спектр для `gamma0` всегда оценивается на coarse-сетке, например `(4, 4, 4)` или `(6, 6, 6)`, независимо от fine-сетки эксперимента.

---

## Модель данных

В `experiments/chapter06_em.py` добавить dataclass-структуры.

```python
@dataclass(frozen=True)
class ExperimentCase:
    name: str
    N: tuple[int, int, int]
    L: tuple[float, float, float]
    k0: float
    geometry: str
    eps_real: float | np.ndarray
    eps_imag: float | np.ndarray
    center: tuple[float, float, float]
    radius: tuple[float, float, float]
    wave_orient: tuple[float, float, float]
    wave_amplitude: tuple[float, float, float]
```

```python
@dataclass(frozen=True)
class SolverRun:
    case_name: str
    solver_name: str
    N: tuple[int, int, int]
    dof: int
    converged: bool
    iterations: int
    final_residual: float
    elapsed_sec: float
    residual_history: list[float]
```

Дополнительные словари/таблицы допускаются для RCS, gamma0 и benchmark-метрик, если их удобнее сохранять в JSON/CSV.

---

## Helper API

Минимальный набор функций:

```python
def n_series_for_mode(mode: str) -> list[int]:
    ...

def make_sphere_case(N: int, *, eps_r: complex, k0a: float, a: float) -> ExperimentCase:
    ...

def make_anisotropic_ellipsoid_case(N: int, *, eps_real, eps_imag, k0: float) -> ExperimentCase:
    ...

def build_problem(case: ExperimentCase, *, precision=em3d.Precision.DOUBLE):
    ...

def estimate_gamma0(problem, *, coarse_N=(4, 4, 4)) -> em3d.gamma0.Gamma0Analysis:
    ...

def run_solver(problem, solver_name: str, *, gamma0_analysis=None, max_iter=500, rtol=1e-8) -> SolverRun:
    ...

def run_solver_suite(problem, solver_names: list[str], *, gamma0_analysis=None) -> list[SolverRun]:
    ...

def compute_mie_rcs_diagnostics(result_u, problem, *, a: float, eps_r: complex, n_phi=180) -> dict:
    ...

def save_runs_csv(runs: list[SolverRun], path) -> None:
    ...

def save_json(data: dict, path) -> None:
    ...
```

Функции построения графиков должны переиспользовать `em3d.vis`, а не дублировать визуализацию.

---

## Сценарии главы 6

### 6.1. Обоснование электродинамической постановки

Notebook фиксирует:

- неизвестное `u` — полное электрическое поле;
- операторная форма `(I - B eta) u = f`;
- `B` — dyadic Green kernel с 3×3 блоками;
- `eta` — локальный тензор диэлектрического контраста.

### 6.2. Анизотропные диэлектрические структуры

Минимальные cases:

- isotropic sphere для Mie/RCS validation;
- diagonal anisotropic ellipsoid;
- off-diagonal anisotropic ellipsoid, если solver устойчив на `quick` режиме.

### 6.3. Дискретный спектр и сходимость SIM

Для одного или двух representative cases:

- построить `Gamma0Analysis` на coarse-сетке;
- вывести `mu`, `radius`, `rho`;
- построить scatter спектра, convex hull и окружность через `plot_gamma0_spectrum`;
- связать `rho` с наблюдаемой сходимостью SIM.

### 6.4. Обобщённый метод простой итерации

Запуск SIM с параметрами из `gamma0_analysis.as_solver_config_kwargs()`.

Метрики:

- `converged`;
- `iterations`;
- `final_residual`;
- `elapsed_sec`;
- residual history plot.

### 6.5. BiCGStab и TwoStep

Сравнить `SIM`, `BiCGStab`, `TwoStep` на одинаковых cases и одинаковых `rtol/max_iter`.

Таблица:

| case | N | dof | solver | converged | iterations | final_residual | elapsed_sec |

### 6.6. FFT-ускоренный matvec

Для малых `N` допускается dense-vs-FFT consistency. Для всей `N_SERIES` считать только FFT timing.

Метрики:

- время сборки `Operator`;
- среднее время `matvec`;
- оценка dof;
- масштабирование относительно `N=8`.

### 6.7. Поле внутри области неоднородности

Использовать только поле `em3d`:

- `plot_field_scalar_slice`;
- `plot_field_vector_slice`;
- `plot_field_scalar_volume`;
- `plot_field_vector_volume`.

`mie_field_at` не использовать как near-field oracle до ремонта boundary-condition issue.

### 6.8. Диаграмма направленности и ЭПР

Для sphere-case:

- считать `em3d.mie.compare_rcs_plane(..., normalize="max")`;
- строить Cartesian и polar сравнения;
- публиковать `shape_err`, `scale_ratio`, `abs_rel_err`.

Абсолютный RCS-gate не делать основным критерием. Он остаётся диагностикой из-за voxelized sphere, effective radius и дискретного self-term.

Для anisotropic cases:

- строить RCS без Mie-эталона;
- сравнивать влияние тензора `eps`.

### 6.9. Вычислительная эффективность

По `N_SERIES` собрать таблицы:

- solver runtime;
- iterations;
- final residual;
- matvec timing;
- RCS timing.

Для финального notebook использовать `full` режим, но хранить результаты в `experiments/outputs/chapter06/`, чтобы повторный render мог читать готовые таблицы.

### 6.10. Выводы

Notebook должен отделять:

- автоматически рассчитанные численные факты;
- текстовые интерпретации автора;
- ограничения текущей реализации.

---

## Ограничения и риски

1. `mie_field_at` near-field не является эталоном до исправления boundary conditions.
2. Абсолютная Mie-RCS ошибка не является основным gate в первом экспериментальном notebook.
3. `N=64` может быть долгим для полного solver-suite. Notebook должен позволять отключать отдельные solver’ы или читать сохранённые результаты.
4. `gamma0` на coarse-сетке не должен использовать спектр FFT/circulant embedding как спектр исходного оператора.
5. Внешняя wiki может быть не обновлена автоматически из-за sandbox/approval ограничений; repo-spec является источником дизайна для реализации.

---

## Проверка

Минимальные проверки после реализации:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest -q -p no:cacheprovider -k "not saves_file"
```

Дополнительно:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m experiments.chapter06_em --mode quick
```

Если CLI не добавляется, вместо второй команды должен быть smoke-тест helper-модуля на `N=8`.

---

## Критерии готовности

- `experiments/chapter06_em.py` строит задачи и запускает quick experiments без ручного редактирования.
- `notebooks/chapter-06-em.ipynb` содержит секции 6.1–6.10.
- Quick mode завершается на CPU за разумное время.
- Full mode поддерживает `N=(8,16,24,32,40,48,56,64)`.
- Таблицы и фигуры сохраняются в `experiments/outputs/chapter06/`.
- Mie near-field не используется как oracle.
- Mie RCS используется как normalized-shape validation плюс scale diagnostics.
