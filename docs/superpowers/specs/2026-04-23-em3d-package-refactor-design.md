# Спецификация: рефакторинг ноутбука EM3D в Python-пакет `em3d`

- **Дата:** 2026-04-23
- **Источник:** `wiki/raw/notes/Yurchenkov-programming-code-EM3D-2026.ipynb`
- **Статус:** draft (одобрено автором по секциям в ходе брейнсторма)

---

## Цель

Переработать исследовательский Jupyter-ноутбук EM3D-решателя Юрченкова (2026) в структурированный Python-пакет `em3d`:

- единый интерфейс CPU/GPU через `xp`-namespace (numpy / cupy);
- параметризуемая численная точность — `SINGLE` (`float32`/`complex64`) или `DOUBLE` (`float64`/`complex128`) на уровне задачи;
- центральная часть — структурированная декартова сетка в объёме + *n*-мерное БПФ-ускорение свёртки («nFFT»-матвек) + итерационные методы, построенные поверх этого матвека;
- чистое разделение пакета (исходники) и вики (заметки).

Верификация против Mie/плавленого кварца, постпроцессинг (ЭПР) и визуализация — **вне скоупа** этой спецификации; для них предполагается отдельный последующий spec.

## Ключевые решения (зафиксированы в брейнсторме)

1. **Backend:** единая переменная `xp` выбирается из `{numpy, cupy}` по доступности CUDA. Никакого torch. Фактические различия API инкапсулируются в одном месте (`backend.py`).
2. **Точности:** две пары — `SINGLE: (float32, complex64)` и `DOUBLE: (float64, complex128)`; выбор на уровне `Backend` при создании, внутри всё согласовано.
3. **MVP-скоуп:**
   - структурированная декартова сетка;
   - ε-профили: цилиндр, ступенька, эллипс (+ `apply_refraction`);
   - плоская волна;
   - функция Грина Гельмгольца и коэффициенты `b_coeff`;
   - оператор на FFT (прямой `matvec` + сопряжённый `rmatvec`) на удвоенном параллелепипеде Π₂;
   - эталонный dense-оператор (только для тестов малой размерности);
   - γ₀-алгоритм (геометрия выпуклой оболочки + описанные окружности) и подбор итерационных параметров;
   - три итерационных метода: SIM (MSGD), BiCGStab, TwoStep.
4. **Расположение:** `src/em3d/` + `tests/` + `pyproject.toml` в корне рабочей директории, рядом с `wiki/` и `docs/`; заметки о пакете — страницы `wiki/code/em3d-*.md` со ссылками на исходники.
5. **Тесты:** unit + интеграционный gate «FFT-оператор vs dense-эталон на малой сетке». Верификация с miepython/кварцом отложена.
6. **Стиль API:** лёгкий ООП — классы-контейнеры состояния (`Backend`, `Grid`, `Problem`, `Operator`, `SolverConfig`, `BaseSolver` + три реализации), операции остаются функциями/методами без скрытого глобального состояния.

## Архитектура и структура директорий

```
<рабочая директория>/
├── pyproject.toml              # core deps: numpy, scipy. Extra "gpu": cupy.
├── src/
│   └── em3d/
│       ├── __init__.py         # реэкспорт публичного API
│       ├── backend.py          # выбор xp, dtype-пары, обёртки FFT при необходимости
│       ├── dtypes.py           # Precision enum
│       ├── grid.py             # Grid (структурированная декартова)
│       ├── refraction.py       # cylinder_refraction, step_refraction, ellipsis_refraction, apply_refraction
│       ├── wave.py             # flat_wave_vec
│       ├── kernel.py           # функция Грина Гельмгольца, b_coeff
│       ├── operator.py         # Operator: prep_coeffs + matvec/rmatvec через FFT на Π₂
│       ├── dense.py            # эталонный B_operator_matrix (только numpy, только для тестов)
│       ├── gamma0.py           # выпуклая оболочка, описанные окружности, find_params(λ)
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── base.py         # BaseSolver (Protocol), SolverResult, SolverConfig
│       │   ├── sim.py          # SIM (MSGD)
│       │   ├── bicgstab.py     # BiCGStab
│       │   └── twostep.py      # TwoStep
│       └── problem.py          # Problem = Grid + ε + волна + k₀ + объём Q
└── tests/
    ├── conftest.py             # фикстуры backend/dtype
    ├── test_grid.py
    ├── test_refraction.py
    ├── test_kernel.py
    ├── test_gamma0.py
    ├── test_operator_vs_dense.py
    └── test_solvers.py
```

**Принципы:**
- один модуль — одна ответственность, файлы <300 строк;
- связанные сущности рядом (решатели — одна папка);
- эталонный dense-оператор изолирован в `dense.py`, чтобы не попадал в основной граф импортов при работе на больших сетках;
- граф зависимостей ациклический: `grid ← refraction/wave/kernel ← operator ← problem ← solvers`.

## Компоненты и контракты

### `Precision` (enum)

`SINGLE` → `(float32, complex64)`, `DOUBLE` → `(float64, complex128)`. Одна истина для всей задачи.

### `Backend` (dataclass)

- Поля: `xp` (модуль `numpy` или `cupy`), `device: Literal["cpu","cuda"]`, `precision: Precision`.
- Свойства: `real_dtype`, `complex_dtype` — выводятся из `precision`.
- Методы:
  - `array(...)`, `zeros(shape, kind)` где `kind ∈ {"real","complex"}`;
  - `to_host(arr) -> np.ndarray` — явный перенос на CPU;
  - при необходимости — обёртки `fftn(x, axes)`, `ifftn(x, axes)` (если API numpy/cupy разойдутся, дифф скрыть здесь).
- Фабрики: `Backend.auto(precision)`, `Backend.numpy(precision)`, `Backend.cupy(precision)`.

### `Grid` (dataclass)

- Поля: `N: tuple[int,int,int]`, `L: tuple[float,float,float]`, `center: tuple[float,float,float]`, `backend: Backend`.
- Производные: `dv = ∏(L/N)`; массивы координат `x, y, z` хранятся как `xp.linspace`.
- Метод: `coords() -> (X, Y, Z)` через `xp.meshgrid`.

### `Problem` (dataclass)

- Поля:
  - `grid: Grid`;
  - `eps_tensor`: массив shape `(3, 3) + grid.N`, complex, η = ε − I;
  - `wave`: массив shape `(3,) + grid.N`, complex — падающая плоская волна на сетке;
  - `k0: float` — волновое число;
  - `volume: float` — объём внутренней подобласти Q (сумма `dv` по ячейкам внутри Q).
- Валидирует в конструкторе: все массивы — на `grid.backend`, правильный dtype, правильный shape.

### `Operator`

- Конструктор: `Operator(problem: Problem)`.
- В `__init__` вычисляет и кеширует `prep_coeffs` — FFT коэффициентов `b_coeff` на удвоенном параллелепипеде Π₂ (реализация `prep_coeffs_em3d` из ноутбука).
- Методы:
  - `matvec(u) -> xp.ndarray` — «nFFT»-матвек (эквивалент `operator(coeffs, u, eta)` из ноутбука);
  - `rmatvec(u) -> xp.ndarray` — сопряжённый матвек (эквивалент `conj_operator`);
  - `to_dense() -> np.ndarray` — делегирует в `dense.B_operator_matrix`, пригоден только для малых сеток; требует `backend.xp is numpy` (иначе `RuntimeError("to_dense requires numpy backend")`), используется тестами.
- Хранит ссылку на `Problem`, через неё — на `Backend`.

### `SolverConfig` (dataclass)

Общие поля: `max_iter: int`, `rtol: float`, `log: bool = False`.

Solver-специфичные поля (например, параметры γ₀ `mu`, `radius` для SIM) принимаются через `**kwargs` или через подклассы конфига — уточняется на этапе плана реализации.

### `BaseSolver` (Protocol)

Сигнатура: `solve(operator: Operator, rhs: xp.ndarray) -> SolverResult`.

`SolverResult` (dataclass): `u: xp.ndarray`, `iterations: int`, `residual_history: list[float]`, `converged: bool`.

### Реализации решателей

`SIM`, `BiCGStab`, `TwoStep` — каждая принимает `SolverConfig` в конструкторе и реализует `solve`. `TwoStep` дополнительно использует `operator.rmatvec`. Параметры γ₀ для SIM вычисляются через `gamma0.find_params(spectrum_samples)` вне решателя и передаются в `SolverConfig`.

### Ключевой инвариант

После создания `Backend` все массивы в `Grid/Problem/Operator` живут на том же устройстве с тем же dtype. Валидация — в конструкторе каждого класса: `assert arr.dtype == backend.complex_dtype` (или `real_dtype`) и `type(arr)` — ожидаемый `xp.ndarray`-тип.

## Поток данных

Типичный сценарий пользователя пакета:

```python
from em3d import Backend, Precision, Grid, Problem, Operator
from em3d.refraction import cylinder_refraction
from em3d.wave import flat_wave_vec
from em3d.solvers import TwoStep, SolverConfig
from em3d.gamma0 import find_params

# 1. Backend + точность — выбираются один раз
be = Backend.auto(precision=Precision.SINGLE)

# 2. Сетка
grid = Grid(N=(64,64,64), L=(1.0,1.0,1.0), center=(0,0,0), backend=be)

# 3. Физика: ε-профиль и падающая волна
eps = cylinder_refraction(grid, eps_real=2.25, eps_imag=0.0, radius=0.3)
wave = flat_wave_vec(grid, k=k0, orient=(0,0,1), amplitude=(1,0,0))

problem = Problem(grid=grid, eps_tensor=eps, wave=wave, k0=k0, volume=volume_Q)

# 4. Оператор — FFT коэффициентов считается один раз
op = Operator(problem)

# 5. Параметры итерации γ₀
cfg = SolverConfig(max_iter=200, rtol=1e-6, **find_params(spectrum_samples))

# 6. Решение
result = TwoStep(cfg).solve(op, rhs=problem.wave)
u = result.u
```

**Инварианты потока:**
- `Backend` создаётся первым и «протекает» через `Grid → Problem → Operator`. Перепрошивка на другое устройство/точность требует пересборки всей цепочки — это сознательное архитектурное решение, избегающее скрытых миграций.
- `Operator` — единственный «дорогой» объект (кеширует FFT коэффициентов). `solver.solve(op, rhs)` дёшев: меняем `rhs` — не пересобираем `op`.
- `SolverResult` возвращает массивы того же типа/устройства, что вход. Перенос на хост — явный: `be.to_host(result.u)`.

**Что скрыто в `Operator`:** удвоение сетки Π₂, FFT, блочная структура 3×3, раскладка тензора `(N,N,m,m) → (Nm,Nm)`. Пользователь этого не видит.

## Backend и dtype-контекст

### Фабричный выбор

```python
@classmethod
def auto(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
    try:
        import cupy as cp
        if cp.cuda.is_available():
            return cls(xp=cp, device="cuda", precision=precision)
    except ImportError:
        pass
    return cls(xp=numpy, device="cpu", precision=precision)
```

Плюс явные конструкторы `Backend.numpy(precision)`, `Backend.cupy(precision)` для тестов и сценариев принудительного выбора.

### Функции `xp.*`, используемые в пакете

Проверяется совпадение API в `numpy ≥ 1.26` и `cupy ≥ 13`:

- арифметика и broadcasting;
- `xp.fft.fftn`, `xp.fft.ifftn` — для Π₂-свёртки;
- `xp.linalg.norm`, `xp.vdot`, `xp.sum`, `xp.conj`;
- `xp.linspace`, `xp.meshgrid`, `xp.zeros`, `xp.empty`;
- `xp.roll`, `xp.reshape`, `xp.transpose`.

Все одинаковые. Тонкие места (если всплывут — например, аргумент `norm=` у FFT) инкапсулируются в `backend.py` как одна-две обёртки `be.fftn(x, axes)`, не размазываясь по модулям.

### Чего пакет **не** делает автоматически

- миграций данных между устройствами (никаких скрытых `cp.asnumpy`);
- смешанной точности (одна пара на задачу);
- fallback на другой backend при ошибке (явная жалоба лучше тихого спада в numpy).

### Изоляция импортов

В модулях пакета прямых `import cupy` или `import numpy as np` нет, кроме:
- `backend.py` — единственная точка выбора `xp`;
- `dense.py` — только `numpy`, так как эталон для тестов на малой сетке.

Решатели и операторы берут `xp` через `operator.backend.xp` / `problem.grid.backend.xp`.

## Обработка ошибок и валидация

Валидация только на границах — внутри кода доверяем типам и инвариантам.

### Где валидируем

- **Конструкторы `Grid`/`Problem`/`Operator`** — проверяют, что входные массивы:
  - имеют `.dtype == backend.complex_dtype` (или `real_dtype`, где уместно);
  - имеют ожидаемый shape (`eps_tensor.shape == (3,3)+grid.N`, `wave.shape == (3,)+grid.N` и т. п.);
  - принадлежат ожидаемому `xp.ndarray`-типу (согласованно с `backend.xp`).
  - Нарушение → `TypeError` / `ValueError` с указанием полученного и ожидаемого значения.
- **`Operator.__init__`** — требует, чтобы `problem.grid.backend is problem.wave.backend` и т. д. (ссылочное равенство одного `Backend`). Нарушение → `ValueError("mixed backends")`.
- **`Solver.solve`** — требует `rhs.shape == problem.wave.shape` и тот же dtype/backend, что у `Operator`.

### Что **не** валидируем

- Численные свойства ε-тензора (положительность, эрмитовость эрмитовых частей и т. п.) — это задача физики, не пакета.
- Сходимость итерации — возвращаем `SolverResult(converged=False, iterations=max_iter, ...)`, пользователь принимает решение; исключений не бросаем.
- NaN/Inf внутри итерации — не ловим; если метод разошёлся, это видно по `residual_history`.

### Особые случаи

- **`gamma0.find_params`** требует ≥ 2 точек спектра, все со строго положительной вещественной частью (иначе геометрия окружностей не определена). Нарушение → `ValueError` с указанием точки-нарушителя.

### Логи

`log=True` в `SolverConfig` — печатают номер итерации и относительную невязку через `print`. Без зависимостей на `logging`. Минимум машинерии.

### Инварианты, которые **не** превращаем в ассерты

- «N — степень двойки» — FFT эффективнее, но не обязателен.
- «Сетка центрирована в нуле» — поддерживаем произвольный `center`.

## Тестирование

Два уровня: unit + интеграция. Верификация с miepython и плавленым кварцом — в следующем spec (вместе с ЭПР/постпроцессингом).

### Фикстуры (`tests/conftest.py`)

- `backend_numpy_single`, `backend_numpy_double` — всегда активны;
- `backend_cupy_single`, `backend_cupy_double` — помечены `@pytest.mark.gpu` и пропускаются, если `cupy` недоступен или нет CUDA-устройства.
- Параметризация через `pytest.mark.parametrize("backend", backends)` там, где важно проверить обе платформы.

### Unit-тесты (<1 с каждый)

- `test_grid.py` — shape координат, шаг `L/N`, `dv = ∏(L/N)`, поведение при `center ≠ 0`.
- `test_refraction.py` — `cylinder/step/ellipsis` дают ожидаемую маску (сверка с прямым геометрическим критерием на малой сетке).
- `test_kernel.py` — значения функции Грина `G(R) = exp(ikR)/(4πR)` в нескольких точках; поведение при `R → 0` (если в коде регуляризация — убедиться, что не NaN).
- `test_gamma0.py` — ключевые случаи геометрии:
  - `compute_circle_two_points` — центр на серединном перпендикуляре;
  - `compute_circle_three_points` — описанная окружность для известного треугольника;
  - `find_params` на 3–5 искусственных точках — сверка с ручным расчётом;
  - `circle_contains_origin` — True/False на явных конфигурациях.

### Интеграционные тесты — gate рефакторинга

- **`test_operator_vs_dense.py`** на малой сетке `N = (4,4,4)` с простым ε-профилем:
  - собираем `Operator.matvec` и `dense.B_operator_matrix`;
  - прогоняем случайный вектор `u` через оба;
  - проверяем `||matvec(u) − dense @ u|| / ||dense @ u|| < 10·eps_dtype`;
  - то же для `rmatvec` vs `dense.conj().T`;
  - запускается для `(numpy, DOUBLE)` как эталон точности и для `(numpy, SINGLE)` с ослабленным порогом.
- **`test_solvers.py`** — для каждой пары `(solver ∈ {SIM, BiCGStab, TwoStep}, backend, precision)`:
  - задача малой размерности, где решение известно; для backend-агностичности `rhs` строится через сам FFT-оператор: `rhs = op.matvec(u_true)` со случайным `u_true`. Это избавляет тест от зависимости от `dense.py` и работает как на numpy, так и на cupy;
  - решатель достигает `rtol` за < `max_iter` итераций;
  - финальная относительная ошибка `||u − u_true|| / ||u_true||` в пределах ожидаемого порога.

### CI-гейт

- Локально основной прогон: `pytest -m "not gpu"`.
- На машине с CUDA дополнительно: `pytest -m gpu`.

### Что вне скоупа MVP

- бенчмарки производительности;
- тесты на больших сетках (> 64³);
- физическая валидация (Mie-рассеяние, плавленый кварц);
- тесты ЭПР и визуализации.

## Открытые вопросы на этап плана реализации

Эти пункты не блокируют дизайн, но будут решены при написании плана:

1. Конкретная сигнатура `find_params` и формат передачи его результата в `SolverConfig` (подклассы конфига vs `**kwargs`).
2. Единая конвенция именования методов: `matvec/rmatvec` (как в `scipy.sparse.linalg.LinearOperator`) — подтвердить, что это предпочтительнее, чем `apply/apply_conj`.
3. Список функций, которые будут выброшены из ноутбука как нерелевантные MVP (например, визуализационные `plot_scalar_*`, `compute_RCS`, сравнительные таблицы по плавленому кварцу) — они уйдут в следующий spec.
4. Стратегия миграции исследовательских данных из ноутбука: какие параметры (N, L, ε-профили, k₀) станут стандартными примерами/фикстурами.

## Следующий шаг

После одобрения этой спецификации — переход к скиллу `superpowers:writing-plans` для пошагового плана реализации с контрольными точками.
