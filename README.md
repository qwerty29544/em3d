# em3d

**Решатель объёмного интегрального уравнения (ОИУ) для трёхмерной электродинамики на структурированных декартовых сетках с БПФ-ускорением.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Обзор

Пакет `em3d` решает объёмное интегральное уравнение (ОИУ) задачи рассеяния ЭМ-волн на трёхмерной диэлектрической структуре:

$$(\mathbf{I} - \mathbf{B}\boldsymbol{\eta})\,\mathbf{u} = \mathbf{f}$$

| Символ | Смысл |
|--------|-------|
| $\mathbf{u}(\mathbf{r})$ | полное электрическое поле внутри расчётной области |
| $\mathbf{f}(\mathbf{r})$ | падающая плоская волна |
| $\boldsymbol{\eta}(\mathbf{r}) = \boldsymbol{\varepsilon}(\mathbf{r}) - \mathbf{I}$ | тензор диэлектрического контраста |
| $\mathbf{B}$ | диадический 3×3 оператор объёмного интеграла, построенный через $G(R) = \dfrac{e^{ik_0 R}}{4\pi R}$, $I$ и $\hat r\hat r^T$ |

Оператор $\mathbf{B}$ применяется как БПФ-свёртка на **удвоенном параллелепипеде $\Pi_2$**, что устраняет артефакты периодизации и даёт сложность $\mathcal{O}(N \log N)$ на итерацию.

---

## Возможности

- **БПФ-матвек** — трёхуровневая блочно-тёплицева свёртка диадического 3×3 Green-ядра на $\Pi_2$; плотная матрица не хранится.
- **Три итерационных метода** — SIM/MSGD, BiCGStab, двухшаговый градиентный спуск.
- **Параметр $\gamma_0$** — оптимальный итерационный параметр через спектр исходной dense-матрицы на грубой сетке, выпуклую оболочку, окружность минимального угла видимости и визуальную диагностику.
- **Два бэкенда** — NumPy (CPU) и CuPy (GPU/CUDA) с единым API.
- **Две точности** — `float64/complex128` (двойная) и `float32/complex64` (одинарная).
- **Типизирован** — маркер `py.typed` (PEP 561), аннотированный публичный API.
- **Аналитический эталон Ми** — `em3d.mie`: коэффициенты, сечения, ЭПР, ближнее поле и сравнение численной ЭПР с аналитической кривой для однородного изотропного шара.
- **Визуализация** — `em3d.vis`: раздельные скалярные и векторные графики поля в 2D/3D (`plot_field_scalar_slice`, `plot_field_vector_slice`, `plot_field_scalar_volume`, `plot_field_vector_volume`), диаграммы ЭПР в декартовых и полярных координатах (`plot_rcs`, `plot_rcs_polar`, `plot_rcs_comparison`, `plot_rcs_comparison_polar`), спектр и окружность $\gamma_0$ (`plot_gamma0_spectrum`).

---

## Установка

### Стабильная версия (v0.2.0) с GitHub

```bash
pip install git+https://github.com/qwerty29544/em3d.git@v0.2.0
```

### Последняя версия (main)

```bash
pip install git+https://github.com/qwerty29544/em3d.git
```

### С поддержкой визуализации (matplotlib)

```bash
pip install "em3d[vis] @ git+https://github.com/qwerty29544/em3d.git@v0.2.0"
```

### С поддержкой GPU (требует CUDA 12 и CuPy)

```bash
pip install "em3d[gpu] @ git+https://github.com/qwerty29544/em3d.git@v0.2.0"
```

### Локальная установка для разработки

```bash
git clone https://github.com/qwerty29544/em3d.git
cd em3d
pip install -e ".[dev]"
```

---

## Быстрый старт

```python
import numpy as np
import em3d

# ── 1. Бэкенд и сетка ─────────────────────────────────────────────────────
be   = em3d.Backend.numpy(em3d.Precision.DOUBLE)
grid = em3d.Grid(N=(16, 16, 16), L=(1.0, 1.0, 1.0),
                 center=(0.0, 0.0, 0.0), backend=be)

# ── 2. Диэлектрический цилиндр вдоль z (ε_r = 2, без поглощения) ─────────
eps_tensor = em3d.cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0,
                                       radius=0.3, axis="z")
# cylinder_refraction возвращает (3,3,Nx,Ny,Nz) — apply_refraction не нужен

# ── 3. Падающая волна (E ∥ x̂, распространение вдоль ẑ, k₀ = 1) ──────────
wave = em3d.flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))

# ── 4. Задача и оператор ──────────────────────────────────────────────────
problem = em3d.Problem(grid=grid, eps_tensor=eps_tensor,
                       wave=wave, k0=1.0, volume=grid.dv * 16**3)
op = em3d.Operator(problem)

# ── 5. Решение методом BiCGStab ───────────────────────────────────────────
cfg    = em3d.SolverConfig(max_iter=500, rtol=1e-8)
result = em3d.BiCGStab(cfg).solve(op, wave)

print(f"Сошлось: {result.converged}  итераций: {result.iterations}")
u = np.asarray(result.u)   # форма (3, 16, 16, 16), complex128
```

### Визуализация ЭПР

Модуль `em3d.farfield` вычисляет кривую ЭПР по результату решателя; `em3d.vis` строит её график. Для использования установите пакет с опцией `[vis]`.

```python
import matplotlib.pyplot as plt
from em3d.vis import plot_rcs, plot_rcs_polar

# u получено выше как np.asarray(result.u)
phi, sigma = em3d.farfield.rcs_plane(u, problem, n_phi=180, plane="xy")

# Декартовы координаты: sigma(φ)
fig, ax = plot_rcs(phi, sigma, title="ЭПР в плоскости xy")
plt.show()

# Полярные координаты, нормированные dБ
fig, ax = plot_rcs_polar(phi, sigma, db=True, title="ЭПР (дБ, полярные)")
plt.show()

# Сохранить в файл без вывода окна
plot_rcs(phi, sigma, filename="rcs_xy.png")
```

### Сравнение численной ЭПР шара с решением Ми

Для изотропной сферы можно построить численную ЭПР и аналитическую кривую Ми в одном масштабе. На текущем этапе полезнее сравнивать нормированную форму диаграммы: абсолютный масштаб отдельно сохраняется в `scale_ratio` и `abs_rel_err`.

```python
import matplotlib.pyplot as plt
import numpy as np

from em3d.vis import plot_rcs_comparison, plot_rcs_comparison_polar

a = 0.3
eps_r = 2.0
k0 = 1.0 / a
n = 32

be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
grid = em3d.Grid(N=(n, n, n), L=(1.0, 1.0, 1.0),
                 center=(0.0, 0.0, 0.0), backend=be)
eps_tensor = em3d.ellipsis_refraction(
    grid,
    eps_real=eps_r, eps_imag=0.0,
    center=(0.0, 0.0, 0.0), radius=(a, a, a),
)
wave = em3d.flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
problem = em3d.Problem(grid=grid, eps_tensor=eps_tensor,
                       wave=wave, k0=k0, volume=grid.dv * n**3)

result = em3d.BiCGStab(em3d.SolverConfig(max_iter=500, rtol=1e-8)).solve(
    em3d.Operator(problem), wave,
)
assert result.converged

comparison = em3d.mie.compare_rcs_plane(
    np.asarray(result.u),
    problem,
    a=a,
    eps_r=eps_r,
    n_phi=180,
    plane="xy",
    normalize="max",
)

print(f"shape_err={comparison['shape_err']:.2%}")
print(f"scale_ratio={comparison['scale_ratio']:.3f}")
print(f"abs_rel_err={comparison['abs_rel_err']:.2%}")

# Декартовы координаты: две нормированные кривые sigma(phi) на одном графике
plot_rcs_comparison(
    comparison["phi"],
    comparison["sigma_num_norm"],
    comparison["sigma_mie_norm"],
    title="Нормированная ЭПР: em3d vs Mie",
)

# Полярные координаты: та же нормированная диаграмма
plot_rcs_comparison_polar(
    comparison["phi"],
    comparison["sigma_num_norm"],
    comparison["sigma_mie_norm"],
    title="Нормированная ЭПР: em3d vs Mie",
)
plt.show()
```

### Визуализация поля

Скалярные и векторные графики строятся отдельными функциями, чтобы не смешивать фоновые карты и стрелочные поля в одной перегруженной картинке.

```python
from em3d.vis import (
    plot_field_scalar_slice,
    plot_field_scalar_volume,
    plot_field_vector_slice,
    plot_field_vector_volume,
)

u = np.asarray(result.u)   # (3, Nx, Ny, Nz) complex128

# 2D: скалярная карта нормы поля на срезе xy
fig, ax = plot_field_scalar_slice(u, grid, plane="xy", part="abs")
plt.show()

# 2D: только векторное поле на том же срезе
fig, ax = plot_field_vector_slice(u, grid, plane="xy", part="real", stride=2)
plt.show()

# 3D: скалярный scatter по норме поля
fig, ax = plot_field_scalar_volume(u, grid, part="abs", stride=2)
plt.show()

# 3D: только векторный объём
fig, ax = plot_field_vector_volume(u, grid, part="real", stride=2)
plt.show()

# Можно сохранить любой график
plot_field_scalar_slice(u, grid, plane="xy", part="abs", filename="field_scalar_xy.png")
```

Параметр `part` принимает значения `"real"` / `"imag"` / `"abs"`. Для скалярных графиков `component=None` означает норму вектора $\|\mathbf{F}\|$, а `component="x"`, `"y"` или `"z"` выбирает отдельную компоненту. Старые функции `plot_field_slice` и `plot_field_volume` сохранены для совместимости: первая строит комбинированный 2D-график, вторая является обёрткой над `plot_field_vector_volume`.

---

### SIM с оптимальным параметром $\gamma_0$

```python
import matplotlib.pyplot as plt

# Спектр берётся у исходного оператора H = I - B·η на грубой dense-сетке.
# Не используйте диагональ FFT-embedding как спектр H: это другой циркулянтный оператор.
analysis = em3d.gamma0.estimate_from_problem(problem, coarse_N=(4, 4, 4))

cfg_sim = em3d.SolverConfig(
    max_iter=500,
    rtol=1e-8,
    **analysis.as_solver_config_kwargs(),
)
result = em3d.SIM(cfg_sim).solve(op, wave)

print(f"mu={analysis.mu:.6g}, radius={analysis.radius:.6g}, rho={analysis.rho:.3f}")

fig, ax = em3d.vis.plot_gamma0_spectrum(
    analysis,
    title="Спектр грубого оператора и окружность gamma0",
)
plt.show()
```

### GPU-бэкенд

```python
be_gpu   = em3d.Backend.cupy(em3d.Precision.DOUBLE)   # требуется CuPy
grid_gpu = em3d.Grid(N=(32, 32, 32), L=(1.0, 1.0, 1.0),
                     center=(0.0, 0.0, 0.0), backend=be_gpu)
# Далее всё идентично CPU-варианту; оператор живёт на GPU
```

---

## Геометрии рефракции

Функции `cylinder_refraction`, `step_refraction`, `ellipsis_refraction` возвращают тензор контраста $\boldsymbol{\eta} = \boldsymbol{\varepsilon} - \mathbf{I}$ формы **(3, 3, Nx, Ny, Nz)** — готовый к передаче в `Problem` как `eps_tensor`. Вне заданной геометрической области значения равны нулю.

### Изотропный рассеиватель (скалярный ε)

Скалярные `eps_real` и `eps_imag` — это сахар для диагональной матрицы $\varepsilon_r \mathbf{I}$:

```python
eps_tensor = em3d.cylinder_refraction(grid, eps_real=2.25, eps_imag=0.0,
                                       radius=0.3, axis="z")
# eps_tensor.shape == (3, 3, Nx, Ny, Nz)
problem = em3d.Problem(..., eps_tensor=eps_tensor, ...)
```

### Анизотропный рассеиватель (матрица ε)

`eps_real` и `eps_imag` можно передать как массивы формы `(3, 3)`. Оба аргумента должны быть одного типа: либо оба скаляры, либо оба матрицы.

```python
import numpy as np

eps_r = np.array([[3.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.5]])   # диагональная анизотропия
eps_i = np.zeros((3, 3))

eps_tensor = em3d.ellipsis_refraction(
    grid,
    eps_real=eps_r, eps_imag=eps_i,
    center=(0.0, 0.0, 0.0), radius=(0.2, 0.2, 0.2),
)
```

### Несколько областей

Поскольку все функции возвращают обычные массивы NumPy/CuPy, области объединяются оператором `+`. Вне каждой области хранятся нули, поэтому сложение равносильно объединению.

**Непересекающиеся области:**

```python
eta_cylinder = em3d.cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0,
                                         radius=0.15, axis="z")
eta_slab     = em3d.step_refraction(grid, eps_real=1.5, eps_imag=0.02,
                                     z_min=0.3, z_max=0.45)

eps_tensor = eta_cylinder + eta_slab   # объединение двух материалов
problem = em3d.Problem(..., eps_tensor=eps_tensor, ...)
```

**Пересекающиеся области** при сложении получают суммарный контраст $\boldsymbol{\eta}_1 + \boldsymbol{\eta}_2$. Если нужно чтобы одна область перекрывала другую (одним материалом поверх другого), маски строятся вручную через `xp.where`:

```python
xp = grid.backend.xp
X, Y, Z = grid.coords()

mask_outer = (X**2 + Y**2) <= 0.3**2
mask_inner = (X**2 + Y**2) <= 0.1**2   # полость внутри цилиндра

eta_outer = complex(2.0 - 1.0, 0.0)   # η = ε - 1
eta_inner = complex(3.5 - 1.0, 0.05)

out = grid.backend.zeros((3, 3) + grid.N, kind="complex")
for d in range(3):
    out[d, d] = xp.where(mask_inner, eta_inner,
                xp.where(mask_outer, eta_outer, out[d, d]))

problem = em3d.Problem(..., eps_tensor=out, ...)
```

### `apply_refraction` — когда ещё нужен

`apply_refraction` остаётся полезным для **вручную построенных скалярных полей** η формы `(Nx, Ny, Nz)`, например после арифметики над массивами:

```python
# Градиентное распределение проницаемости вдоль z
_, _, Z = grid.coords()
scalar_eta = (1.5 - 1.0) * (1.0 + 0.3 * Z / grid.L[2])   # (Nx, Ny, Nz), float

eps_tensor = em3d.apply_refraction(grid, scalar_eta=scalar_eta)
```

> **Важно:** не передавайте результат geometry-функции в `apply_refraction(scalar_eta=...)` — он уже имеет форму `(3,3,Nx,Ny,Nz)` и будет отклонён с понятным сообщением об ошибке.

---

## Справочник модулей

| Модуль | Публичные символы | Назначение |
|--------|-------------------|------------|
| `em3d` | см. `__all__` | Верхнеуровневые реэкспорты |
| `em3d.backend` | `Backend`, `Precision` | Абстракция NumPy/CuPy, пары dtype |
| `em3d.grid` | `Grid` | Структурированная декартова сетка, объём ячейки, координаты |
| `em3d.refraction` | `cylinder_refraction`, `step_refraction`, `ellipsis_refraction`, `apply_refraction` | Построение тензора контраста $\boldsymbol{\eta}$; geometry-функции возвращают `(3,3,Nx,Ny,Nz)` напрямую (см. [секцию выше](#геометрии-рефракции)) |
| `em3d.wave` | `flat_wave_vec` | Выборка плоской волны на сетке |
| `em3d.problem` | `Problem` | Контейнер: сетка + $\boldsymbol{\varepsilon}$ + волна + $k_0$ |
| `em3d.operator` | `Operator` | БПФ-оператор ОИУ: `matvec` $(\mathbf{A})$, `rmatvec` $(\mathbf{A}^\dagger)$, `to_dense` |
| `em3d.gamma0` | `find_params`, `analyze_spectrum`, `estimate_from_problem`, `coarse_operator_matrix`, `sequential_chain`, `compute_circle_*` | $\gamma_0$ через грубый dense-спектр исходного оператора, выпуклую оболочку и окружность минимального угла видимости |
| `em3d.farfield` | `rcs`, `rcs_plane` | ЭПР в произвольном направлении и кривая ЭПР в координатной плоскости |
| `em3d.solvers` | `SIM`, `BiCGStab`, `TwoStep`, `SolverConfig`, `SolverResult`, `BaseSolver` | Итерационные методы |
| `em3d.mie` | `mie_coefficients`, `mie_cross_sections`, `mie_rcs_plane`, `compare_rcs_plane`, `mie_field_at`, `mie_field` | Аналитический эталон Ми и сравнение численной ЭПР с аналитической кривой |
| `em3d.vis` | `plot_rcs`, `plot_rcs_polar`, `plot_rcs_comparison`, `plot_rcs_comparison_polar`, `plot_gamma0_spectrum`, `plot_field_scalar_slice`, `plot_field_vector_slice`, `plot_field_scalar_volume`, `plot_field_vector_volume`, `plot_field_slice`, `plot_field_volume` | Визуализация ЭПР, спектра $\gamma_0$ и раздельных scalar/vector-графиков электромагнитного поля *(требует `pip install em3d[vis]`)* |

### Поля `SolverConfig`

| Поле | По умолч. | Описание |
|------|-----------|----------|
| `max_iter` | `200` | Максимальное число итераций |
| `rtol` | `1e-6` | Порог относительной невязки $\|\mathbf{Au}-\mathbf{f}\| / \|\mathbf{f}\|$ |
| `log` | `False` | Печатать невязку на каждой итерации |
| `mu` | `None` | Центр $\mu$ круга $\gamma_0$ (только SIM — из `gamma0.find_params`) |
| `radius` | `None` | Радиус $r$ круга $\gamma_0$ (только SIM) |

### Сравнение солверов

| Метод | Нужен $\gamma_0$ | Использует `rmatvec` | Примечание |
|-------|-----------------|----------------------|------------|
| `SIM` | ✅ | ✗ | Простая итерация; быстро при малом спектральном радиусе $\mathbf{B}\boldsymbol{\eta}$ |
| `BiCGStab` | ✗ | ✗ | Обычно наилучшая сходимость; параметры не нужны |
| `TwoStep` | ✗ | ✅ | Двухшаговый MSGD; эффективен для несамосопряжённых операторов |

---

### Эксперименты главы 6

Для диссертационных численных экспериментов добавлен CPU-first harness:

- `experiments/chapter06_em.py` — совместимый фасад для notebook и локальных запусков;
- `experiments/materials.py` — изотропные, анизотропные, поглощающие и Drude-плазменные материалы;
- `experiments/cases.py` — сферы, эллипсоиды, одноосный кристалл и слоистый параллелепипед;
- `experiments/scans.py` — `gamma0`, solver convergence, FFT-vs-dense и RCS scans;
- `experiments/experiment_logging.py` — внешний лог экспериментов в `raw/*.jsonl` и `raw/*.log`;
- `experiments/plots.py` — групповые графики срезов поля, FFT-vs-dense timing и RCS diagrams для notebook;
- `src/em3d/experiments/structured_lattice.py` — пакетный эксперимент решётки неоднородных включений, три солвера, ЭПР, field plots и логи метрик;
- `notebooks/chapter-06-em.ipynb` — narrative notebook по разделам 6.1–6.10.
- `notebooks/structured-lattice-kaggle.ipynb` — отдельный Kaggle-блокнот, запускающий эксперимент через установленный пакет.

В notebook раздел 6.6 сравнивает FFT-матвек с dense NumPy на `N=2..10`, раздел 6.8 строит нормированные диаграммы ЭПР для `k0a = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0]` при `a=0.5`, а раздел 6.10 содержит выключенный по умолчанию `RUN_CRASH_TEST=False` блок для TwoStep на слоистой структуре `N=128`.

Быстрый smoke-запуск:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m experiments.chapter06_em --mode quick --output-root experiments/outputs/chapter06-smoke --max-iter 50 --rtol 1e-5
```

Установка с GitHub на удалённой машине:

```bash
pip install "em3d[vis] @ git+https://github.com/qwerty29544/em3d.git"
```

Рабочий запуск решётки структурированных включений через пакетный API:

```python
from em3d.experiments.structured_lattice import (
    make_structured_lattice_case,
    run_structured_lattice_experiment,
)

case = make_structured_lattice_case(
    N=(100, 100, 100),
    coarse_N=(9, 9, 9),
    lattice_shape=(5, 5, 5),
)

summary = run_structured_lattice_experiment(
    case=case,
    output_root="experiments/outputs/structured-lattice-n100",
    max_iter=500,
    rtol=1e-6,
    rcs_n_phi=120,
)
print(summary)
```

## Математическое обоснование

### Дискретизация

Область рассеяния $Q \subset \mathbb{R}^3$ покрывается равномерной декартовой сеткой
$N = N_x \times N_y \times N_z$ ячеек объёма

$$\Delta V = \frac{L_x L_y L_z}{N_x N_y N_z}$$

Дискретное ОИУ в точках коллокации:

$$\mathbf{u}_i + \sum_{j=1}^{N} B_{ij}\,\boldsymbol{\eta}_j\,\mathbf{u}_j = \mathbf{f}_i, \qquad i = 1,\ldots,N$$

где

$$B_{ij} = \begin{cases} \displaystyle\int_0^{r_0} e^{ik_0 r}\,r\,dr, & i = j \\[6pt] \Delta V \cdot \dfrac{e^{ik_0|\mathbf{r}_i - \mathbf{r}_j|}}{4\pi|\mathbf{r}_i - \mathbf{r}_j|}, & i \neq j \end{cases}$$

Здесь $r_0 = \left(\dfrac{3\,\Delta V}{4\pi}\right)^{1/3}$ — радиус сферы, эквивалентной по объёму ячейке (формула исключённой сферы для самодействия).

### БПФ-ускорение

Матрица $\mathbf{B}$ является трёхуровневой блочно-тёплицевой: каждый блок — диадическая матрица 3×3. Вложение в удвоенную решётку $\Pi_2 = [0,\,2L]^3$ делает оператор блочно-циркулянтным; матвек становится:

$$(\mathbf{B}\boldsymbol{\eta}\mathbf{u})_i =
\mathrm{IFFT}\!\left[
\sum_{j=1}^{3}\hat{K}_{ij}\,
\mathrm{FFT}(\text{zero-pad}((\boldsymbol{\eta}\mathbf{u})_j))
\right]_{i},\quad i=1,2,3.$$

Стоимость: $\mathcal{O}(N\log N)$ против $\mathcal{O}(N^2)$ для плотной матрицы.

### Параметр $\gamma_0$ (для SIM)

Итерация SIM:

$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - \gamma_0\!\left(\mathbf{A}\mathbf{u}^{(k)} - \mathbf{f}\right), \qquad \mathbf{A} = \mathbf{I} - \mathbf{B}\boldsymbol{\eta}$$

сходится при $\gamma_0 = 1/\mu$, где $\mu$ — центр окружности, содержащей спектр исходного дискретного оператора $\mathbf{A}$ и не содержащей начало координат. На практике `em3d.gamma0.estimate_from_problem` строит грубую dense-матрицу $\mathbf{A}_{coarse}$, считает её собственные значения, строит выпуклую оболочку и выбирает окружность с минимальным отношением $R/|\mu|$.

Спектр расширенного циркулянтного FFT-embedding-оператора для этой настройки не используется: он соответствует другому оператору и не совпадает со спектром конечной блочно-тёплицевой матрицы $\mathbf{A}$.

Модуль `gamma0` реализует:
1. Алгоритм Эндрю (монотонная цепь) для выпуклой оболочки точек $\{\lambda_k\} \subset \mathbb{C}$.
2. Перебор пар и троек вершин для поиска допустимой окружности минимального угла видимости из начала координат.
3. Проверку, что $0 \notin$ МОО (иначе $\gamma_0$ не определён).

### Двухшаговый метод (TwoStep)

Минимизирует $\|\mathbf{A}\mathbf{u} - \mathbf{f}\|^2$ методом наискорейшего спуска:

$$\mathbf{p}^{(k)} = \mathbf{A}^\dagger\,\mathbf{r}^{(k)}, \qquad \tau_k = \frac{\|\mathbf{p}^{(k)}\|^2}{\|\mathbf{A}\mathbf{p}^{(k)}\|^2}, \qquad \mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - \tau_k\,\mathbf{p}^{(k)}$$

где $\mathbf{r}^{(k)} = \mathbf{A}\mathbf{u}^{(k)} - \mathbf{f}$.

---

## Разработка

```bash
# Клонировать и установить с dev-зависимостями
git clone https://github.com/qwerty29544/em3d.git
cd em3d
pip install -e ".[dev]"

# Запуск тестов (CPU)
pytest

# Только GPU-тесты (требует CUDA-устройство + CuPy)
pytest -m gpu

# Без GPU-тестов
pytest -m "not gpu"
```

---

## Зависимости

| Пакет | Версия |
|-------|--------|
| Python | ≥ 3.11 |
| NumPy | ≥ 1.26 |
| SciPy | ≥ 1.11 |
| CuPy | ≥ 13 *(опционально, GPU)* |

---

## Лицензия

MIT — см. файл [LICENSE](LICENSE).
