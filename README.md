# em3d

**Решатель объёмного интегрального уравнения (ОИУ) для трёхмерной электродинамики на структурированных декартовых сетках с БПФ-ускорением.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Обзор

Пакет `em3d` решает объёмное интегральное уравнение (ОИУ) задачи рассеяния ЭМ-волн на трёхмерной диэлектрической структуре:

$$(\mathbf{I} + \mathbf{B}\boldsymbol{\eta})\,\mathbf{u} = \mathbf{f}$$

| Символ | Смысл |
|--------|-------|
| $\mathbf{u}(\mathbf{r})$ | рассеянное электрическое поле внутри рассеивателя |
| $\mathbf{f}(\mathbf{r})$ | падающая плоская волна |
| $\boldsymbol{\eta}(\mathbf{r}) = \boldsymbol{\varepsilon}(\mathbf{r}) - \mathbf{I}$ | тензор диэлектрического контраста |
| $\mathbf{B}$ | оператор объёмного интеграла с функцией Грина Гельмгольца $G(R) = \dfrac{e^{ik_0 R}}{4\pi R}$ |

Оператор $\mathbf{B}$ применяется как БПФ-свёртка на **удвоенном параллелепипеде $\Pi_2$**, что устраняет артефакты периодизации и даёт сложность $\mathcal{O}(N \log N)$ на итерацию.

---

## Возможности

- **БПФ-матвек** — тёплицева свёртка на $\Pi_2$; плотная матрица не хранится.
- **Три итерационных метода** — SIM/MSGD, BiCGStab, двухшаговый градиентный спуск.
- **Параметр $\gamma_0$** — оптимальный итерационный параметр через выпуклую оболочку и наименьшую описанную окружность выборки спектра.
- **Два бэкенда** — NumPy (CPU) и CuPy (GPU/CUDA) с единым API.
- **Две точности** — `float64/complex128` (двойная) и `float32/complex64` (одинарная).
- **Типизирован** — маркер `py.typed` (PEP 561), аннотированный публичный API.

---

## Установка

### Стабильная версия (v0.1.0) с GitHub

```bash
pip install git+https://github.com/qwerty29544/em3d.git@v0.1.0
```

### Последняя версия (main)

```bash
pip install git+https://github.com/qwerty29544/em3d.git
```

### С поддержкой GPU (требует CUDA 12 и CuPy)

```bash
pip install "git+https://github.com/qwerty29544/em3d.git@v0.1.0[gpu]"
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
scalar_eta = em3d.cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0,
                                       radius=0.3, axis="z")
eps_tensor = em3d.apply_refraction(grid, scalar_eta=scalar_eta)

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

### SIM с оптимальным параметром $\gamma_0$

```python
import em3d.gamma0 as g0

# Берём несколько значений оператора для оценки спектра B·η
samples = []
rng = np.random.default_rng(42)
for _ in range(30):
    v    = be.array((rng.standard_normal((3,) + grid.N)
                     + 1j * rng.standard_normal((3,) + grid.N)).astype(np.complex128))
    Av   = np.asarray(op.matvec(v))
    v_np = np.asarray(v)
    # приближение собственного значения B·η через отношение Рэлея
    num = np.vdot(v_np.ravel(), (Av - v_np).ravel())
    den = np.vdot(v_np.ravel(), v_np.ravel())
    if abs(den) > 0:
        samples.append(complex(num / den))

params  = g0.find_params(samples)          # {"mu": ..., "radius": ...}
cfg_sim = em3d.SolverConfig(max_iter=500, rtol=1e-8,
                             mu=params["mu"], radius=params["radius"])
result  = em3d.SIM(cfg_sim).solve(op, wave)
```

### GPU-бэкенд

```python
be_gpu   = em3d.Backend.cupy(em3d.Precision.DOUBLE)   # требуется CuPy
grid_gpu = em3d.Grid(N=(32, 32, 32), L=(1.0, 1.0, 1.0),
                     center=(0.0, 0.0, 0.0), backend=be_gpu)
# Далее всё идентично CPU-варианту; оператор живёт на GPU
```

---

## Справочник модулей

| Модуль | Публичные символы | Назначение |
|--------|-------------------|------------|
| `em3d` | см. `__all__` | Верхнеуровневые реэкспорты |
| `em3d.backend` | `Backend`, `Precision` | Абстракция NumPy/CuPy, пары dtype |
| `em3d.grid` | `Grid` | Структурированная декартова сетка, объём ячейки, координаты |
| `em3d.refraction` | `cylinder_refraction`, `step_refraction`, `ellipsis_refraction`, `apply_refraction` | Построение тензора контраста $\boldsymbol{\eta}$ |
| `em3d.wave` | `flat_wave_vec` | Выборка плоской волны на сетке |
| `em3d.problem` | `Problem` | Контейнер: сетка + $\boldsymbol{\varepsilon}$ + волна + $k_0$ |
| `em3d.operator` | `Operator` | БПФ-оператор ОИУ: `matvec` $(\mathbf{A})$, `rmatvec` $(\mathbf{A}^\dagger)$, `to_dense` |
| `em3d.gamma0` | `find_params`, `sequential_chain`, `compute_circle_*` | $\gamma_0$ через выпуклую оболочку + МОО |
| `em3d.solvers` | `SIM`, `BiCGStab`, `TwoStep`, `SolverConfig`, `SolverResult`, `BaseSolver` | Итерационные методы |

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

Матрица $\mathbf{B}$ тёплицева на периодической решётке. Вложение в удвоенную решётку $\Pi_2 = [0,\,2L]^3$ делает оператор циркулянтным; матвек становится:

$$(\mathbf{B}\boldsymbol{\eta}\mathbf{u})_i = \mathrm{IFFT}\!\left[\,\hat{K} \odot \mathrm{FFT}(\text{zero-pad}(\boldsymbol{\eta}\mathbf{u}))\,\right]_{i},\quad i \leq N$$

Стоимость: $\mathcal{O}(N\log N)$ против $\mathcal{O}(N^2)$ для плотной матрицы.

### Параметр $\gamma_0$ (для SIM)

Итерация SIM:

$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - \gamma_0\!\left(\mathbf{A}\mathbf{u}^{(k)} - \mathbf{f}\right), \qquad \mathbf{A} = \mathbf{I} + \mathbf{B}\boldsymbol{\eta}$$

сходится при $\gamma_0 = 1/\mu$, где $\mu$ — центр **наименьшей описанной окружности** (МОО) выпуклой оболочки выборки спектра $\mathbf{B}\boldsymbol{\eta}$.

Модуль `gamma0` реализует:
1. Алгоритм Эндрю (монотонная цепь) для выпуклой оболочки точек $\{\lambda_k\} \subset \mathbb{C}$.
2. Перебор пар и троек вершин для поиска МОО.
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
