# em3d

Исследовательский пакет Python для моделирования трёхмерного рассеяния электромагнитных и акустических волн методом объёмных интегральных уравнений. Текущая версия пакета в ветке `main` — `0.8.0`.

## Возможности

- **Электродинамика.** Расчёт векторного поля в изотропных и анизотропных материалах, в том числе с поглощением. Модуль `em3d.geometry` задаёт материал по центрам ячеек и поддерживает объёмные доли для прямоугольных включений.
- **Акустика.** Решение скалярной задачи рассеяния, расчёт амплитуды дальнего поля и диаграмм рассеяния в модуле `em3d.acoustics`.
- **Численные методы.** `SIM`, `BiCGStab` и `TwoStep` работают с БПФ-операторами на равномерных декартовых сетках. Доступны NumPy на CPU и CuPy на GPU, одинарная и двойная точность.
- **Спектральный анализ.** Локализация спектра, приближения Арнольди и эксперименты по переносу параметра простой итерации между сетками.
- **Проверка и визуализация.** Сечения и объёмные графики полей, диаграммы эффективной площади рассеяния (ЭПР) и аналитическое решение Ми для сравнения с результатами расчёта однородного шара.

Электромагнитное уравнение имеет вид

$$
A\mathbf u = \mathbf f,\qquad A = I - B\chi,\qquad \chi = \varepsilon_r - I.
$$

Здесь $\mathbf u$ — полное электрическое поле, $\mathbf f$ — падающая волна, $B$ — дискретный оператор с диадическим ядром Грина. Для применения $B$ используется свёртка на сетке, удвоенной по каждой оси. Одно применение оператора требует $O(N\log N)$ операций и $O(N)$ памяти при $N=N_xN_yN_z$ ячейках.

## Установка

Требуется Python 3.10 или новее. Команда установит пакет из `main` вместе с NumPy, SciPy, Matplotlib и pandas.

```bash
python -m pip install "git+https://github.com/qwerty29544/em3d.git@main"
```

Для расчётов на GPU выберите вариант CuPy, соответствующий среде CUDA.

```bash
# Для CUDA 12
python -m pip install "em3d[cuda12] @ git+https://github.com/qwerty29544/em3d.git@main"

# Для CUDA 13
python -m pip install "em3d[cuda13] @ git+https://github.com/qwerty29544/em3d.git@main"
```

Передайте `em3d.Backend.cupy()` при создании сетки, чтобы использовать GPU.

## Электромагнитное рассеяние

Пример рассчитывает поле диэлектрического шара на CPU, сохраняет диаграмму ЭПР в плоскости $xz$ и сравнивает её с решением Ми.

```python
from math import prod

import em3d
from em3d.geometry import Ellipsoid, sample_contrast_tensor

center = (0.0, 0.0, 0.0)
grid = em3d.Grid(
    N=(16, 16, 16), L=(1.0, 1.0, 1.0), center=center,
    backend=em3d.Backend.numpy(),
)
radius, eps_r, k0 = 0.25, 2.0, 4.0
contrast = sample_contrast_tensor(
    grid,
    geometry=Ellipsoid(center=center, radii=(radius,) * 3),
    background_eps_r=1.0,
    feature_eps_r=eps_r,
)
wave = em3d.flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
problem = em3d.Problem(
    grid=grid, eps_tensor=contrast, wave=wave, k0=k0, volume=prod(grid.L),
)
solver = em3d.BiCGStab(em3d.SolverConfig(max_iter=200, rtol=1e-8))
result = solver.solve(em3d.Operator(problem), wave)
if not result.converged:
    raise RuntimeError(f"BiCGStab завершился без сходимости ({result.status})")

print(f"Число итераций — {result.iterations}.")
print(f"Относительная невязка — {result.true_final_residual:.2e}.")

comparison = em3d.mie.compare_rcs_plane(
    result.u, problem, a=radius, eps_r=eps_r, n_phi=120, plane="xz",
)
em3d.vis.plot_rcs(comparison["phi"], comparison["sigma_num"], filename="rcs-xz.png")
print(f"Относительная ошибка формы ЭПР — {comparison['shape_err']:.2%}.")
print(f"Отношение максимумов ЭПР — {comparison['scale_ratio']:.3f}.")
```

Поле `eps_tensor` конструктора `Problem` принимает тензор контраста $\chi$ формы `(3, 3, Nx, Ny, Nz)`. Поле решения `result.u` имеет форму `(3, Nx, Ny, Nz)`. При сравнении ЭПР `shape_err` измеряет расхождение нормированных диаграмм, а `scale_ratio` показывает отношение численного и аналитического максимумов.

### Выбор решателя

| Метод | Условия применения |
| --- | --- |
| `BiCGStab` | Использует прямое действие оператора `matvec`. |
| `SIM` | Использует `matvec` и спектральный параметр $\gamma_0=1/\mu$, где $\mu$ — центр круга локализации спектра. `em3d.gamma0.estimate_from_problem` оценивает параметры по матрице оператора на грубой сетке. |
| `TwoStep` | Использует прямое и сопряжённое действия оператора `matvec` и `rmatvec`. |

В серии расчётов с одинаковыми сеткой, бэкендом, точностью и волновым числом `PreparedEMKernel` позволяет повторно использовать спектр ядра при смене материала. На малых сетках `Operator.to_dense_operator()` позволяет сопоставить БПФ-вычисления с плотной матрицей $A$.

## Акустическое рассеяние

Скалярный оператор использует контраст $\chi=\eta-1$, где $\eta=1$ соответствует однородному фону. Пример вычисляет нормированную диаграмму рассеяния акустической волны на шаре.

```python
import em3d
from em3d import acoustics

grid = em3d.Grid(
    N=(16, 16, 16), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0),
    backend=em3d.Backend.numpy(),
)
eta = acoustics.eta_sphere(grid, center=grid.center, radius=0.25, eta_inside=2.0)
problem = acoustics.make_acoustic_problem(grid, eta, k0=4.0)
solver = em3d.BiCGStab(em3d.SolverConfig(rtol=1e-8))
result = solver.solve(acoustics.AcousticOperator(problem), problem.wave)
if not result.converged:
    raise RuntimeError(f"BiCGStab завершился без сходимости ({result.status})")

phi, pattern = acoustics.pattern_plane(result.u, problem, plane="xz", n_angles=120)
print(f"Рассчитано направлений — {phi.size}.")
acoustics.visualization.plot_pattern(phi, pattern, filename="acoustic-pattern-xz.png")
```

## Воспроизведение экспериментов

Для запуска сценариев из `experiments/` клонируйте репозиторий и установите пакет в режиме разработки.

```bash
git clone https://github.com/qwerty29544/em3d.git
cd em3d
python -m pip install -e .
```

Короткий запуск вычислительной серии главы 4 на CPU выполняет спектральные эксперименты и проверяет решатели на малых сетках. Манифесты и таблицы сохраняются в `out/chapter4-smoke/`.

```bash
python -m experiments.chapter4_final.cli --stage smoke --device cpu --no-figures --output-root out/chapter4-smoke
```

- Протокол главы 4, серии для больших сеток и сборка результатов. [Вычислительная серия главы 4](docs/chapter4_final_campaign.md)
- Сравнение решателей по истинной невязке, полю и ЭПР, включая сравнение с решением Ми. [Методика проверки](docs/chapter4_stationary_validation.md)
- Построение спектров и перенос $\gamma_0$ на мелкие сетки. [Описание спектральных экспериментов](docs/spectral_transfer_refactor.md) · [Блокнот](notebooks/em-spectral-transfer.ipynb)
- Запуск полной расчётной серии на GPU. [Блокнот Kaggle](notebooks/em-chapter4-final-kaggle.ipynb)
- Эксперименты для решётки включений и акустического рассеяния. [Блокнот для решётки включений](notebooks/structured-lattice-kaggle.ipynb) · [Блокнот по акустике](notebooks/acoustic-scattering-kaggle.ipynb)

## Тесты

Из корня локального клона запустите тесты на CPU.

```bash
python -m pip install pytest nbformat
python -m pytest tests -m "not gpu"
```

GPU-тесты запускаются командой `python -m pytest tests -m gpu` в среде с CuPy и доступным устройством CUDA.

## Лицензия

Пакет распространяется по лицензии MIT. [Текст лицензии](LICENSE)
