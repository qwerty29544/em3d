# em3d.mie — Аналитическое решение Ми: дизайн

## Цель

Реализовать публичный модуль `em3d.mie` с точным аналитическим решением задачи Ми для изотропного диэлектрического шара. Модуль предоставляет дальнее поле (бистатическая ЭПР, интегральные сечения) и ближнее поле (полное поле в произвольных точках и на сетке `Grid`). Служит верификационным эталоном для сравнения с численным решением em3d.

---

## Скоуп

**В скоупе:**
- Коэффициенты Ми $a_n, b_n$ (внешние) и $c_n, d_n$ (внутренние)
- Интегральные сечения рассеяния $\sigma_{\text{scat}}$, экстинкции $\sigma_{\text{ext}}$, поглощения $\sigma_{\text{abs}}$
- Бистатическая ЭПР в координатной плоскости (`mie_rcs_plane`) — зеркальный интерфейс `farfield.rcs_plane`
- Ближнее поле в произвольных декартовых точках (`mie_field_at`)
- Обёртка ближнего поля на сетке `Grid` (`mie_field`) — прямое сравнение с `result.u`
- Поддержка произвольной ориентации волны через поворот координат
- Юнит-тесты модуля + интеграционный верификационный тест em3d vs. Ми

**Вне скоупа:**
- Многослойный шар
- Несферические геометрии
- Анизотропный рассеиватель
- Быстрое суммирование Ми при $x > 10$ (логарифмическая рекуррентная схема Вилера)

---

## Архитектура

### Файлы

| Файл | Действие | Назначение |
|------|----------|------------|
| `src/em3d/mie.py` | Создать | Всё аналитическое решение Ми |
| `src/em3d/__init__.py` | Изменить | Добавить `from . import mie` и `"mie"` в `__all__` |
| `tests/test_mie.py` | Создать | 6 юнит-тестов + 1 верификационный тест |

Новых зависимостей нет — `scipy >= 1.11` уже в `requires`.

---

## Публичный API

```python
__all__ = ["mie_coefficients", "mie_cross_sections", "mie_rcs_plane",
           "compare_rcs_plane", "mie_field_at", "mie_field"]
```

### `mie_coefficients`

```python
def mie_coefficients(a: float, eps_r: complex, k0: float) -> dict:
    """Коэффициенты Ми.

    Parameters
    ----------
    a     : float   — радиус шара (> 0)
    eps_r : complex — относительная диэлектрическая проницаемость
    k0    : float   — волновое число в свободном пространстве (> 0)

    Returns
    -------
    dict с ключами:
        "a" : ndarray (n_max,) complex — внешние коэффициенты TM
        "b" : ndarray (n_max,) complex — внешние коэффициенты TE
        "c" : ndarray (n_max,) complex — внутренние коэффициенты для M-гармоник
        "d" : ndarray (n_max,) complex — внутренние коэффициенты для N-гармоник
        "n_max" : int — порядок усечения
    """
```

### `mie_cross_sections`

```python
def mie_cross_sections(a: float, eps_r: complex, k0: float) -> dict:
    """Интегральные сечения.

    Returns
    -------
    dict: {"scat": float, "ext": float, "abs": float}
    """
```

### `mie_rcs_plane`

```python
def mie_rcs_plane(
    a: float,
    eps_r: complex,
    k0: float,
    n_phi: int = 180,
    plane: str = "xy",
) -> tuple:
    """Бистатическая ЭПР в координатной плоскости.

    Сигнатура зеркалит em3d.farfield.rcs_plane.
    Падающая волна: E ∥ x̂, распространение ∥ ẑ.

    Returns
    -------
    (phi, sigma) : оба ndarray (n_phi,) float64
    """
```

### `compare_rcs_plane`

```python
def compare_rcs_plane(
    u,
    problem,
    *,
    a: float,
    eps_r: complex,
    n_phi: int = 180,
    plane: str = "xy",
    method: str = "direct",
    batch_size: int = 64,
    normalize: str = "max",
) -> dict:
    """Сравнение численной и аналитической ЭПР в одной координатной плоскости.

    Возвращает сырые кривые, кривые после нормировки на максимум,
    shape_err, scale_ratio и abs_rel_err.
    """
```

`normalize="max"` нормирует только итоговые кривые ЭПР: `sigma / max(sigma)`. Нельзя нормировать каждый вектор дальнего поля `F(phi)` независимо, потому что это уничтожает диаграмму рассеяния. Абсолютный масштаб сохраняется в `scale_ratio` и `abs_rel_err`.

### `mie_field_at`

```python
def mie_field_at(
    xyz,
    a: float,
    eps_r: complex,
    k0: float,
    amplitude=(1, 0, 0),
    orient=(0, 0, 1),
) -> np.ndarray:
    """Полное поле E в произвольных декартовых точках.

    Parameters
    ----------
    xyz       : array (M, 3) float — декартовы координаты наблюдения
    a         : float   — радиус шара
    eps_r     : complex — ε_r шара
    k0        : float   — волновое число
    amplitude : array (3,) — поперечный вектор поляризации (не обязан быть единичным)
    orient    : array (3,) — направление распространения волны

    Returns
    -------
    ndarray (M, 3) complex128
        r < a : передаточное поле Ми (полное поле внутри шара)
        r ≥ a : E_inc + E_scat (полное поле снаружи)
    """
```

### `mie_field`

```python
def mie_field(
    grid,
    a: float,
    eps_r: complex,
    k0: float,
    amplitude=(1, 0, 0),
    orient=(0, 0, 1),
) -> np.ndarray:
    """Обёртка mie_field_at на сетке Grid.

    Returns
    -------
    ndarray (3, Nx, Ny, Nz) complex128 — прямое сравнение с result.u
    """
```

---

## Приватные вспомогательные функции

```python
def _n_max(x: float) -> int:
    """Критерий усечения Висконба: round(x + 4*x**(1/3) + 2), min 1."""

def _riccati(n: int, z: complex) -> tuple:
    """(psi_n, psi_n', xi_n, xi_n') через scipy.special.spherical_jn/yn."""

def _angle_functions(n_max: int, cos_theta: np.ndarray) -> tuple:
    """Массивы pi_n(cos_theta), tau_n(cos_theta) формы (n_max, M) через рекуррентность."""

def _build_frame(orient, amplitude) -> tuple:
    """Матрица поворота R (3×3): лабораторный фрейм → канонический (ẑ-пропагация, x̂-поляризация).
    Возвращает (R, E0) где E0 = скаляр амплитуды в канон. фрейме.
    Raises ValueError если amplitude ∥ orient.
    """

def _field_at_canonical(
    xyz_can: np.ndarray,
    a: float,
    k0: float,
    m: complex,
    coeffs: dict,
) -> np.ndarray:
    """Поле (M, 3) в каноническом фрейме для единичной амплитуды.
    Внутри шара: ряд по c_n M^(1) и d_n N^(1).
    Снаружи:     E_inc_can + ряд по a_n M^(3) и b_n N^(3).
    """
```

---

## Математические формулы

### Параметры

$$x = k_0 a, \qquad m = \sqrt{\varepsilon_r}, \qquad n_{\max} = \left\lfloor x + 4x^{1/3} + 2 \right\rceil$$

### Функции Риккати–Бесселя

$$\psi_n(z) = z\,j_n(z), \quad \psi_n'(z) = j_n(z) + z\,j_n'(z)$$
$$\xi_n(z) = z\,h_n^{(1)}(z), \quad \xi_n'(z) = h_n^{(1)}(z) + z\,{h_n^{(1)}}'(z)$$

Вронскиан: $\psi_n \xi_n' - \xi_n \psi_n' = i$

### Коэффициенты

$$a_n = \frac{m\,\psi_n(mx)\,\psi_n'(x) - \psi_n(x)\,\psi_n'(mx)}{m\,\psi_n(mx)\,\xi_n'(x) - \xi_n(x)\,\psi_n'(mx)}, \qquad b_n = \frac{\psi_n(mx)\,\psi_n'(x) - m\,\psi_n(x)\,\psi_n'(mx)}{\psi_n(mx)\,\xi_n'(x) - m\,\xi_n(x)\,\psi_n'(mx)}$$

$$c_n = \frac{i}{\psi_n(mx)\,\xi_n'(x) - m\,\xi_n(x)\,\psi_n'(mx)}, \qquad d_n = \frac{im}{m\,\psi_n(mx)\,\xi_n'(x) - \xi_n(x)\,\psi_n'(mx)}$$

### Интегральные сечения

$$\sigma_{\text{scat}} = \frac{2\pi}{k_0^2}\sum_{n=1}^{n_{\max}}(2n+1)(|a_n|^2+|b_n|^2), \qquad \sigma_{\text{ext}} = \frac{2\pi}{k_0^2}\sum_{n=1}^{n_{\max}}(2n+1)\,\mathrm{Re}(a_n+b_n)$$

### Угловые функции (рекуррентность)

$$\pi_0 = 0,\; \pi_1 = 1,\; \pi_n = \tfrac{2n-1}{n-1}\cos\theta\cdot\pi_{n-1} - \tfrac{n}{n-1}\,\pi_{n-2}; \qquad \tau_n = n\cos\theta\cdot\pi_n - (n+1)\,\pi_{n-1}$$

### Амплитуды рассеяния и ЭПР

$$S_1(\theta) = \sum_n \frac{2n+1}{n(n+1)}[a_n\pi_n + b_n\tau_n], \qquad S_2(\theta) = \sum_n \frac{2n+1}{n(n+1)}[a_n\tau_n + b_n\pi_n]$$

$$\sigma(\theta,\varphi) = \frac{|S_2(\theta)|^2\cos^2\!\varphi + |S_1(\theta)|^2\sin^2\!\varphi}{k_0^2}$$

Плоскость $xy$ ($\theta = \pi/2$): прямое сравнение с `farfield.rcs_plane(..., plane="xy")`.

### Ближнее поле — внутри шара ($r < a$), $k_{\text{int}} = mk_0$

$$\mathbf{E}_{\text{int}} = E_0\sum_{n=1}^{n_{\max}} i^n\frac{2n+1}{n(n+1)}\bigl[c_n\,\mathbf{M}^{(1)}_{o1n} - i\,d_n\,\mathbf{N}^{(1)}_{e1n}\bigr]$$

Компоненты $(\hat{r}, \hat{\theta}, \hat{\varphi})$:

| | $\hat{r}$ | $\hat{\theta}$ | $\hat{\varphi}$ |
|---|---|---|---|
| $\mathbf{M}^{(1)}_{o1n}$ | $0$ | $\cos\varphi\cdot\pi_n\cdot j_n(k_{\text{int}}r)$ | $-\sin\varphi\cdot\tau_n\cdot j_n(k_{\text{int}}r)$ |
| $\mathbf{N}^{(1)}_{e1n}$ | $\sin\theta\cos\varphi\cdot\pi_n\cdot\dfrac{n(n+1)\,j_n(k_{\text{int}}r)}{k_{\text{int}}r}$ | $\cos\varphi\cdot\tau_n\cdot\dfrac{\psi_n'(k_{\text{int}}r)}{k_{\text{int}}r}$ | $-\sin\varphi\cdot\pi_n\cdot\dfrac{\psi_n'(k_{\text{int}}r)}{k_{\text{int}}r}$ |

Особый случай $r = 0$: все члены суммы равны нулю кроме предела $n=1$ N-гармоники, поэтому $\mathbf{E}_{\text{int}}(0) = E_0 d_1\hat{x}$. В статическом пределе $k_0a \to 0$ это даёт классический результат $3E_0/(\varepsilon_r+2)$, а не падающее поле $E_0$.

### Ближнее поле — снаружи шара ($r \geq a$)

$$\mathbf{E} = \mathbf{E}^0 + \mathbf{E}^{\text{scat}}, \qquad \mathbf{E}^{\text{scat}} = E_0\sum_{n=1}^{n_{\max}}(-i)^n\frac{2n+1}{n(n+1)}\bigl[i\,a_n\,\mathbf{M}^{(3)}_{o1n} + b_n\,\mathbf{N}^{(3)}_{e1n}\bigr]$$

$\mathbf{M}^{(3)}, \mathbf{N}^{(3)}$ — те же формулы с $j_n \to h_n^{(1)}$.

### Перевод сферических компонент в декартовы

$$\begin{pmatrix}E_x\\E_y\\E_z\end{pmatrix} = \begin{pmatrix}\sin\theta\cos\varphi & \cos\theta\cos\varphi & -\sin\varphi \\ \sin\theta\sin\varphi & \cos\theta\sin\varphi & \cos\varphi \\ \cos\theta & -\sin\theta & 0\end{pmatrix}\begin{pmatrix}E_r\\E_\theta\\E_\varphi\end{pmatrix}$$

### Связь с `result.u`

Уравнение em3d $(I - B\eta)\mathbf{u} = \mathbf{f}$ — это уравнение Липпмана–Швингера для полного поля: $\mathbf{u} = \mathbf{E}_{\text{total}}$ всюду. Следовательно `mie_field(grid, ...)` даёт прямой эталон для сравнения с `result.u`.

---

## Обработка ошибок

| Условие | Исключение |
|---------|------------|
| `a <= 0` | `ValueError: a must be > 0, got {a}` |
| `k0 <= 0` | `ValueError: k0 must be > 0, got {k0}` |
| `real(eps_r) <= 0 and imag(eps_r) == 0` | `ValueError: Re(eps_r) must be > 0 for lossless medium` |
| `xyz.ndim != 2 or xyz.shape[1] != 3` | `ValueError: xyz must have shape (M, 3), got shape={xyz.shape}` |
| `plane` не в `{"xy","xz","yz"}` | `ValueError: plane must be 'xy', 'xz', or 'yz', got {plane!r}` |
| `n_phi < 1` | `ValueError: n_phi must be >= 1, got {n_phi}` |
| `amplitude ∥ orient` | `ValueError: amplitude must not be parallel to orient` |
| `amplitude` имеет продольную компоненту | `ValueError: amplitude must be transverse to orient` |
| `orient` или `amplitude` нулевой | `ValueError: orient/amplitude must be non-zero` |
| `x > 10` | `UserWarning: mie_coefficients: size parameter x={x:.2f} > 10; series may be inaccurate` |

---

## Тесты (`tests/test_mie.py`)

### Юнит-тесты (без решателя)

**`test_mie_coefficients_rayleigh_limit`**
При $x = 0.05$, $\varepsilon_r = 2.0$: первый коэффициент $a_1$ должен совпадать с пределом Релея
$a_1 \approx -\tfrac{2i}{3}\tfrac{m^2-1}{m^2+2}x^3$ с точностью $10^{-4}$ по относительной ошибке.

**`test_mie_cross_sections_lossless_zero_absorption`**
При вещественном $\varepsilon_r$: `cross["abs"]` < `1e-10 * cross["scat"]`.

**`test_mie_cross_sections_rayleigh`**
При $x = 0.1$, $\varepsilon_r = 2.0$: сравнение $\sigma_{\text{scat}}$ с формулой Релея
$\sigma_{\text{Релей}} = \tfrac{128\pi^5 a^6}{3\lambda^4}\left|\tfrac{m^2-1}{m^2+2}\right|^2$, допуск $1\%$.

**`test_mie_rcs_plane_symmetry`**
$\sigma_{xy}(\varphi) = \sigma_{xy}(\varphi + \pi)$ для всех $\varphi$ (двукратная симметрия плоскости рассеяния).

**`test_mie_field_matches_grid_wrapper`**
Значения `mie_field_at(xyz, ...)` и `mie_field(grid, ...)` совпадают в одних и тех же точках с точностью `1e-12`.

**`test_mie_field_center_is_continuous`**
`mie_field_at([[0,0,0]], a, eps_r, k0)` совпадает с пределом поля при $r \to 0$.

**`test_mie_field_center_rayleigh_static_limit`**
При $k_0a \ll 1$ поле в центре стремится к $3E_0/(\varepsilon_r+2)$.

### Интеграционный верификационный тест

**`test_mie_verification_rcs_normalized_shape`**

Этот gate сравнивает форму диаграммы ЭПР после нормировки на максимум, а абсолютный масштаб фиксирует отдельной диагностикой. Причина: даже после восстановления dyadic Green kernel абсолютная ЭПР зависит от voxelized sphere, эффективного радиуса и дискретного self-term, тогда как угловая форма уже является более устойчивым validation signal.

```python
@pytest.mark.parametrize("eps_r,k0a", [(2.0, 1.0), (1.5, 0.5)])
def test_mie_verification_rcs_normalized_shape(eps_r, k0a, backend_numpy_double):
    a = 0.3
    k0 = k0a / a
    L = 1.0
    N = 32

    be   = backend_numpy_double
    grid = em3d.Grid(N=(N, N, N), L=(L, L, L), center=(0,0,0), backend=be)
    eta  = em3d.ellipsis_refraction(grid, eps_real=eps_r, eps_imag=0.0,
                                    center=(0,0,0), radius=(a, a, a))
    wave = em3d.flat_wave_vec(grid, k=k0, orient=(0,0,1), amplitude=(1,0,0))
    prob = em3d.Problem(grid=grid, eps_tensor=eta, wave=wave,
                        k0=k0, volume=grid.dv * N**3)
    op   = em3d.Operator(prob)
    cfg  = em3d.SolverConfig(max_iter=500, rtol=1e-8)
    res  = em3d.BiCGStab(cfg).solve(op, wave)
    assert res.converged

    comparison = em3d.mie.compare_rcs_plane(
        np.asarray(res.u), prob, a=a, eps_r=eps_r,
        n_phi=180, plane="xy", normalize="max",
    )

    assert comparison["shape_err"] <= 0.02
    assert 0.5 <= comparison["scale_ratio"] <= 2.0
```

Строгий абсолютный gate `max(abs(sigma_num - sigma_mie)) / max(sigma_mie) <= 0.10` переносится в следующий этап после проверки эффективного радиуса voxelized sphere и дискретного self-term.

### Сравнительная визуализация ЭПР

Для визуальной проверки используются нормированные кривые из `compare_rcs_plane`:

```python
comparison = em3d.mie.compare_rcs_plane(
    result.u, problem, a=a, eps_r=eps_r, n_phi=180, plane="xy",
)

em3d.vis.plot_rcs_comparison(
    comparison["phi"],
    comparison["sigma_num_norm"],
    comparison["sigma_mie_norm"],
    title=f"Normalized RCS: shape_err={comparison['shape_err']:.2%}",
)

em3d.vis.plot_rcs_comparison_polar(
    comparison["phi"],
    comparison["sigma_num_norm"],
    comparison["sigma_mie_norm"],
    title="Normalized RCS polar comparison",
)
```

Декартовый график показывает `sigma(phi)` как обычную функцию угла. Полярный график показывает ту же нормированную диаграмму в геометрически привычном виде. Обе функции принимают уже подготовленные кривые и не нормируют данные неявно.

---

## Предупреждение о большом параметре размера

```python
if x > 10:
    import warnings
    warnings.warn(
        f"mie_coefficients: size parameter x={x:.2f} > 10; "
        f"Riccati-Bessel series may lose accuracy — consider logarithmic recurrence",
        UserWarning,
        stacklevel=3,
    )
```
