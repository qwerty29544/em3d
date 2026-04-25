# Спецификация: подпакет `em3d.farfield` — ЭПР и дальнезонная асимптотика

- **Дата:** 2026-04-24
- **Источник:** `wiki/raw/papers/Kulikov-Samokhin-1990.md`, `wiki/raw/notes/Yurchenkov-programming-code-EM3D-2026.ipynb`
- **Статус:** draft (одобрено автором по секциям в ходе брейнсторма)

---

## Цель

Добавить в пакет `em3d` подпакет `em3d.farfield` для расчёта дальнезонной характеристики и эффективной поверхности рассеяния (ЭПР) по результату решения объёмного интегрального уравнения.

Реализуются:
- **скалярный интеграл рассеяния** F(ê_p) — Фурье-образ поляризационного тока;
- **ЭПР одного направления** rcs(u, problem, direction);
- **ЭПР на плоскости** rcs_plane(u, problem, n_phi, plane) — кривая σ(φ);
- **два бэкенда вычисления**: Variant A (пакетный матмул) и Variant B (3D FFT + интерполяция).

Mie-верификация и визуализация — вне скоупа этой спецификации.

---

## Контекст и исправляемые ошибки ноутбука

Исходная функция `compute_RCS` из `Yurchenkov-programming-code-EM3D-2026.ipynb` (ячейка 11) содержит:

1. **Ошибка знака** — используется `exp(+1j * k0 * ...)` вместо `exp(-1j * k0 * ...)`. Это зеркалит угловой рисунок ЭПР.
2. **Ошибка памяти** — создаётся массив `waves` формы `(N1, N2, N3, n_phi)`: при сетке 100³ и 80 направлениях занимает ≈1,15 ГБ.

Оба дефекта устраняются в данной реализации.

---

## Физические формулы

### Интеграл рассеяния (уравнение 35–36 из Куликова–Самохина 1990)

```
F(ê_p) = ΔV · Σ_q  η_q · u_q · exp(−ik₀ · ê_p · r_q)
```

где:
- `η_q = eps_tensor[:, :, q]` — контрастный тензор η = ε − I (форма `(3, 3)`)
- `u_q = E[:, q]` — вектор поля в точке q (форма `(3,)`)
- `r_q` — координаты центра ячейки q
- `ΔV = grid.dv` — объём ячейки
- `ê_p` — единичный вектор направления наблюдения

Результат F(ê_p) имеет форму `(3,)` (вектор в 3D).

### ЭПР (уравнение 36)

```
σ(ê_p) = k₀⁴ / (16π²) · |ê_p × F(ê_p)|²
```

где `|ê_p × F(ê_p)|²` — квадрат нормы векторного произведения.

---

## Ключевые решения (зафиксированы в брейнсторме)

1. **Структура подпакета**: `src/em3d/farfield/__init__.py`, `_core.py`, `_fft.py`. Файл `__init__.py` реэкспортирует публичный API; каждый бэкенд-модуль содержит функцию `scatter_integral`.
2. **Два бэкенда**:
   - `method="direct"` → `_core.py`: пакетный матмул, O(3NM) вычислений, O(N + batch·3) памяти
   - `method="fft"` → `_fft.py`: 3D FFT (9 компонент J·ΔV) + fftshift + `map_coordinates` в `k·e·L/(2π)+N//2`
3. **Публичный API через `Problem`**: принимает `u: array (3, N1, N2, N3)` и `problem: Problem` — не требует отдельной передачи сетки/k0.
4. **Исправление знака**: `-1j * k0 * dot(e_p, r_q)` (не `+1j`).
5. **Фазовая коррекция для FFT-бэкенда**: умножение F на `exp(−ik₀ · ê_p · r₀)`, где r₀ = center − L/2 + Δr/2 — угол сетки.
6. **CuPy fallback**: в FFT-бэкенде `map_coordinates` вызывается через `scipy.ndimage`, предварительно перенося данные в NumPy, если бэкенд — CuPy.
7. **Интеграция с `em3d`**: `em3d.__init__` импортирует `from . import farfield` и добавляет `"farfield"` в `__all__`.

---

## Архитектура

```
src/em3d/farfield/
├── __init__.py      # реэкспорт: scatter_integral, rcs, rcs_plane
├── _core.py         # Variant A: пакетный матмул (backend-agnostic)
└── _fft.py          # Variant B: 3D FFT + map_coordinates

tests/
└── test_farfield.py  # 6 тестов
```

---

## Публичный API

### `scatter_integral(u, problem, directions, *, method="direct", batch_size=64)`

```python
def scatter_integral(
    u: array,             # (3, N1, N2, N3) complex — поле E
    problem: Problem,     # содержит eps_tensor, k0, grid
    directions: array,    # (M, 3) float — единичные векторы ê_p
    *,
    method: str = "direct",   # "direct" | "fft"
    batch_size: int = 64,     # используется только для method="direct"
) -> array:               # (M, 3) complex — F(ê_p) для каждого направления
```

**Формула:**
```
J = eta @ u     # (3, N1*N2*N3) — ток поляризации
F[m] = dv * J_flat @ phase[:, m]     # phase[n, m] = exp(-1j * k0 * dot(e_p[m], r[n]))
```

### `rcs(u, problem, direction)`

```python
def rcs(
    u: array,              # (3, N1, N2, N3) complex
    problem: Problem,
    direction: array,      # (3,) float — единичный вектор ê_p
) -> float:                # скалярная ЭПР в этом направлении
```

**Формула:** `k0^4 / (16*pi^2) * norm(cross(e_p, F))**2`

### `rcs_plane(u, problem, n_phi=80, plane="xy", method="direct", batch_size=64)`

```python
def rcs_plane(
    u: array,
    problem: Problem,
    n_phi: int = 80,
    plane: str = "xy",     # "xy" | "yz" | "xz"
    method: str = "direct",
    batch_size: int = 64,
) -> tuple[array, array]:  # (phi_rad, rcs_vals) — оба (n_phi,)
```

Генерирует `n_phi` равномерно распределённых направлений в плоскости `plane`, вызывает `scatter_integral`, возвращает `(phi, sigma)`.

---

## Variant A: Пакетный матмул (`_core.py`)

```
J = einsum('ij...,j...->i...', eta, u)    # (3, N1, N2, N3) ток поляризации
J_flat = J.reshape(3, -1)                 # (3, N) — N = N1*N2*N3
r_flat = coords_flat                      # (3, N) — координаты ячеек

for batch in chunks(directions, batch_size):
    phase = exp(-1j * k0 * (e_p @ r_flat))  # (batch, N)
    F_batch = dv * (J_flat @ phase.T)        # (3, batch)
```

Память: O(N) для J_flat, O(batch · N) для phase. При batch_size=64 и N=10⁶ это 64 · 10⁶ · 16 байт ≈ 1 ГБ — поэтому batch_size должен быть небольшим (16–64).

---

## Variant B: 3D FFT + интерполяция (`_fft.py`)

Шаги:
1. Вычислить J = η · u, умножить на ΔV: `J_dv[i, :, :, :] = eta[i, j, ...] * u[j, ...] * dv`
2. Применить `fftn` к каждому из 9 (или 3) компонентов J_dv на исходной сетке N1×N2×N3
3. `fftshift` — перенести нулевую частоту в центр
4. Для каждого направления ê_p = (ex, ey, ez) вычислить индексы интерполяции:
   ```
   ix = k0 * ex * Lx / (2*pi) + Nx/2
   iy = k0 * ey * Ly / (2*pi) + Ny/2
   iz = k0 * ez * Lz / (2*pi) + Nz/2
   ```
5. Интерполировать (`map_coordinates`, порядок 1) каждый компонент в найденной точке
6. Применить фазовую коррекцию: `F *= exp(-1j * k0 * dot(e_p, r0))`
   где `r0[i] = center[i] - L[i]/2 + (L[i]/N[i])/2`

**Вывод (почему FFT быстрее при M >> 1):** для M направлений прямой метод стоит O(3·N·M), FFT-метод — O(9·N·log N + 9·M).

---

## Тесты (`tests/test_farfield.py`)

| # | Название | Суть |
|---|----------|------|
| 1 | `test_zero_contrast` | η=0 → F=0, σ=0 для обоих методов |
| 2 | `test_rcs_nonnegative` | σ≥0 для произвольного ненулевого поля |
| 3 | `test_single_cell_analytic` | Сетка 1×1×1: F = η·u·ΔV·exp(−ik·r₀), проверить аналитически |
| 4 | `test_fft_vs_direct_agreement` | Для малой сетки (8×8×8): `method="fft"` совпадает с `method="direct"` до atol=1e-4 |
| 5 | `test_rcs_plane_shape` | `rcs_plane(..., n_phi=12)` возвращает (phi, sigma) формы (12,) |
| 6 | `test_rcs_plane_symmetry` | Изотропная сфера: ЭПР в плоскости xy симметрична (σ(φ) ≈ σ(φ+π)) |

Все тесты используют `numpy` бэкенд, `DOUBLE` точность, сетку ≤ 8³.

---

## Интеграция с основным пакетом

В `src/em3d/__init__.py` добавить:
```python
from . import farfield
```
и в `__all__` добавить `"farfield"`.

Версия пакета: остаётся `0.1.0` (минорная функциональность, не требует бампа до 0.2.0 в этой спеке).

---

## Не входит в скоуп

- Mie-верификация
- Полный тензор Грина Γ с членами ê⊗ê (только скалярное G·δᵢⱼ, как в текущем операторе)
- Визуализация
- Дальнезонное поле E_рас (формула 35) — только ЭПР (формула 36)
