# Gamma0 Research Tools Design

## Цель

Добавить в `em3d` исследовательский блок для настройки и визуального контроля обобщённого метода простой итерации: оценка спектра исходного оператора на грубой сетке, построение выпуклой оболочки методом Эндрю, поиск оптимального параметра `gamma0`/`mu` по геометрическому алгоритму и визуализация спектра, hull и окружности.

## Контекст

Текущий `src/em3d/gamma0.py` уже содержит базовые функции для convex hull и окружностей, но реализация шага A упрощена до окружности с центром в середине отрезка. Это не совпадает с notebook и wiki: для пары точек используется специальная формула `mu_2points/radius_2points`, минимизирующая угол видимости из начала координат. Также сейчас нет API, который строит грубую блочно-развёрнутую матрицу оператора по `Problem` и возвращает полный объект анализа для экспериментов.

FFT-embedding не используется для спектра: диагональные значения расширенного циркулянтного оператора не совпадают со спектром исходной конечной блочно-тёплицевой матрицы `H = I - B·eta` на области `Q`. FFT остаётся способом быстрого `matvec`, а для настройки `gamma0` в этой фиче используется dense-матрица на грубой сетке.

## API

В `src/em3d/gamma0.py`:

- `Gamma0Analysis`: dataclass с полями `mu`, `radius`, `rho`, `spectrum`, `hull`, `coarse_N`, `matrix_shape`.
- `analyze_spectrum(spectrum_samples, *, coarse_N=None, matrix_shape=None) -> Gamma0Analysis`.
- `find_params(spectrum_samples) -> dict`: обратная совместимость для `SolverConfig(**find_params(...))`.
- `coarse_operator_matrix(problem, coarse_N=(4, 4, 4)) -> np.ndarray`: строит dense-матрицу исходного оператора на грубой сетке.
- `estimate_from_problem(problem, coarse_N=(4, 4, 4)) -> Gamma0Analysis`.
- `find_params_from_problem(problem, coarse_N=(4, 4, 4)) -> dict`.

В `src/em3d/vis.py`:

- `plot_gamma0_spectrum(analysis, *, title=None, filename=None) -> (fig, ax)`.

## Алгоритм

1. Нормализовать спектр в `np.complex128`.
2. Построить convex hull в координатах `(real(lambda), imag(lambda))`.
3. Шаг A: перебрать пары вершин hull, построить окружности через `mu_2points/radius_2points`, проверить включение hull и отсутствие начала координат.
4. Шаг B: перебрать тройки вершин hull, построить описанные окружности, проверить включение hull и отсутствие начала координат.
5. Среди всех допустимых кандидатов выбрать минимум `rho = radius / abs(mu)`.
6. Для `Problem` строить coarse grid с теми же `L`, `center`, `k0`, nearest-neighbor ресэмплингом `eps_tensor`; затем собрать `H = I - B_dense @ Eta_dense` и считать `np.linalg.eigvals(H)`.

## Визуализация

`plot_gamma0_spectrum` рисует scatter спектра, пунктирную замкнутую выпуклую оболочку, центр `mu`, начало координат и окружность радиуса `radius`. Оси имеют `aspect='equal'`, подписи `Re(lambda)` и `Im(lambda)`, сетку и legend.

## Тестирование

- Проверить notebook-формулу для окружности по двум точкам.
- Проверить `analyze_spectrum` на известном спектре: hull содержит внешние точки, `rho == radius / abs(mu)`, круг содержит spectrum и не содержит ноль.
- Проверить `coarse_operator_matrix` на toy `Problem`: размер `(3*Nx*Ny*Nz, 3*Nx*Ny*Nz)` и совпадение с full dense matrix при одинаковой грубой сетке.
- Проверить `estimate_from_problem` на `coarse_N=(2,2,2)`: возвращает спектр правильной длины и параметры, пригодные для `SolverConfig`.
- Проверить `plot_gamma0_spectrum`: возвращает `Figure/Axes`, aspect equal, содержит scatter, hull line и circle patch.
