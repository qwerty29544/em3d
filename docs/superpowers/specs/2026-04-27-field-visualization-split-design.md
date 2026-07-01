# Field Visualization Split Design

## Цель

Развести визуализации скалярных и векторных полей, чтобы исследовательские графики не смешивали тепловую карту и стрелки в одном рисунке. Новые функции должны явно отвечать на один вопрос: либо показать скалярное распределение, либо показать направление/амплитуду вектора.

## Контекст

Сейчас `plot_field_slice` строит 2D-срез с фоном `||u||` и quiver-стрелками поверх него. Это полезно как обзорный рисунок, но перегружает график при сравнении серий расчётов. `plot_field_volume` уже является векторной 3D-визуализацией, но по имени не отделяет scalar/vector режимы.

## Новый API

В `src/em3d/vis.py` добавляются:

- `plot_field_scalar_slice(u, grid, *, plane="xy", idx=None, component=None, part="abs", stride=1, cmap="viridis", title=None, filename=None)`
- `plot_field_vector_slice(u, grid, *, plane="xy", idx=None, part="real", stride=1, cmap="RdBu_r", title=None, filename=None)`
- `plot_field_scalar_volume(u, grid, *, component=None, part="abs", stride=2, elev=30.0, azim=-60.0, cmap="viridis", title=None, filename=None)`
- `plot_field_vector_volume(u, grid, *, part="real", stride=2, elev=30.0, azim=-60.0, cmap="RdBu_r", title=None, filename=None)`

`component=None` означает норму трёх компонент после выбора `part`; `component=0/1/2` означает конкретную компоненту поля.

## Совместимость

Старые функции остаются:

- `plot_field_slice` сохраняет текущую combined-визуализацию: scalar background + vector quiver.
- `plot_field_volume` становится совместимой обёрткой над `plot_field_vector_volume`, потому что старый 3D-график уже был векторным.

README должен продвигать новые разделённые функции, а старые упоминать как compatibility helpers.

## Тестирование

- 2D scalar slice возвращает `Figure/Axes`, содержит pcolormesh и не содержит quiver.
- 2D vector slice возвращает `Figure/Axes`, содержит quiver и не содержит pcolormesh/colorbar.
- 3D scalar volume возвращает `Figure/Axes3D`, содержит scatter collection и не содержит quiver lines.
- 3D vector volume возвращает `Figure/Axes3D`, содержит 3D quiver.
- `component` валидируется: допустимы `None`, `0`, `1`, `2`.
- Старые тесты для `plot_field_slice` и `plot_field_volume` продолжают проходить.

