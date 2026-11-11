# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Gráficos de Test & Score Pro. Devuelven figuras de matplotlib sin pasar por pyplot.

El widget y el informe usan exactamente las mismas figuras.
"""
import io
import textwrap
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st


@dataclass
class Theme:
    background: str = "#ffffff"
    foreground: str = "#0b0b0b"
    muted: str = "#52514e"
    grid: str = "#e5e4df"
    accent: str = "#2a78d6"
    opposite: str = "#eb6834"
    neutral: str = "#b9b8b0"
    comma: bool = False  # coma decimal, para el informe


CAPTION_POINTS = 8.5
CAPTION_WIDTH_FACTOR = 0.56  # ancho medio de un carácter respecto al tamaño de letra

LIGHT = Theme()
DARK = Theme("#262624", "#f0efec", "#c3c2b7", "#3b3b38", "#3987e5", "#d95926", "#6f6e68")
REPORT = Theme(comma=True)


def _decimal(text, theme):
    return text.replace(".", ",") if theme.comma else text


def _comma_ticks(axis, theme):
    if theme.comma:
        axis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}".replace(".", ",")))


def _short_name(name, width_inches):
    """Acorta por el centro: los nombres largos no deben aplastar el área del gráfico."""
    limit = max(int(width_inches * 4.2), 16)
    if len(name) <= limit:
        return name
    tail = max(limit // 3, 6)
    return name[:limit - tail - 1].rstrip() + "…" + name[-tail:].lstrip()


def _wrapped(text, width_inches, points=10.0, share=0.72):
    """Ajusta un texto al ancho aproximado de los ejes para que no se recorte."""
    return "\n".join(textwrap.wrap(text, max(int(width_inches * share * 72 / (points * 0.53)), 20)))


def new_figure(width, height, theme, dpi=100):
    figure = Figure(figsize=(max(width, 3.0), max(height, 2.0)), dpi=dpi, layout="constrained",
                    facecolor=theme.background)
    FigureCanvasAgg(figure)
    return figure


def style_axes(ax, theme):
    ax.set_facecolor(theme.background)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, length=0)
    ax.xaxis.label.set_color(theme.muted)
    ax.yaxis.label.set_color(theme.muted)
    ax.title.set_color(theme.foreground)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)


def message_figure(text, width, height, theme):
    figure = new_figure(width, height, theme)
    figure.text(0.5, 0.5, text, ha="center", va="center", color=theme.muted, wrap=True, fontsize=11)
    return figure


def row_height(count):
    return min(0.55, max(0.24, 3.6 / max(count, 1)))


def scores_figure(report, scoreset, width, height, theme):
    """Distribución por particiones de cada modelo, con su media e intervalo."""
    if report.best < 0:
        return message_figure("No hay puntuaciones que mostrar.", width, height, theme)
    order = [i for i in report.order if np.isfinite(report.values[i])][::-1]
    figure = new_figure(width, max(height, 1.0 + 0.24 * len(order)), theme)
    ax = figure.add_subplot()
    style_axes(ax, theme)
    _comma_ticks(ax.xaxis, theme)
    folds = scoreset.folds.get(report.metric) if scoreset.has_folds else None
    for y, model in enumerate(order):
        color = theme.accent if model == report.best else theme.neutral
        if folds is not None:
            values = folds[model][np.isfinite(folds[model])]
            if len(values):
                jitter = np.linspace(-0.22, 0.22, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(values, y + jitter, s=14, color=color, alpha=0.55, linewidths=0, zorder=2)
        low, high = report.ci_low[model], report.ci_high[model]
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [y, y], color=theme.foreground, linewidth=2, zorder=3,
                    solid_capstyle="round")
        ax.scatter([report.values[model]], [y], s=64, marker="D", color=color, zorder=4,
                   edgecolors=theme.background, linewidths=1.5)
    ax.set_yticks(range(len(order)), [_short_name(report.names[i], width) for i in order], color=theme.foreground)
    ax.set_ylim(-0.7, len(order) - 0.3)
    interval = "" if not np.isfinite(report.ci_low).any() else f", IC {1 - report.alpha:.0%}"
    points = f" · puntos: {getattr(report, 'blocks', 'particiones')}" if folds is not None else ""
    ax.set_xlabel(_wrapped(_decimal(f"{report.metric_label} (rombo: valor global{interval}{points})", theme), width))
    return figure


def _add_caption(figure, paragraphs, theme):
    """Pie de figura ajustado al ancho, con su espacio reservado bajo los ejes."""
    width, height = figure.get_size_inches()
    per_line = max(int(width * 72 / (CAPTION_POINTS * CAPTION_WIDTH_FACTOR)), 20)
    lines = [line for paragraph in paragraphs for line in textwrap.wrap(paragraph, per_line)]
    reserved = (len(lines) * CAPTION_POINTS * 1.3 + 8) / 72 / height
    figure.get_layout_engine().set(rect=(0, reserved, 1, 1 - reserved))
    figure.text(0.5, 0.008, "\n".join(lines), ha="center", va="bottom", fontsize=CAPTION_POINTS,
                color=theme.muted, linespacing=1.3)


def critical_difference_figure(report, width, height, theme):
    """Diagrama de diferencias críticas de Demšar."""
    friedman = report.friedman
    if friedman is None or friedman.n_models < 3 or not np.isfinite(friedman.critical_difference):
        return message_figure("El diagrama necesita al menos 3 modelos y particiones comparables.",
                              width, height, theme)
    ranks = friedman.mean_ranks
    k = len(ranks)
    order = np.argsort(ranks)
    half = (k + 1) // 2
    figure = new_figure(width, max(height, 2.0 + 0.3 * half), theme)
    ax = figure.add_subplot()
    ax.set_facecolor(theme.background)
    ax.axis("off")
    top = 0.0
    ax.set_xlim(0.2 - (k - 1) * 0.35, k - 0.2 + (k - 1) * 0.35)
    label_step = 1.0
    ax.set_ylim(-(half + 1.9) * label_step, 1.6)
    ax.plot([1, k], [top, top], color=theme.foreground, linewidth=1.4)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [top, top + 0.22], color=theme.foreground, linewidth=1.2)
        ax.text(tick, top + 0.32, str(tick), ha="center", va="bottom", color=theme.muted, fontsize=9)
    distance = friedman.critical_difference
    ax.plot([1, 1 + distance], [1.25, 1.25], color=theme.foreground, linewidth=2.2)
    ax.text(1 + distance / 2, 1.33, _decimal(f"CD = {distance:.2f}", theme), ha="center", va="bottom",
            color=theme.foreground, fontsize=9)
    for position, model in enumerate(order):
        left = position < half
        level = -(position + 1) if left else -(k - position)
        x_end = 1 - (k - 1) * 0.08 if left else k + (k - 1) * 0.08
        color = theme.accent if model == report.best else theme.foreground
        ax.plot([ranks[model], ranks[model], x_end], [top, level, level], color=color, linewidth=1.1)
        ax.text(x_end + (-0.08 if left else 0.08), level,
                f"{_short_name(report.names[model], width * 0.75)} ({_decimal(f'{ranks[model]:.2f}', theme)})",
                ha="right" if left else "left", va="center", color=color, fontsize=9)
    groups = []
    for start in range(k):
        end = start
        while end + 1 < k and ranks[order[end + 1]] - ranks[order[start]] <= distance:
            end += 1
        if end > start and not any(g[0] <= start and end <= g[1] for g in groups):
            groups.append((start, end))
    for level, (start, end) in enumerate(groups):
        y = -0.32 - 0.14 * level
        ax.plot([ranks[order[start]] - 0.04, ranks[order[end]] + 0.04], [y, y],
                color=theme.muted, linewidth=3.5, solid_capstyle="round")
    caption = ["Rango medio (1 es el mejor). Los modelos unidos por una barra no se distinguen "
               "según Nemenyi."]
    if friedman.on_folds:
        caption.append("Los bloques son particiones de un solo conjunto de datos, que no son "
                       "independientes: diagrama descriptivo, no prueba de generalización.")
    _add_caption(figure, caption, theme)
    return figure


def matrix_figure(report, test, width, height, theme):
    """Matriz de comparaciones: fila contra columna según el test elegido."""
    pairs = report.pairs_of(test)
    if not pairs:
        return message_figure("Este test no se aplica con el muestreo actual.", width, height, theme)
    k = len(report.names)
    figure = new_figure(width, max(height, 1.0 + 0.28 * k), theme)
    ax = figure.add_subplot()
    ax.set_facecolor(theme.background)
    order = list(report.order)
    position = {model: i for i, model in enumerate(order)}
    label_cells = k <= 16
    for pair in pairs:
        row, col = position[pair.a], position[pair.b]
        for r, c, mine, flipped in ((row, col, st.A_BETTER, False), (col, row, st.B_BETTER, True)):
            if test == st.BAYES:
                value = pair.p_b if flipped else pair.p_a
                wins = np.isfinite(value) and value >= st.BAYES_DECISION
                other = pair.p_a if flipped else pair.p_b
                loses = np.isfinite(other) and other >= st.BAYES_DECISION
                text = "—" if not np.isfinite(value) else _decimal(f"{value:.2f}", theme)
            else:
                wins = pair.decision == mine
                loses = pair.decision == (st.B_BETTER if mine == st.A_BETTER else st.A_BETTER)
                value = pair.p_adjusted
                text = "—" if not np.isfinite(value) else ("<0,001" if theme.comma else "<.001") if value < 0.001 else _decimal(f"{value:.3f}", theme)
            color = theme.accent if wins else theme.opposite if loses else theme.grid
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor=theme.background,
                                   linewidth=2, alpha=0.9 if wins or loses else 0.55))
            if label_cells:
                ax.text(c + 0.5, r + 0.5, text, ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if wins or loses else theme.foreground)
    for model in range(k):
        cell = position[model]
        ax.add_patch(Rectangle((cell, cell), 1, 1, facecolor=theme.background, edgecolor=theme.grid))
    names = [_short_name(report.names[i], width * 0.7) for i in order]
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    ax.set_xticks(np.arange(k) + 0.5, names, rotation=35, ha="right", color=theme.foreground, fontsize=9)
    ax.set_yticks(np.arange(k) + 0.5, names, color=theme.foreground, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ("P(fila mejor que columna)" if test == st.BAYES else "p ajustado") + (
        " · azul: la fila gana · naranja: la fila pierde · gris: sin evidencia")
    ax.set_title(_wrapped(f"{st.TEST_NAMES[test]} — {legend}", width, 9.5, 0.68), fontsize=9.5, color=theme.muted, loc="left")
    return figure


def bayes_figure(report, width, height, theme):
    """Probabilidades del mejor modelo frente a cada uno de los demás."""
    pairs = report.pairs_of(st.BAYES)
    if not pairs or report.best < 0:
        return message_figure("Las probabilidades bayesianas necesitan al menos 2 modelos y "
                              "varias particiones.", width, height, theme)
    rows = []
    for rival in report.order:
        if rival == report.best:
            continue
        for pair in pairs:
            if {pair.a, pair.b} == {report.best, rival} and np.isfinite(pair.p_a):
                flip = pair.a != report.best
                rows.append((rival, pair.p_b if flip else pair.p_a, pair.p_rope,
                             pair.p_a if flip else pair.p_b))
    if not rows:
        return message_figure("No se han podido calcular las probabilidades.", width, height, theme)
    rows = rows[::-1]
    figure = new_figure(width, max(height, 1.0 + 0.3 * len(rows)), theme)
    ax = figure.add_subplot()
    style_axes(ax, theme)
    ax.grid(False)
    for y, (rival, better, rope, worse) in enumerate(rows):
        left = 0.0
        for value, color in ((better, theme.accent), (rope, theme.neutral), (worse, theme.opposite)):
            ax.barh(y, max(value - 0.004, 0), left=left + 0.002, height=0.62, color=color)
            if value >= 0.09:
                text_color = "#ffffff" if color != theme.neutral else theme.foreground
                ax.text(left + value / 2, y, f"{value:.0%}", ha="center", va="center", fontsize=8.5,
                        color=text_color)
            left += value
    ax.set_yticks(range(len(rows)), [_short_name(report.names[r[0]], width) for r in rows], color=theme.foreground)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5), ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel(_wrapped(f"Azul: {_short_name(report.names[report.best], width)} mejor · gris: equivalentes "
                           f"(ROPE ±{_decimal(f'{report.rope:.3g}', theme)}) · naranja: rival mejor", width))
    return figure


def times_figure(report, scoreset, width, height, theme):
    order = list(report.order)[::-1]
    figure = new_figure(width, max(height, 1.0 + 0.26 * len(order)), theme)
    ax = figure.add_subplot()
    style_axes(ax, theme)
    _comma_ticks(ax.xaxis, theme)
    train, test = scoreset.train_time[order], scoreset.test_time[order]
    ax.barh(range(len(order)), train, height=0.6, color=theme.accent, label="Entrenamiento")
    ax.barh(range(len(order)), test, left=train, height=0.6, color=theme.opposite, label="Predicción")
    ax.set_yticks(range(len(order)), [_short_name(report.names[i], width) for i in order], color=theme.foreground)
    ax.set_xlabel("Segundos en total")
    legend = ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9,
                       borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(theme.foreground)
    return figure


# ------------------------------------------------------- ventana deslizante ----

SERIES = ("#2a78d6", "#eb6834", "#3a9e6a", "#8e5bd0", "#d6a32a", "#2ab0c0", "#c0398a", "#6f6e68")
MAX_DRAWN_ITERATIONS = 40


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _compact(value, theme):
    """Número corto y legible para una celda: 1290, 35,2, 0,0412, 2,1·10⁵."""
    size = abs(value)
    if size >= 1e5 or 0 < size < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return _decimal(mantissa, theme) + "·10" + str(int(exponent)).translate(SUPERSCRIPT)
    if size >= 100:
        return f"{value:.0f}"
    return _decimal(f"{value:.3g}", theme)


def _series_color(index, theme):
    colors = (theme.accent, theme.opposite) + SERIES[2:]
    return colors[index] if index < len(colors) else theme.neutral


def _window_axis(ax, plan, theme):
    """Eje x temporal (fechas) o de filas, con el formato del tema."""
    if plan.times is not None:
        import matplotlib.dates as mdates
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=True))
    else:
        _comma_ticks(ax.xaxis, theme)
        ax.set_xlabel("Fila (en el orden de los datos)")


def _edges(plan, lo, hi):
    """Bordes [x0, x1) en el eje de unas filas [lo, hi): fechas en días o números de fila."""
    if plan.times is None:
        return lo + 0.5, hi + 0.5
    spacing = plan.spacing if np.isfinite(plan.spacing) else 0.0
    left = plan.times[lo]
    right = plan.times[hi] if hi < plan.n_rows else plan.times[hi - 1] + spacing
    return left / 86400.0, right / 86400.0


def shown_iterations(count, limit=MAX_DRAWN_ITERATIONS):
    if count <= limit:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, limit).round().astype(int).tolist()))


def windows_figure(plan, width, height, theme, title=True):
    """Línea temporal de las ventanas: entrenamiento, hueco y test de cada iteración."""
    if plan is None or plan.n_rows == 0:
        return message_figure("Sin datos para formar ventanas.", width, height, theme)
    iterations = plan.iterations
    if not iterations:
        return message_figure(plan.error or "No hay ninguna iteración con estas opciones.", width, height, theme)
    shown = shown_iterations(len(iterations))
    figure = new_figure(width, max(height, 1.4 + 0.16 * len(shown)), theme)
    ax = figure.add_subplot()
    style_axes(ax, theme)
    bar = 0.72 if len(shown) <= 20 else 0.86
    for row, index in enumerate(shown):
        it = iterations[index]
        x0, x1 = _edges(plan, it.train_lo, it.train_hi)
        ax.barh(row, x1 - x0, left=x0, height=bar, color=theme.accent, linewidth=0)
        if it.test_lo > it.train_hi:
            g0, g1 = _edges(plan, it.train_hi, it.test_lo)
            ax.barh(row, g1 - g0, left=g0, height=bar, color=theme.neutral, linewidth=0, alpha=0.6)
        t0, t1 = _edges(plan, it.test_lo, it.test_hi)
        ax.barh(row, t1 - t0, left=t0, height=bar, color=theme.opposite, linewidth=0,
                alpha=1.0 if it.complete else 0.5, hatch=None if it.complete else "//")
    first, last = _edges(plan, 0, plan.n_rows)
    start, _ = _edges(plan, plan.start, plan.start + 1) if plan.start < plan.n_rows else (last, last)
    end, _ = _edges(plan, plan.end, plan.end + 1) if plan.end < plan.n_rows else (last, last)
    if plan.start > 0:
        ax.axvspan(first, start, color=theme.grid, alpha=0.7, linewidth=0, zorder=0)
    if plan.end < plan.n_rows:
        ax.axvspan(end, last, color=theme.grid, alpha=0.7, linewidth=0, zorder=0)
    ax.set_xlim(first, last)
    ax.set_ylim(len(shown) - 0.5, -0.5)
    step = max(1, len(shown) // 12)
    ticks = list(range(0, len(shown), step))
    ax.set_yticks(ticks, [str(shown[i] + 1) for i in ticks], color=theme.foreground, fontsize=8.5)
    ax.set_ylabel("Iteración")
    ax.grid(axis="x", color=theme.grid, linewidth=0.6)
    _window_axis(ax, plan, theme)
    from matplotlib.patches import Patch
    handles = [Patch(color=theme.accent, label="Entrenamiento"), Patch(color=theme.opposite, label="Test")]
    if any(it.test_lo > it.train_hi for it in iterations):
        handles.insert(1, Patch(color=theme.neutral, alpha=0.6, label="Hueco"))
    if plan.start > 0 or plan.end < plan.n_rows:
        handles.append(Patch(color=theme.grid, label="Fuera del estudio"))
    legend = ax.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                       ncol=len(handles), fontsize=8.5, borderaxespad=0.2, handlelength=1.2)
    for text in legend.get_texts():
        text.set_color(theme.foreground)
    if title:
        text = f"{len(iterations)} iteraciones"
        if len(shown) < len(iterations):
            text += f" (se dibujan {len(shown)})"
        ax.set_title(_decimal(text, theme), fontsize=9.5, color=theme.muted, loc="left")
    return figure


def iterations_figure(report, scoreset, plan, width, height, theme):
    """Métrica de cada modelo en cada iteración, a lo largo del tiempo."""
    folds = scoreset.folds.get(report.metric) if scoreset is not None and scoreset.has_folds else None
    if folds is None or plan is None or not plan.iterations:
        return message_figure("Hacen falta al menos 2 iteraciones para ver la evolución.", width, height, theme)
    figure = new_figure(width, height, theme)
    ax = figure.add_subplot()
    style_axes(ax, theme)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8)
    _comma_ticks(ax.yaxis, theme)
    iterations = plan.iterations[:folds.shape[1]]
    x = np.array([np.mean(_edges(plan, it.test_lo, it.test_hi)) for it in iterations])
    order = list(report.order)
    for rank, model in enumerate(order):
        color = _series_color(rank, theme)
        width_line = 2.2 if model == report.best else 1.3
        ax.plot(x, folds[model], color=color, linewidth=width_line, marker="o" if len(x) <= 30 else None,
                markersize=3.5, label=_short_name(report.names[model], width * 0.9), zorder=3 if rank == 0 else 2)
    _window_axis(ax, plan, theme)
    if plan.times is not None:
        ax.set_xlabel("Centro del test de cada iteración")
    else:
        ax.set_xlabel("Centro del test de cada iteración (fila)")
    ax.set_ylabel(report.metric_label)
    legend = ax.legend(frameon=False, fontsize=8.5, loc="best", ncol=1 if len(order) <= 6 else 2)
    for text in legend.get_texts():
        text.set_color(theme.foreground)
    return figure


def horizons_figure(names, target_names, values, metric, lower_better, width, height, theme):
    """Matriz modelo × objetivo y curva de la métrica según el horizonte."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return message_figure("No hay puntuaciones por objetivo.", width, height, theme)
    n_models, n_targets = values.shape
    figure = new_figure(width, max(height, 1.6 + 0.3 * n_models), theme)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.1, 1))
    ax = figure.add_subplot(grid[0])
    ax.set_facecolor(theme.background)
    best = np.nanargmin(np.where(np.isfinite(values), values, np.inf), axis=0) if lower_better else \
        np.nanargmax(np.where(np.isfinite(values), values, -np.inf), axis=0)
    for col in range(n_targets):
        column = values[:, col]
        finite = column[np.isfinite(column)]
        low, high = (finite.min(), finite.max()) if finite.size else (0, 1)
        for row in range(n_models):
            value = column[row]
            share = 0.0 if not np.isfinite(value) or high == low else (value - low) / (high - low)
            goodness = 1 - share if lower_better else share
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=theme.accent, alpha=0.12 + 0.6 * goodness,
                                   edgecolor=theme.background, linewidth=2))
            if n_models * n_targets <= 160:
                text = "—" if not np.isfinite(value) else _compact(value, theme)
                ax.text(col + 0.5, row + 0.5, text, ha="center", va="center", fontsize=8,
                        color=theme.foreground, fontweight="bold" if row == best[col] else "normal")
    ax.set_xlim(0, n_targets)
    ax.set_ylim(n_models, 0)
    short = [_short_name(name, width * 0.25) for name in target_names]
    ax.set_xticks(np.arange(n_targets) + 0.5, short, rotation=35 if n_targets > 3 else 0,
                  ha="right" if n_targets > 3 else "center", color=theme.foreground, fontsize=8.5)
    ax.set_yticks(np.arange(n_models) + 0.5, [_short_name(n, width * 0.45) for n in names],
                  color=theme.foreground, fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(_wrapped(f"{metric} por modelo y objetivo (más intenso: mejor)", width / 2, 9.5),
                 fontsize=9.5, color=theme.muted, loc="left")
    curve = figure.add_subplot(grid[1])
    style_axes(curve, theme)
    curve.grid(axis="y", color=theme.grid, linewidth=0.8)
    _comma_ticks(curve.yaxis, theme)
    horizons = np.arange(1, n_targets + 1)
    overall = np.nanmean(values, axis=1)
    ranking = np.argsort(np.where(np.isfinite(overall), overall if lower_better else -overall, np.inf))
    for rank, model in enumerate(ranking):
        curve.plot(horizons, values[model], color=_series_color(rank, theme), marker="o", markersize=4,
                   linewidth=2.0 if rank == 0 else 1.3, label=_short_name(names[model], width * 0.5))
    curve.set_xticks(horizons, [str(h) for h in horizons], fontsize=8.5)
    curve.set_xlabel("Horizonte (posición del objetivo)")
    curve.set_ylabel(metric)
    legend = curve.legend(frameon=False, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_color(theme.foreground)
    return figure


def figure_png(figure, dpi=160):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    return buffer.getvalue()


def save_figure(figure, path):
    figure.savefig(path, facecolor=figure.get_facecolor(), dpi=200)
