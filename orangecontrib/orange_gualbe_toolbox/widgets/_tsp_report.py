# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Contenido del informe de Test & Score Pro y su versión HTML.

`build_blocks` describe el informe como una lista de bloques; HTML y PDF (ver _tsp_pdf.py)
los dibujan cada uno a su manera, así que dicen exactamente lo mismo.
"""
import base64
import html
import importlib.metadata
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

from Orange.modelling.base import Fitter

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st
from orangecontrib.orange_gualbe_toolbox.widgets._tsp_engine import metric_label

EXPLANATIONS = {
    st.TTEST_CORRECTED: (
        "Compara las diferencias entre particiones ampliando la varianza con el factor de Nadeau y "
        "Bengio (1/J + n_test/n_train), porque las particiones comparten datos de entrenamiento. "
        "Un valor p ajustado menor que α indica evidencia de diferencia."),
    st.TTEST_PAIRED: (
        "t clásico, solo como referencia: supone particiones independientes, cosa que no son. Su "
        "tasa de falsos positivos es muy superior a α."),
    st.WILCOXON: (
        "Test no paramétrico de rangos con signo sobre las diferencias entre particiones. No "
        "supone normalidad, pero sí independencia, que las particiones no cumplen: es orientativo."),
    st.MCNEMAR: (
        "Compara dos clasificadores por las filas en que solo acierta uno. Cada fila debe haberse "
        "predicho una vez. Usa el binomial exacto si hay menos de 25 filas discordantes."),
    st.BAYES: (
        "Calcula la probabilidad a posteriori de que un modelo sea mejor, de que sean equivalentes "
        "(diferencia dentro del ROPE) y de que lo sea el otro, considerando la correlación entre "
        "particiones. Se decide con una probabilidad de al menos 0,95; si no, el resultado es "
        "indeciso. Con la ventana deslizante no hay una correlación conocida entre iteraciones: se "
        "aplica sin correlación (ρ = 0) y es orientativo."),
    st.NEMENYI: (
        "Tras Friedman, considera distintos dos modelos si sus rangos medios difieren más que la "
        "diferencia crítica. Se concibió para varios conjuntos de datos."),
    st.HOLM_POSTHOC: (
        "Tras Friedman, compara los rangos con un z y ajusta con Holm. Es más potente que Nemenyi "
        "y comparte su limitación con las particiones."),
    st.ROW_TTEST: (
        "Con un solo conjunto de test, compara por filas el error absoluto. Las filas de test son "
        "independientes si proceden de un conjunto separado."),
    st.ROW_WILCOXON: (
        "Versión no paramétrica del anterior: rangos con signo del error absoluto por fila."),
    st.DIEBOLD_MARIANO: (
        "Compara la pérdida de dos modelos instancia a instancia en el test de todas las iteraciones, en "
        "orden temporal. La varianza tiene en cuenta la autocorrelación hasta el retardo h − 1, siendo h el "
        "horizonte, y se corrige para muestras pequeñas (Harvey, Leybourne y Newbold). Es el test "
        "principal de la ventana deslizante."),
}

MAX_ITERATION_MODELS = 6

COPYRIGHT = "orange_gualbe_toolbox © 2026 Gualberto Asencio Cortés · GPL-3.0-or-later"
SUMMARY_MODELS = 10
MAX_LISTED_PAIRS = 6


# ------------------------------------------------------------------ bloques ----

@dataclass
class Title:
    text: str
    subtitle: str


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Paragraph:
    text: str
    small: bool = False
    keep_next: bool = False      # va con lo que le sigue (introducción)
    keep_previous: bool = False  # va con lo anterior (pie de tabla)
    bullet: bool = False


@dataclass
class Note:
    text: str


@dataclass
class Result:
    lines: List[str]


@dataclass
class TableBlock:
    headers: List[str]
    rows: List[List[str]]
    numeric: tuple = ()
    best: Optional[int] = None
    key_value: bool = False
    label: str = ""


@dataclass
class FigureSpec:
    title: str
    make: Callable


@dataclass
class Figures:
    specs: List[FigureSpec]
    wide: bool = False
    full: bool = False  # una figura a todo el ancho y media página de alto


@dataclass
class Summary:
    facts: str
    table: TableBlock
    figure: FigureSpec
    total_models: int = 0


@dataclass
class PageBreak:
    pass


@dataclass
class Document:
    title: str
    footer: str
    blocks: list = field(default_factory=list)


# ------------------------------------------------------------------ formato ----

def _esc(value):
    return html.escape(str(value))


def format_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else str(value)


def format_p(value):
    if value is None or not np.isfinite(value):
        return "—"
    return "&lt; 0,001" if value < 0.001 else f"{value:.4f}".replace(".", ",")


MINUS = "\u2212"
SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def number(value, decimals=4):
    """Un solo formato en todo el informe: coma decimal y notación científica uniforme."""
    if value is None or not np.isfinite(value):
        return "—"
    size = abs(value)
    if size != 0 and (size >= 1e6 or size < 1e-3):
        mantissa, exponent = f"{value:.2e}".split("e")
        text = mantissa.replace(".", ",") + "·10" + str(int(exponent)).translate(SUPERSCRIPTS)
    else:
        text = f"{value:.{decimals}f}".replace(".", ",")
    return text.replace("-", MINUS, 1) if text.startswith("-") else text


def probability(value):
    if value is None or not np.isfinite(value):
        return "—"
    return "< 0,001" if value < 0.001 else f"{value:.4f}".replace(".", ",")


def _localise(text, names):
    """Coma decimal en un texto del motor, sin tocar los nombres de los modelos."""
    if names:
        pattern = "(" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + ")"
        pieces = re.split(pattern, text)
    else:
        pieces = [text]
    return "".join(piece if index % 2 else re.sub(r"(?<=\d)\.(?=\d)", ",", piece)
                   for index, piece in enumerate(pieces))


_INTEGER = re.compile(r"(?<![\d.,])\d+(?![\d]|[.,]\d)")


def grouped_warnings(pairs, names, seen):
    """Agrupa los avisos repetidos de un test sin perder qué dicen ni a quién afectan."""
    groups = {}
    for pair in pairs:
        for line in pair.warnings:
            found = _INTEGER.findall(line)
            key = _INTEGER.sub("N", line)
            entry = groups.setdefault(key, dict(values=[], pairs=[], line=line))
            entry["values"].extend(int(item) for item in found[:1])
            entry["pairs"].append(f"{names[pair.a]} – {names[pair.b]}")
    out = []
    for key, entry in groups.items():
        distinct = sorted(set(entry["values"]))
        text = entry["line"] if len(distinct) <= 1 else key
        if text in seen:
            continue
        seen.add(text)
        count = len(entry["pairs"])
        detail = ""
        if len(distinct) > 1:
            detail = f" N va de {distinct[0]} a {distinct[-1]}."
        if count == 1:
            who = f" Afecta a: {entry['pairs'][0]}."
        elif count <= MAX_LISTED_PAIRS:
            who = f" Afecta a {count} comparaciones: {'; '.join(entry['pairs'])}."
        else:
            who = (f" Afecta a {count} comparaciones; el detalle por pareja está en la salida "
                   "Pairwise Comparisons.")
        out.append(_localise(text, names) + detail + who)
    return out


def versions():
    out = [("Python", sys.version.split()[0]), ("Sistema", platform.platform())]
    for package in ("Orange3", "orange_gualbe_toolbox", "numpy", "scipy", "scikit-learn", "baycomp",
                    "matplotlib", "pandas"):
        try:
            out.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            out.append((package, "desconocida"))
    out.append(("Autoría del add-on", COPYRIGHT))
    return out


def figure_specs(report, scoreset):
    """Gráficos del informe, dibujables al tamaño que necesite cada formato."""
    theme = plots.REPORT
    specs = [FigureSpec("Comparación de modelos",
                        lambda w, h: plots.scores_figure(report, scoreset, w, h, theme))]
    if report.friedman is not None and report.friedman.n_models >= 3:
        specs.append(FigureSpec("Diagrama de diferencias críticas",
                                lambda w, h: plots.critical_difference_figure(report, w, h, theme)))
    if report.primary:
        specs.append(FigureSpec(f"Matriz de comparaciones: {st.TEST_NAMES[report.primary]}",
                                lambda w, h: plots.matrix_figure(report, report.primary, w, h, theme)))
    if report.pairs_of(st.BAYES):
        specs.append(FigureSpec("Probabilidades bayesianas",
                                lambda w, h: plots.bayes_figure(report, w, h, theme)))
    specs.append(FigureSpec("Tiempos", lambda w, h: plots.times_figure(report, scoreset, w, h, theme)))
    return specs


def report_figures(report, scoreset):
    """Figuras para el informe nativo de Orange."""
    return [(spec.title, spec.make(8.6, 3.4)) for spec in figure_specs(report, scoreset)]


def learner_parameters(learner, domain=None):
    """Parámetros legibles de un modelo. Nunca impide generar el informe.

    Los modelos que sirven para clasificación y regresión llegan como `Fitter`, cuyo
    atributo `params` lanza una excepción: sus parámetros dependen del tipo de problema.
    """
    try:
        params = learner.get_params(domain) if isinstance(learner, Fitter) else learner.params
    except Exception:  # pylint: disable=broad-except
        return "—"
    if not isinstance(params, dict):
        return "—"
    clean = {key: value for key, value in params.items()
             if key not in ("self", "preprocessors") and not key.startswith("_")}
    return ", ".join(f"{key}={value!r}" for key, value in sorted(clean.items())) or "—"


def _summary_table(report, scoreset):
    label = report.metric_label
    interval = f"IC {1 - report.alpha:.0%}".replace(".", ",")
    rows = []
    shown = list(report.order)[:SUMMARY_MODELS]
    for model in shown:
        low, high = report.ci_low[model], report.ci_high[model]
        rows.append([report.names[model], number(report.values[model]),
                     f"{number(low)} – {number(high)}" if np.isfinite(low) else "—",
                     number(report.mean_ranks[model], 2),
                     number(scoreset.train_time[model] + scoreset.test_time[model], 3)])
    return TableBlock(["Modelo", label, interval, "Rango medio", "Tiempo (s)"], rows, numeric=(1, 2, 3, 4), best=0,
                      label="Resumen")


def _setup_facts(context, report):
    setup = dict(context["setup"])
    return " · ".join(str(item) for item in (
        setup.get("Conjunto de datos"), f"{setup.get('Filas')} filas", setup.get("Tipo de problema"),
        setup.get("Muestreo"), f"métrica {report.metric_label}",
        _localise(f"α = {report.alpha:g}", ())))


def _scores_table(report, scoreset):
    labels = [metric_label(scorer) for scorer in scoreset.scorers]
    interval = f"IC {1 - report.alpha:.0%} ({report.metric_label})"
    headers = ["Modelo"] + labels + [interval, "Rango medio", "Entrenamiento (s)", "Predicción (s)"]
    rows = []
    for model in report.order:
        row = [report.names[model]]
        row += [number(scoreset.overall[s.__name__][model]) for s in scoreset.scorers]
        low, high = report.ci_low[model], report.ci_high[model]
        row.append(f"{number(low)} – {number(high)}" if np.isfinite(low) else "—")
        row += [number(report.mean_ranks[model], 2), number(scoreset.train_time[model], 3),
                number(scoreset.test_time[model], 3)]
        rows.append(row)
    return TableBlock(headers, rows, numeric=tuple(range(1, len(headers))), best=0, label="Puntuaciones")


def _test_table(test, pairs, names):
    if test == st.BAYES:
        headers = ["A", "B", "Diferencia (A − B)", "P(A mejor)", "P(equivalentes)", "P(B mejor)", "Decisión"]
        rows = [[names[p.a], names[p.b], number(p.mean_diff), number(p.p_a, 3), number(p.p_rope, 3),
                 number(p.p_b, 3), p.decision] for p in pairs]
        return TableBlock(headers, rows, numeric=(2, 3, 4, 5), label=st.TEST_NAMES[test])
    effect_name = next((p.effect_name for p in pairs if p.effect_name), "")
    effect_header = effect_name or "Efecto"
    headers = ["A", "B", "Estadístico", "p", "p ajustado", "Diferencia (A − B)", "IC", effect_header, "Magnitud", "Decisión"]
    rows = []
    for p in pairs:
        interval = f"{number(p.ci_low)} – {number(p.ci_high)}" if np.isfinite(p.ci_low) else "—"
        rows.append([names[p.a], names[p.b], number(p.statistic), probability(p.p_value), probability(p.p_adjusted),
                     number(p.mean_diff), interval, number(p.effect, 3), p.effect_label or "—", p.decision])
    return TableBlock(headers, rows, numeric=(2, 3, 4, 5, 6, 7), label=st.TEST_NAMES[test])


def build_blocks(context, scoreset, report):
    """Describe el informe completo."""
    names = report.names
    specs = figure_specs(report, scoreset)
    generated = f"{datetime.now():%Y-%m-%d %H:%M}"
    document = Document(context["title"], f"Test & Score Pro · {dict(context['setup']).get('Conjunto de datos')} · {generated}")
    blocks = document.blocks
    blocks.append(Title(context["title"], f"Generado el {generated} con Test & Score Pro"))
    blocks.append(Result([_localise(line, names) for line in report.recommendation]))
    blocks.append(Summary(_setup_facts(context, report), _summary_table(report, scoreset), specs[0], len(names)))
    blocks.append(PageBreak())

    blocks.append(Heading(2, "Datos y muestreo"))
    blocks.append(TableBlock(["Concepto", "Valor"], [[str(k), _localise(str(v), names)] for k, v in context["setup"]],
                             key_value=True, label="Datos y muestreo"))
    if context.get("failures"):
        blocks.append(Heading(3, "Modelos que han fallado"))
        blocks.extend(Note(line) for line in context["failures"])
    blocks.extend(Note(_localise(line, names)) for line in (context.get("notes") or []))

    if context.get("window"):
        _window_blocks(blocks, context["window"], scoreset, report)

    blocks.append(Heading(2, "Modelos"))
    blocks.append(TableBlock(["Modelo", "Parámetros"], [[str(n), str(p)] for n, p in context["models"]], label="Modelos"))

    blocks.append(Heading(2, "Reproducibilidad"))
    blocks.append(TableBlock(["Software", "Versión"], [[k, v] for k, v in versions()], key_value=True,
                             label="Reproducibilidad"))

    blocks.append(Heading(2, "Puntuaciones"))
    blocks.append(_scores_table(report, scoreset))
    for scorer in scoreset.scorers:
        if scoreset.undefined.get(scorer.__name__):
            blocks.append(Paragraph(f"{metric_label(scorer)}: no definida en "
                                    f"{scoreset.undefined[scorer.__name__]} particiones.", small=True))

    blocks.append(Heading(2, "Gráficos"))
    blocks.append(Figures(specs, wide=len(names) > 8))

    blocks.append(Heading(2, "Comparaciones estadísticas"))
    correction = st.CORRECTIONS.get(report.correction, report.correction)
    blocks.append(Paragraph(_localise(f"Métrica: {report.metric_label} · α = {report.alpha:g} · corrección múltiple: "
                                      f"{correction} · ROPE = ±{report.rope:.4g}", names), keep_next=True))
    seen = set()
    if report.friedman is not None and report.friedman.n_blocks:
        fried = report.friedman
        blocks.append(Heading(3, "Friedman e Iman-Davenport"))
        blocks.append(TableBlock(["Estadístico", "Valor", "p"],
                                 [["χ² de Friedman", number(fried.chi2), probability(fried.p_value)],
                                  ["F de Iman-Davenport", number(fried.iman_davenport), probability(fried.p_iman_davenport)],
                                  ["Diferencia crítica (Nemenyi)", number(fried.critical_difference), "—"]],
                                 numeric=(1, 2), label="Friedman"))
        blocks.append(Paragraph(f"Supuesto: {_localise(fried.assumption, names)}", small=True, keep_previous=True))
        for line in fried.warnings:
            if line not in seen:
                seen.add(line)
                blocks.append(Note(_localise(line, names)))
    for test in report.available:
        pairs = report.pairs_of(test)
        blocks.append(Heading(3, st.TEST_NAMES[test]))
        blocks.append(Paragraph(EXPLANATIONS[test], keep_next=True))
        blocks.append(_test_table(test, pairs, names))
        if pairs:
            blocks.append(Paragraph(f"Aporta evidencia sobre: {pairs[0].evidence}. La diferencia es A − B, en "
                                    f"{pairs[0].units}: en métricas de error y en rangos, un valor negativo significa "
                                    "que A es mejor.", small=True, keep_previous=True))
        assumption = next((p.assumption for p in pairs if p.assumption), "")
        if assumption:
            blocks.append(Paragraph(f"Supuesto: {_localise(assumption, names)}", small=True, keep_previous=True))
        if test == report.primary:
            blocks.append(Heading(4, "Interpretación"))
            blocks.extend(Paragraph(_localise(st.interpret(p, names, report.metric_label, report.alpha), names),
                                    small=True, bullet=True) for p in pairs)
        blocks.extend(Note(line) for line in grouped_warnings(pairs, names, seen))
    return document


def _window_blocks(blocks, window, scoreset, report):
    """Configuración, vista de las ventanas, resultados por iteración y por horizonte."""
    theme = plots.REPORT
    plan, names = window["plan"], report.names
    blocks.append(Heading(2, "Ventana deslizante"))
    blocks.append(TableBlock(["Opción", "Valor"], [[str(k), str(v)] for k, v in window["config"]],
                             key_value=True, label="Ventana deslizante"))
    blocks.append(Figures([FigureSpec("Ventanas de entrenamiento y test",
                                      lambda w, h: plots.windows_figure(plan, w, h, theme))], full=True))
    folds = scoreset.folds.get(report.metric) if scoreset.has_folds else None
    shown = list(report.order)[:MAX_ITERATION_MODELS]
    headers = ["Iteración", "Entrenamiento", "Filas", "Test", "Filas test"]
    if folds is not None:
        headers += [names[m] for m in shown]
    rows = []
    sizes = scoreset.fold_sizes
    for position, it in enumerate(plan.iterations):
        row = [str(it.index + 1), plan.span(it.train_lo, it.train_hi), str(it.n_train),
               plan.span(it.test_lo, it.test_hi) + ("" if it.complete else " (incompleto)"),
               str(int(sizes[position]) if sizes is not None and position < len(sizes) else it.n_test)]
        if folds is not None:
            row += [number(folds[m, position]) if position < folds.shape[1] else "—" for m in shown]
        rows.append(row)
    target = window.get("target")
    blocks.append(Heading(3, "Iteraciones"))
    intro = "Fechas de cada ventana"
    if folds is not None:
        intro += f" y {report.metric_label} de cada modelo en su test"
        if len(names) > len(shown):
            intro += f" (los {len(shown)} mejores de {len(names)} modelos)"
    if target:
        intro += f", para el objetivo «{target}»"
    blocks.append(Paragraph(_localise(intro + ".", names), keep_next=True))
    blocks.append(TableBlock(headers, rows, numeric=tuple(range(5, len(headers))) + (2, 4), label="Iteraciones"))
    if folds is not None:
        blocks.append(Figures([FigureSpec(f"Evolución de {report.metric_label} por iteración",
                                          lambda w, h: plots.iterations_figure(report, scoreset, plan, w, h, theme))],
                              full=True))
    targets = window.get("target_names") or []
    values = window.get("horizon_values")
    if len(targets) > 1 and values is not None:
        blocks.append(Heading(2, "Horizontes"))
        blocks.append(Paragraph(_localise(
            f"{report.metric_label} global de cada modelo con cada variable objetivo. Cada objetivo se entrena "
            "y evalúa por separado, con las mismas ventanas; la posición del objetivo es su horizonte.",
            names), keep_next=True))
        best = (np.nanargmin if report.lower_better else np.nanargmax)
        rows = []
        for model in report.order:
            rows.append([names[model]] + [number(values[model, t]) for t in range(len(targets))])
        table = TableBlock(["Modelo"] + list(targets), rows, numeric=tuple(range(1, len(targets) + 1)), best=0,
                           label="Horizontes")
        blocks.append(table)
        finite = np.isfinite(values).any(axis=0)
        winners = [f"{targets[t]}: {names[int(best(values[:, t]))]}" for t in range(len(targets)) if finite[t]]
        if winners:
            blocks.append(Paragraph("Mejor modelo por objetivo: " + "; ".join(winners) + ".", small=True,
                                    keep_previous=True))
        blocks.append(Figures([FigureSpec("Métrica por horizonte",
                                          lambda w, h: plots.horizons_figure(
                                              names, targets, values, report.metric_label, report.lower_better,
                                              w, h, theme))], full=True))


# ------------------------------------------------------------------- HTML ----

STYLE = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1b1b19;
margin:0 auto;max-width:1040px;padding:24px;line-height:1.5;background:#fff}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 10px;border-bottom:2px solid #2a78d6;
padding-bottom:4px}h3{font-size:15px;margin:22px 0 6px}h4{font-size:13px;margin:12px 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid #e5e4df;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f4f3ef}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{font-weight:600;background:#e8f1fb}
.note{background:#fff6e5;border-left:4px solid #eda100;padding:8px 12px;margin:8px 0;font-size:13px}
.rec{background:#e8f1fb;border-left:4px solid #2a78d6;padding:12px 16px;margin:14px 0}
.small{color:#52514e;font-size:12px}img{max-width:100%;height:auto}ul.small{margin:4px 0;padding-left:20px}
.facts{color:#52514e;margin:8px 0}
"""


def _html_table(block):
    head = "".join(f"<th>{_esc(text)}</th>" for text in block.headers)
    body = []
    for index, row in enumerate(block.rows):
        cells = "".join(f'<td class="n">{_esc(cell)}</td>' if column in block.numeric else f"<td>{_esc(cell)}</td>"
                        for column, cell in enumerate(row))
        body.append(f"<tr{' class=best' if index == block.best else ''}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _html_figure(spec, width=8.6, height=4.0):
    data = base64.b64encode(plots.figure_png(spec.make(width, height), 130)).decode("ascii")
    return f"<h3>{_esc(spec.title)}</h3><img alt='{_esc(spec.title)}' src='data:image/png;base64,{data}'>"


def render_html(document):
    parts = []
    in_list = False
    for block in document.blocks:
        if in_list and not (isinstance(block, Paragraph) and block.bullet):
            parts.append("</ul>")
            in_list = False
        if isinstance(block, Title):
            parts.append(f"<h1>{_esc(block.text)}</h1><p class='small'>{_esc(block.subtitle)}</p>")
        elif isinstance(block, Result):
            parts.append("<div class='rec'>" + "<br>".join(_esc(line) for line in block.lines) + "</div>")
        elif isinstance(block, Summary):
            parts.append(f"<p class='facts'>{_esc(block.facts)}</p>")
            parts.append(_html_table(block.table))
            if len(block.table.rows) < block.total_models:
                parts.append(f"<p class='small'>Se muestran los {len(block.table.rows)} mejores de "
                             f"{block.total_models} modelos; la tabla completa está en Puntuaciones.</p>")
        elif isinstance(block, Heading):
            parts.append(f"<h{block.level}>{_esc(block.text)}</h{block.level}>")
        elif isinstance(block, Paragraph):
            if block.bullet:
                if not in_list:
                    parts.append("<ul class='small'>")
                    in_list = True
                parts.append(f"<li>{_esc(block.text)}</li>")
            else:
                parts.append(f"<p{' class=small' if block.small else ''}>{_esc(block.text)}</p>")
        elif isinstance(block, Note):
            parts.append(f"<div class='note'>{_esc(block.text)}</div>")
        elif isinstance(block, TableBlock):
            parts.append(_html_table(block))
        elif isinstance(block, Figures):
            parts.extend(_html_figure(spec) for spec in block.specs)
    if in_list:
        parts.append("</ul>")
    body = "\n".join(parts)
    return (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>{_esc(document.title)}"
            f"</title><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<style>{STYLE}</style></head><body>{body}</body></html>")


def build_html(context, scoreset, report):
    return render_html(build_blocks(context, scoreset, report))
