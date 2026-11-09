"""Informe autocontenido de Test & Score Pro (HTML con imágenes incrustadas, o PDF)."""
import base64
import html
import importlib.metadata
import platform
import sys
from datetime import datetime

import numpy as np

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
        "indeciso."),
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
}

STYLE = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1b1b19;
margin:0 auto;max-width:980px;padding:24px;line-height:1.5;background:#fff}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 10px;border-bottom:2px solid #2a78d6;
padding-bottom:4px}h3{font-size:15px;margin:20px 0 6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid #e5e4df;padding:5px 7px;text-align:left;vertical-align:top}
th{background:#f4f3ef}td.n{text-align:right;font-variant-numeric:tabular-nums}
tr.best td{font-weight:600;background:#e8f1fb}
.note{background:#fff6e5;border-left:4px solid #eda100;padding:8px 12px;margin:8px 0;font-size:13px}
.rec{background:#e8f1fb;border-left:4px solid #2a78d6;padding:10px 14px;margin:12px 0}
.small{color:#52514e;font-size:12px}img{max-width:100%;height:auto}
"""


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


def _rows(headers, rows, numeric=(), best=None):
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        cells = "".join(f'<td class="n">{cell}</td>' if column in numeric else f"<td>{cell}</td>"
                        for column, cell in enumerate(row))
        marker = ' class="best"' if index == best else ""
        body.append(f"<tr{marker}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def versions():
    out = [("Python", sys.version.split()[0]), ("Sistema", platform.platform())]
    for package in ("Orange3", "orange_gualbe_toolbox", "numpy", "scipy", "scikit-learn", "baycomp",
                    "matplotlib", "pandas"):
        try:
            out.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            out.append((package, "desconocida"))
    return out


def report_figures(report, scoreset):
    theme = plots.LIGHT
    width, height = 8.6, 4.0
    out = [("Comparación de modelos", plots.scores_figure(report, scoreset, width, height, theme))]
    if report.friedman is not None and report.friedman.n_models >= 3:
        out.append(("Diagrama de diferencias críticas",
                    plots.critical_difference_figure(report, width, 3.2, theme)))
    if report.primary:
        out.append((f"Matriz de comparaciones: {st.TEST_NAMES[report.primary]}",
                    plots.matrix_figure(report, report.primary, width, height, theme)))
    if report.pairs_of(st.BAYES):
        out.append(("Probabilidades bayesianas", plots.bayes_figure(report, width, height, theme)))
    out.append(("Tiempos", plots.times_figure(report, scoreset, width, 3.0, theme)))
    return out


def learner_parameters(learner):
    params = getattr(learner, "params", None)
    if not isinstance(params, dict):
        return "—"
    clean = {key: value for key, value in params.items()
             if key not in ("self", "preprocessors") and not key.startswith("_")}
    return ", ".join(f"{key}={value!r}" for key, value in sorted(clean.items())) or "—"


def build_html(context, scoreset, report):
    """Devuelve el informe completo. `context` describe datos y muestreo."""
    parts = [f"<h1>{_esc(context['title'])}</h1>",
             f"<p class='small'>Generado el {datetime.now():%Y-%m-%d %H:%M} con Test &amp; Score Pro."
             "</p>"]
    parts.append("<div class='rec'>" + "<br>".join(_esc(line) for line in report.recommendation) + "</div>")

    parts.append("<h2>Datos y muestreo</h2>")
    parts.append(_rows(["Concepto", "Valor"], [[_esc(k), _esc(v)] for k, v in context["setup"]]))
    if context.get("failures"):
        parts.append("<h3>Modelos que han fallado</h3>")
        parts.extend(f"<div class='note'>{_esc(line)}</div>" for line in context["failures"])
    if context.get("notes"):
        parts.extend(f"<div class='note'>{_esc(line)}</div>" for line in context["notes"])

    parts.append("<h2>Modelos</h2>")
    parts.append(_rows(["Modelo", "Parámetros"],
                       [[_esc(n), _esc(p)] for n, p in context["models"]]))

    parts.append("<h2>Puntuaciones</h2>")
    labels = [metric_label(scorer) for scorer in scoreset.scorers]
    headers = ["Modelo"] + labels + [f"IC {1 - report.alpha:.0%} ({report.metric_label})", "Rango medio",
                                     "Entrenamiento (s)", "Predicción (s)"]
    rows = []
    order = list(report.order)
    for model in order:
        row = [_esc(report.names[model])]
        row += [format_number(scoreset.overall[s.__name__][model]) for s in scoreset.scorers]
        low, high = report.ci_low[model], report.ci_high[model]
        row.append(f"{format_number(low)} – {format_number(high)}" if np.isfinite(low) else "—")
        row += [format_number(report.mean_ranks[model], 2),
                format_number(scoreset.train_time[model], 3),
                format_number(scoreset.test_time[model], 3)]
        rows.append(row)
    parts.append(_rows(headers, rows, numeric=range(1, len(headers)), best=0))
    for scorer in scoreset.scorers:
        if scoreset.undefined.get(scorer.__name__):
            parts.append(f"<p class='small'>{_esc(metric_label(scorer))}: no definida en "
                         f"{scoreset.undefined[scorer.__name__]} particiones.</p>")

    parts.append("<h2>Gráficos</h2>")
    for title, figure in report_figures(report, scoreset):
        data = base64.b64encode(plots.figure_png(figure, 130)).decode("ascii")
        parts.append(f"<h3>{_esc(title)}</h3><img alt='{_esc(title)}' src='data:image/png;base64,{data}'>")

    parts.append("<h2>Comparaciones estadísticas</h2>")
    correction = st.CORRECTIONS.get(report.correction, report.correction)
    parts.append(f"<p>Métrica: <b>{_esc(report.metric_label)}</b> · α = {report.alpha:g} · "
                 f"corrección múltiple: {_esc(correction)} · ROPE = ±{report.rope:.4g}</p>")
    if report.friedman is not None and report.friedman.n_blocks:
        fried = report.friedman
        parts.append("<h3>Friedman e Iman-Davenport</h3>")
        parts.append(_rows(
            ["Estadístico", "Valor", "p"],
            [["χ² de Friedman", format_number(fried.chi2), format_p(fried.p_value)],
             ["F de Iman-Davenport", format_number(fried.iman_davenport), format_p(fried.p_iman_davenport)],
             ["Diferencia crítica (Nemenyi)", format_number(fried.critical_difference), "—"]],
            numeric=(1, 2)))
        parts.append(f"<p class='small'>Supuesto: {_esc(fried.assumption)}</p>")
        parts.extend(f"<div class='note'>{_esc(line)}</div>" for line in fried.warnings)
    for test in report.available:
        pairs = report.pairs_of(test)
        parts.append(f"<h3>{_esc(st.TEST_NAMES[test])}</h3><p>{_esc(EXPLANATIONS[test])}</p>")
        if test == st.BAYES:
            headers = ["A", "B", "Diferencia media", "P(A mejor)", "P(equivalentes)", "P(B mejor)", "Decisión"]
            rows = [[_esc(report.names[p.a]), _esc(report.names[p.b]), format_number(p.mean_diff),
                     format_number(p.p_a, 3), format_number(p.p_rope, 3), format_number(p.p_b, 3),
                     _esc(p.decision)] for p in pairs]
            numeric = (2, 3, 4, 5)
        else:
            headers = ["A", "B", "Estadístico", "p", "p ajustado", "Diferencia", "IC", "Efecto", "Decisión"]
            rows = []
            for p in pairs:
                interval = f"{format_number(p.ci_low)} – {format_number(p.ci_high)}" if np.isfinite(p.ci_low) else "—"
                effect = (f"{format_number(p.effect, 3)} ({_esc(p.effect_name)}, {_esc(p.effect_label)})"
                          if np.isfinite(p.effect) else "—")
                rows.append([_esc(report.names[p.a]), _esc(report.names[p.b]), format_number(p.statistic),
                             format_p(p.p_value), format_p(p.p_adjusted), format_number(p.mean_diff),
                             interval, effect, _esc(p.decision)])
            numeric = (2, 3, 4, 5)
        parts.append(_rows(headers, rows, numeric=numeric))
        assumption = next((p.assumption for p in pairs if p.assumption), "")
        if assumption:
            parts.append(f"<p class='small'>Supuesto: {_esc(assumption)}</p>")
        repeated = {}
        for p in pairs:
            for line in p.warnings:
                repeated[line] = repeated.get(line, 0) + 1
        parts.extend(f"<div class='note'>{_esc(line)}"
                     + (f" ({count} comparaciones)" if count > 1 else "") + "</div>"
                     for line, count in repeated.items())

    parts.append("<h2>Reproducibilidad</h2>")
    parts.append(_rows(["Software", "Versión"], [[_esc(k), _esc(v)] for k, v in versions()]))
    body = "\n".join(parts)
    return (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>{_esc(context['title'])}"
            f"</title><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<style>{STYLE}</style></head><body>{body}</body></html>")
