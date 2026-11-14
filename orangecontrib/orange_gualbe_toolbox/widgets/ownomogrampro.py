# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Widget Nomogram Pro: nomograma de cualquier modelo de clasificación o regresión.

El cálculo (descomposición exacta o aproximada, fidelidad, escalas de puntos) vive en
_nomo_engine.py, sin Qt. Aquí están la escena, que se dibuja de nuevo al cambiar la vista
y solo en su parte dinámica (marcadores, contribuciones y pie) al mover un valor, el
arrastre de los marcadores, las entradas y las salidas.
"""
import numpy as np

from AnyQt.QtCore import QPointF, QRectF, Qt, QTimer, QSize
from AnyQt.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPalette, QPen,
    QPolygonF,
)
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QInputDialog, QLabel,
)

from Orange.base import Model
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import AttributeList, Input, Msg, OWWidget, Output

from orangecontrib.orange_gualbe_toolbox.widgets import _nomo_engine as E

SORTS = ("Orden original", "Nombre", "Rango de puntos", "Importancia en los datos",
         "Contribución a la instancia")
SORT_NONE, SORT_NAME, SORT_RANGE, SORT_DATA, SORT_INSTANCE = range(5)
METHODS = ("Automático (el más fiel)", "Dependencia parcial", "Ajuste aditivo directo")
SPACES = ("Automática (la más fiel)", "Probabilidad", "Log-odds")
SCALES = ("Puntos (0–100)", "Unidades del modelo")
POINTS, UNITS = range(2)

POSITIVE = QColor("#2B6CB0")
NEGATIVE = QColor("#C53030")
NEUTRAL = QColor("#718096")
RUG = QColor(160, 160, 160, 140)
BAND = QColor(43, 108, 176, 55)
REAL = QColor("#1A202C")

ROW_H = 50          # fila de una escala de una dimensión
CURVE_H = 92        # fila de un efecto no monótono (curva)
LEFT_MIN = 150
CONTRIB_W = 150
AXIS_H = 44


def run_analysis(model, data, params, state):
    """Tarea en segundo plano: la descomposición aditiva del modelo."""
    return E.analyze(model, data, state=state, **params)


def describe_model(analysis):
    name = analysis.model_name
    return name if name else "modelo"


def fidelity_summary(view):
    """Texto (HTML) del modo y la fidelidad y nivel: ok, medium o low."""
    a = view.analysis
    fid = view.fidelity
    link = view.link
    if a.mode == E.EXACT:
        parts = ["<b>✔ Nomograma exacto</b>: la suma de los puntos reproduce la "
                 "puntuación del modelo"]
        if fid.get("max_error") is not None:
            parts[0] += (f" (error relativo máximo {E.format_number(fid['max_error'], 2)}"
                         f" en {fid.get('rows', 0)} filas)")
        parts[0] += "."
        if a.kind == "classification":
            if link.kind == "logistic":
                parts.append("La escala de probabilidad también es exacta.")
            elif link.kind == "monotone":
                parts.append(
                    "La escala de probabilidad es aproximada (error medio "
                    f"{E.format_number(fid.get('mae', float('nan')), 2)}): "
                    + ("con más de dos clases la probabilidad depende de todas."
                       if view.mask.sum() > 2 else
                       "el modelo calibra la probabilidad de forma no exactamente "
                       "logística."))
            elif link.kind == "decision":
                parts.append("El modelo no da probabilidades: el pie muestra su "
                             "puntuación de decisión.")
            elif link.kind == "score":
                parts.append("Con más de dos clases la probabilidad depende de todas: "
                             "el pie muestra la puntuación de la clase (log-odds frente "
                             "a la media de las demás) y la probabilidad real se da "
                             "debajo.")
        return "<br/>".join(parts), "ok"
    var = view.variant
    method = {"pd": "dependencia parcial", "fit": "ajuste aditivo directo"}.get(
        var.method, var.method)
    space = {"prob": "probabilidad", "logit": "log-odds", "value": "valor"}.get(
        var.space, var.space)
    r2 = fid.get("r2", float("nan"))
    level = "ok" if r2 >= 0.95 else ("medium" if r2 >= 0.8 else "low")
    lines = [f"<b>≈ Nomograma aproximado</b> ({method}, en {space}): el modelo no es "
             "una suma de efectos de cada atributo."]
    detail = f"Fidelidad de la suma aditiva: R² = {E.format_number(r2, 3)}"
    if a.kind == "classification":
        detail += (f", error medio {E.format_number(fid.get('mae', float('nan')), 2)}"
                   " en probabilidad")
        if "agreement" in fid:
            detail += f", misma clase en el {fid['agreement'] * 100:.0f} %"
    else:
        detail += (f", error medio {E.format_number(fid.get('mae', float('nan')), 3)}"
                   f" ({E.format_number(100 * fid.get('mae', 0) / fid.get('scale', 1), 2)}"
                   " % de su desviación)")
    detail += f" en {fid.get('rows', 0)} filas."
    lines.append(detail)
    lines.append({"ok": "Alta: el nomograma representa bien al modelo.",
                  "medium": "Media: el modelo tiene interacciones que el nomograma no "
                            "muestra; úsalo como orientación.",
                  "low": "Baja: el modelo depende sobre todo de combinaciones de "
                         "atributos; el nomograma solo da una idea general."}[level])
    lines.append("La predicción que se muestra siempre es la del modelo real.")
    if var.note:
        lines.append(var.note)
    return "<br/>".join(lines), level


class NomogramView(QGraphicsView):
    """Vista de la escena; gestiona el arrastre de los marcadores."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.TextAntialiasing)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._dragging = None
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.widget.schedule_layout()

    def sizeHint(self):
        return QSize(900, 600)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint()
                                  if hasattr(event, "position") else event.pos())
            row = self.widget.row_at(pos)
            if row is not None and row.contains_scale(pos):
                self._dragging = row
                self.widget.set_from_position(row, pos, dragging=True)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint()
                              if hasattr(event, "position") else event.pos())
        if self._dragging is not None:
            self.widget.set_from_position(self._dragging, pos, dragging=True)
            event.accept()
            return
        row = self.widget.row_at(pos)
        cursor = Qt.CursorShape.PointingHandCursor \
            if row is not None and row.contains_scale(pos) else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging is not None and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            self.widget.drag_finished()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        pos = self.mapToScene(event.position().toPoint()
                              if hasattr(event, "position") else event.pos())
        row = self.widget.row_at(pos)
        if row is not None and not row.contains_scale(pos):
            self.widget.edit_value(row)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class Row:
    """Geometría de la fila de un atributo y conversión posición ↔ valor."""

    def __init__(self, effect, y, height, x0, x1, curve):
        self.effect = effect
        self.y = y
        self.height = height
        self.x0, self.x1 = x0, x1
        self.curve = curve
        self.xs = None          # posiciones x de la rejilla (1D) o de la curva
        self.ys = None          # (curva) posiciones y de la rejilla
        self.values = None      # valores de la rejilla
        self.left = 0           # extremos horizontales de la escala
        self.right = 0

    @property
    def line_y(self):
        return self.y + 26

    def contains_scale(self, pos):
        return (self.y <= pos.y() < self.y + self.height
                and self.left - 12 <= pos.x() <= self.right + 12)

    def value_at(self, pos):
        e = self.effect
        if e.is_discrete:
            i = int(np.argmin(np.abs(self.xs - pos.x())))
            return float(self.values[i])
        if self.curve:
            top, bottom = self.y + 10, self.y + self.height - 18
            t = (bottom - pos.y()) / max(1.0, bottom - top)
            t = min(1.0, max(0.0, t))
            return float(e.lo + t * (e.hi - e.lo))
        # escala monótona: se invierte por interpolación
        xs, vs = self.xs, self.values
        order = np.argsort(xs)
        xs, vs = xs[order], vs[order]
        if xs[-1] - xs[0] < 1e-9:
            return float(vs[0])
        return float(np.interp(pos.x(), xs, vs))


class OWNomogramPro(OWWidget, ConcurrentWidgetMixin):
    name = "Nomogram Pro"
    description = ("Nomograma de cualquier modelo de clasificación o regresión: exacto "
                   "para los modelos aditivos y aproximado, con su fidelidad, para el "
                   "resto. Permite cambiar los valores y ver la predicción real.")
    icon = "icons/ownomogrampro.svg"
    priority = 50
    keywords = ["nomogram", "nomogram pro", "nomograma", "explicar", "explain",
                "contribuciones", "partial dependence", "dependencia parcial",
                "puntos", "points"]

    class Inputs:
        model = Input("Model", Model, default=True)
        data = Input("Data", Table)
        instance = Input("Instance", Table)

    class Outputs:
        features = Output("Features", AttributeList, dynamic=False)
        instance = Output("Instance", Table, dynamic=False)
        contributions = Output("Contributions", Table, dynamic=False)

    class Error(OWWidget.Error):
        unsupported = Msg("{}")
        failed = Msg("No se pudo calcular el nomograma: {}")
        no_target = Msg("La clase «{}» no aparece en los datos con los que se entrenó el "
                        "modelo: elige otra clase objetivo.")

    class Warning(OWWidget.Warning):
        low_fidelity = Msg("Fidelidad baja (R² = {}): el modelo depende de combinaciones "
                           "de atributos que un nomograma no puede mostrar.")
        difference = Msg("En esta instancia la suma aditiva ({}) se aleja de la "
                         "predicción del modelo real ({}).")
        instance = Msg("{}")

    class Information(OWWidget.Information):
        notes = Msg("{}")
        hidden = Msg("Se muestran {} de {} atributos; los demás se suman en "
                     "«Otros atributos».")

    graph_name = "scene"

    # Ojo: ningún Setting puede llamarse como un método de QWidget/OWWidget.
    target_class_name = Setting("", schema_only=True)
    scale_index = Setting(POINTS)
    sort_index = Setting(SORT_RANGE)
    n_attributes = Setting(10)
    show_all = Setting(False)
    method_index = Setting(E.METHOD_AUTO)
    space_index = Setting(E.SPACE_AUTO)
    sample_size = Setting(1000)
    grid_size = Setting(20)
    show_distribution = Setting(True)
    show_contributions = Setting(True)
    saved_values = Setting({}, schema_only=True)
    auto_commit = Setting(True)

    want_main_area = True

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.model = None
        self.data = None
        self.instance_data = None
        self.analysis = None
        self.view_ = None           # TargetView actual
        self.values = None          # valores de la instancia (uno por atributo)
        self.real = None            # predicción real de la instancia
        self.spread = None
        self.shown = []
        self.hidden = []
        self.scale = None
        self.rows = []
        self._dynamic = []
        self._cache = []
        self._order_frozen = None
        self._layout_timer = QTimer(self, singleShot=True, interval=60,
                                    timeout=self.relayout)
        self._predict_timer = QTimer(self, singleShot=True, interval=40,
                                     timeout=self._update_real)

        self._build_controls()
        self.scene = QGraphicsScene(self)
        self.view = NomogramView(self)
        self.view.setScene(self.scene)
        self.mainArea.layout().addWidget(self.view)
        self._refresh_controls()

    # ------------------------------------------------------------------ controles
    def _build_controls(self):
        box = gui.vBox(self.controlArea, "Modelo")
        self.model_label = QLabel("Conecta un modelo de clasificación o regresión.")
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.TextFormat.RichText)
        # un QLabel con ajuste de línea pide tanto ancho como su texto: se acota
        # para que el panel de controles no se recorte
        self.model_label.setMinimumWidth(230)
        self.model_label.setMaximumWidth(300)
        box.layout().addWidget(self.model_label)

        box = gui.vBox(self.controlArea, "Objetivo")
        self.target_combo = gui.comboBox(box, self, "target_class_name",
                                         label="Clase objetivo:", orientation=Qt.Orientation.Horizontal,
                                         sendSelectedValue=True, callback=self._target_changed,
                                         contentsLength=12)
        self.target_label = gui.widgetLabel(box, "")

        box = gui.vBox(self.controlArea, "Escala")
        gui.radioButtons(box, self, "scale_index", SCALES, callback=self.relayout)

        box = gui.vBox(self.controlArea, "Atributos")
        gui.comboBox(box, self, "sort_index", label="Ordenar por:", items=SORTS,
                     orientation=Qt.Orientation.Horizontal, callback=self._sort_changed)
        hb = gui.hBox(box)
        self.n_spin = gui.spin(hb, self, "n_attributes", 1, 500, label="Mostrar los",
                               callback=self.relayout, controlWidth=60)
        gui.widgetLabel(hb, "más importantes")
        gui.checkBox(box, self, "show_all", "Mostrar todos", callback=self._show_all_changed)

        box = gui.vBox(self.controlArea, "Vista")
        gui.checkBox(box, self, "show_distribution", "Distribución de los datos",
                     callback=self.relayout)
        gui.checkBox(box, self, "show_contributions", "Contribuciones de la instancia",
                     callback=self.relayout)
        gui.button(box, self, "Valores típicos", callback=self.reset_values,
                   tooltip="Mediana de cada atributo numérico y valor más frecuente de "
                           "cada categórico (o la instancia de la entrada Instance).")

        self.approx_box = box = gui.vBox(self.controlArea, "Aproximación")
        gui.comboBox(box, self, "method_index", label="Método:", items=METHODS,
                     callback=self._params_changed)
        self.space_combo = gui.comboBox(box, self, "space_index", label="Escala aditiva:",
                                        items=SPACES, callback=self._space_changed)
        gui.spin(box, self, "sample_size", 100, 20000, step=100,
                 label="Filas de referencia (máx.):", callback=self._params_changed,
                 controlWidth=70, keyboardTracking=False)
        gui.spin(box, self, "grid_size", 5, 100, label="Puntos por atributo numérico:",
                 callback=self._params_changed, controlWidth=70, keyboardTracking=False)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def _refresh_controls(self):
        a = self.analysis
        self.n_spin.setEnabled(not self.show_all)
        cls = a is not None and a.kind == "classification"
        self.target_combo.setEnabled(cls)
        self.approx_box.setEnabled(a is not None and a.mode == E.APPROXIMATE)
        self.space_combo.setEnabled(cls)

    # ------------------------------------------------------------------ entradas
    @Inputs.model
    def set_model(self, model):
        self.model = model

    @Inputs.data
    def set_data(self, data):
        self.data = data if data is not None and len(data) else None

    @Inputs.instance
    def set_instance(self, data):
        self.instance_data = data if data is not None and len(data) else None

    def handleNewSignals(self):
        self._recompute()

    def _params(self):
        return dict(method=self.method_index, sample_size=int(self.sample_size),
                    grid_size=int(self.grid_size))

    def _recompute(self):
        self.cancel()
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if self.model is None:
            self._set_analysis(None)
            return
        try:
            E.model_kind(self.model)
        except E.NomogramError as exc:
            self.Error.unsupported(str(exc))
            self._set_analysis(None)
            return
        params = self._params()
        for key, analysis in self._cache:
            if key[0] is self.model and key[1] is self.data and key[2] == params:
                self._set_analysis(analysis)
                return
        self.model_label.setText("Calculando…")
        self.start(run_analysis, self.model, self.data, params)

    def on_done(self, analysis):
        params = self._params()
        self._cache = self._cache[-3:]
        self._cache.append(((self.model, self.data, params), analysis))
        self._set_analysis(analysis)

    def on_exception(self, ex):
        if isinstance(ex, E.NomogramError):
            self.Error.unsupported(str(ex))
        else:
            self.Error.failed(f"{type(ex).__name__}: {ex}")
        self._set_analysis(None)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    # ------------------------------------------------------------------ análisis
    def _set_analysis(self, analysis):
        prev_names = None
        if self.analysis is not None and self.values is not None:
            prev_names = {v.name: (v, val) for v, val in
                          zip(self.analysis.domain.attributes, self.values)}
        self.analysis = analysis
        self.view_ = None
        self.values = None
        self.real = None
        if analysis is None:
            self.target_combo.clear()
            self.target_label.setText("")
            self.model_label.setText("Conecta un modelo de clasificación o regresión.")
            self.scene.clear()
            self.rows = []
            self._dynamic = []
            self._refresh_controls()
            self.commit.now()
            return
        if analysis.kind == "classification":
            values = list(analysis.class_var.values)
            self.target_combo.clear()
            self.target_combo.addItems(values)
            if self.target_class_name not in values:
                self.target_class_name = values[E.default_target(analysis.class_var)]
            self.target_combo.setCurrentIndex(values.index(self.target_class_name))
            self.target_label.setText("")
        else:
            self.target_combo.clear()
            self.target_label.setText(f"Regresión: predice «{analysis.class_var.name}».")
        self.values = self._initial_values(prev_names)
        if analysis.notes:
            self.Information.notes(" ".join(analysis.notes))
        self._refresh_controls()
        self._make_view()

    def _target_index(self):
        a = self.analysis
        if a is None or a.kind != "classification":
            return 0
        values = list(a.class_var.values)
        return values.index(self.target_class_name) \
            if self.target_class_name in values else 0

    def _make_view(self):
        a = self.analysis
        self.Error.no_target.clear()
        self.Warning.low_fidelity.clear()
        if a is None:
            return
        target = self._target_index()
        if a.kind == "classification" and a.present is not None \
                and not a.present[target]:
            self.Error.no_target(a.class_var.values[target])
        self.view_ = a.target_view(target, self.space_index)
        html, level = fidelity_summary(self.view_)
        color = {"ok": "#276749", "medium": "#975A16", "low": "#9B2C2C"}[level]
        self.model_label.setText(
            f"<span style='font-size:110%'><b>{describe_model(a)}</b></span><br/>"
            f"<span style='color:{color}'>{html}</span>")
        r2 = self.view_.fidelity.get("r2")
        if a.mode == E.APPROXIMATE and r2 is not None and np.isfinite(r2) and r2 < 0.8:
            self.Warning.low_fidelity(E.format_number(r2, 3))
        self._order_frozen = None
        self._update_real(redraw=False)
        self.relayout()
        self.commit.deferred()

    def _initial_values(self, previous=None):
        a = self.analysis
        values = self.typical_values()
        saved = self.saved_values or {}
        for i, var in enumerate(a.domain.attributes):
            if previous and var.name in previous:
                pvar, pval = previous[var.name]
                v = self._convert_value(pvar, pval, var)
                if v is not None:
                    values[i] = v
                    continue
            if var.name in saved:
                v = saved[var.name]
                if var.is_discrete and isinstance(v, str):
                    v = float(var.values.index(v)) if v in var.values else None
                if v is not None:
                    values[i] = float(v)
        self._apply_instance(values)
        return values

    def typical_values(self):
        a = self.analysis
        values = np.full(len(a.domain.attributes), np.nan)
        effects = {e.index: e for e in next(iter(a.variants.values())).effects}
        for i, var in enumerate(a.domain.attributes):
            e = effects[i]
            col = a.sample[:, i] if a.sample is not None else None
            if col is not None and np.isfinite(col).any():
                known = col[np.isfinite(col)]
                if var.is_discrete:
                    values[i] = float(np.bincount(known.astype(int)).argmax())
                else:
                    values[i] = float(np.median(known))
            elif e.kind == "discrete":
                values[i] = 0.0
            elif e.kind == "bins":
                values[i] = float(e.grid[len(e.grid) // 2])
            else:
                values[i] = (e.lo + e.hi) / 2
        return values

    @staticmethod
    def _convert_value(src_var, value, var):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if var.is_discrete:
            if not src_var.is_discrete:
                return None
            name = src_var.values[int(value)]
            return float(var.values.index(name)) if name in var.values else None
        if src_var.is_discrete:
            return None
        return float(value)

    def _apply_instance(self, values):
        self.Warning.instance.clear()
        data = self.instance_data
        if data is None or self.analysis is None:
            return
        row = data[0]
        names = {v.name: v for v in data.domain.variables + data.domain.metas}
        missing, bad = [], []
        for i, var in enumerate(self.analysis.domain.attributes):
            src = names.get(var.name)
            if src is None:
                missing.append(var.name)
                continue
            v = self._convert_value(src, float(row[src]), var) \
                if src.is_primitive() else None
            if v is None:
                bad.append(var.name)
            else:
                values[i] = v
        msgs = []
        if missing:
            msgs.append("La instancia no tiene " + E._names(missing) + ".")
        if bad:
            msgs.append("No se pudo usar el valor de " + E._names(bad) + ".")
        if len(data) > 1:
            msgs.append(f"La entrada Instance tiene {len(data)} filas: se usa la primera.")
        if msgs:
            self.Warning.instance(" ".join(msgs))

    # ------------------------------------------------------------------ callbacks
    def _target_changed(self):
        self._make_view()

    def _space_changed(self):
        self._make_view()

    def _params_changed(self):
        self._recompute()

    def _sort_changed(self):
        self._order_frozen = None
        self.relayout()

    def _show_all_changed(self):
        self.n_spin.setEnabled(not self.show_all)
        self.relayout()

    def reset_values(self):
        if self.analysis is None:
            return
        self.values = self.typical_values()
        self._apply_instance(self.values)
        self._values_changed(final=True)

    def schedule_layout(self):
        self._layout_timer.start()

    # ------------------------------------------------------------------ orden
    def _ordered_effects(self):
        view = self.view_
        effects = list(view.effects)
        if self.sort_index == SORT_NAME:
            effects.sort(key=lambda e: e.name.lower())
        elif self.sort_index == SORT_RANGE:
            rng = {e.index: np.ptp(view.grid_component(e)) for e in effects}
            effects.sort(key=lambda e: -rng[e.index])
        elif self.sort_index == SORT_DATA:
            effects.sort(key=lambda e: -view.importance(e))
        elif self.sort_index == SORT_INSTANCE:
            if self._order_frozen is not None:
                pos = {i: k for k, i in enumerate(self._order_frozen)}
                effects.sort(key=lambda e: pos.get(e.index, 0))
            else:
                comp = {e.index: abs(float(view.component(e, [self.values[e.index]])[0]))
                        for e in effects}
                effects.sort(key=lambda e: -comp[e.index])
        return effects

    # ------------------------------------------------------------------ dibujo
    def _palette(self):
        pal = self.view.palette()
        return (pal.color(QPalette.ColorRole.Base), pal.color(QPalette.ColorRole.Text))

    def relayout(self):
        self.scene.clear()
        self.rows = []
        self._dynamic = []
        self.Information.hidden.clear()
        view = self.view_
        if view is None or self.values is None:
            return
        bg, fg = self._palette()
        self.scene.setBackgroundBrush(QBrush(bg))
        self._fg = fg
        effects = self._ordered_effects()
        n = len(effects) if self.show_all else max(1, int(self.n_attributes))
        self.shown, self.hidden = effects[:n], effects[n:]
        if self.hidden:
            self.Information.hidden(len(self.shown), len(effects))
        extra = {}
        for e in self.shown:
            v = self.values[e.index]
            if e.kind == "linear" and np.isfinite(v):
                extra[e.index] = [v]
        self.scale = E.point_scale(view, self.shown, extra)
        font = QFont(self.view.font())
        self._font = font
        bold = QFont(font)
        bold.setBold(True)
        self._bold = bold
        small = QFont(font)
        small.setPointSizeF(max(7.0, font.pointSizeF() * 0.86))
        self._small = small
        fm = QFontMetricsF(bold)
        left = max(LEFT_MIN, min(260, max((fm.horizontalAdvance(e.name) for e in self.shown),
                                          default=0) + 24))
        # reserva el sitio de la barra vertical aunque aún no se vea: si no, al
        # aparecer quitaría ancho y saldría una barra horizontal
        vbar = self.view.verticalScrollBar()
        reserve = 0 if vbar.isVisible() else vbar.sizeHint().width()
        width = max(640, self.view.viewport().width() - reserve - 4)
        contrib_w = CONTRIB_W if self.show_contributions else 0
        x0 = left + 12
        x1 = width - contrib_w - 24
        self._x0, self._x1 = x0, x1
        self._width = width
        self._contrib = (x1 + 30, width - 12) if contrib_w else None
        y = 4
        y = self._draw_header(y, x0, x1, left)
        for e in self.shown:
            y = self._draw_row(e, y, x0, x1)
        if self.hidden:
            y = self._draw_hidden_row(y, x0)
        self._footer_y = y + 8
        self._draw_dynamic()

    def _px(self, points):
        return self._x0 + points / 100.0 * (self._x1 - self._x0)

    def _text(self, text, x, y, font=None, color=None, anchor="left", dynamic=False,
              tooltip=None):
        item = QGraphicsSimpleTextItem(text)
        item.setFont(font or self._font)
        item.setBrush(QBrush(color or self._fg))
        w = item.boundingRect().width()
        if anchor == "center":
            x -= w / 2
        elif anchor == "right":
            x -= w
        item.setPos(x, y)
        if tooltip:
            item.setToolTip(tooltip)
        self.scene.addItem(item)
        if dynamic:
            self._dynamic.append(item)
        return item

    def _line(self, x0, y0, x1, y1, color=None, width=1.0, dynamic=False, style=None):
        pen = QPen(color or self._fg, width)
        pen.setCosmetic(True)
        if style is not None:
            pen.setStyle(style)
        item = self.scene.addLine(x0, y0, x1, y1, pen)
        if dynamic:
            self._dynamic.append(item)
        return item

    def _rect(self, x, y, w, h, color, dynamic=False, pen=None):
        item = self.scene.addRect(QRectF(x, y, w, h), pen or QPen(Qt.PenStyle.NoPen),
                                  QBrush(color))
        if dynamic:
            self._dynamic.append(item)
        return item

    def _axis(self, y, x0, x1, lo, hi, label_of, ticks=None, small=False):
        """Eje horizontal entre x0 y x1 para valores de lo a hi."""
        self._line(x0, y, x1, y, width=1.2)
        ticks = E.nice_ticks(lo, hi, 10) if ticks is None else ticks
        last = -1e9
        for t in ticks:
            x = x0 + (t - lo) / (hi - lo) * (x1 - x0) if hi > lo else x0
            self._line(x, y - 4, x, y + 4)
            txt = label_of(t)
            fm = QFontMetricsF(self._small)
            w = fm.horizontalAdvance(txt)
            if x - w / 2 > last + 4:
                self._text(txt, x, y - 20, self._small, anchor="center")
                last = x + w / 2

    def _draw_header(self, y, x0, x1, left):
        view = self.view_
        k = self.scale.k
        if self.scale_index == POINTS:
            title = "Puntos"
            self._axis(y + 26, x0, x1, 0, 100, lambda t: E.format_number(t))
        else:
            unit = {"score": "log-odds", "logit": "log-odds", "prob": "probabilidad",
                    "value": f"unidades de {self.analysis.class_var.name}"}[view.variant.space]
            if view.variant.space == "score" and self.analysis.kind == "regression":
                unit = "valor"
            if view.link.kind == "decision":
                unit = "puntuación"
            title = f"Efecto ({unit})"
            span = 100 / k
            self._axis(y + 26, x0, x1, 0, span, lambda t: E.format_number(t))
        self._text(title, 8, y + 10, self._bold)
        if self._contrib:
            c0, c1 = self._contrib
            self._text("Contribución", (c0 + c1) / 2, y + 2, self._bold, anchor="center")
            self._text("frente a la media", (c0 + c1) / 2, y + 20, self._small,
                       color=NEUTRAL, anchor="center")
        return y + AXIS_H + 6

    def _effect_label(self, e, value):
        return E.format_value(e.var, value) if e.kind != "bins" else (
            e.labels[e.position_value(value)] if e.position_value(value) is not None
            else "?")

    def _draw_row(self, e, y, x0, x1):
        view = self.view_
        comp = view.grid_component(e)
        pts = self.scale.points(e.index, comp)
        curve = False
        if e.kind == "curve":
            curve = E.is_monotone(comp) == 0
        height = CURVE_H if curve else ROW_H
        row = Row(e, y, height, x0, x1, curve)
        self.rows.append(row)
        # fondo alterno suave
        if len(self.rows) % 2 == 0:
            self._rect(0, y, self._width, height, QColor(128, 128, 128, 18))
        self._text(e.name, 8, y + 6, self._bold, tooltip=self._row_tooltip(e))
        if e.is_discrete:
            self._draw_discrete(row, e, pts)
        elif curve:
            self._draw_curve(row, e, pts)
        else:
            self._draw_ruler(row, e, comp, pts)
        return y + height

    def _row_tooltip(self, e):
        view = self.view_
        lines = [f"<b>{e.name}</b>",
                 f"Rango: {E.format_number(np.ptp(view.grid_component(e)) * self.scale.k, 3)}"
                 " puntos"]
        if e.cov is not None:
            lines.append("Importancia en los datos (desviación de su contribución): "
                         f"{E.format_number(view.importance(e) * self.scale.k, 3)} puntos")
        if e.n_unknown:
            lines.append(f"Desconocidos en los datos: {e.n_unknown}")
        lines.append("Doble clic en el nombre para escribir el valor.")
        return "<br/>".join(lines)

    def _draw_discrete(self, row, e, pts):
        xs = np.array([self._px(p) for p in pts])
        ly = row.line_y
        if not len(xs):
            return
        row.xs, row.values = xs, np.arange(len(xs), dtype=float)
        row.left, row.right = float(xs.min()), float(xs.max())
        self._line(row.left, ly, row.right, ly, width=1.2)
        labels = e.labels if e.labels else [str(i) for i in range(len(xs))]
        counts = e.counts if e.counts is not None else None
        total = counts.sum() if counts is not None and counts.sum() else None
        order = np.argsort(xs)
        fm = QFontMetricsF(self._small)
        levels = [-1e9, -1e9]
        for i in order:
            x = xs[i]
            self._line(x, ly - 4, x, ly + 4)
            txt = labels[i]
            if len(txt) > 18:
                txt = txt[:17] + "…"
            w = fm.horizontalAdvance(txt)
            for lev in (0, 1):
                if x - w / 2 > levels[lev] + 3:
                    yy = ly - 20 if lev == 0 else ly + 6
                    if lev == 1 and self.show_distribution:
                        yy = ly + 13
                    self._text(txt, x, yy, self._small, anchor="center",
                               tooltip=labels[i])
                    levels[lev] = x + w / 2
                    break
            if self.show_distribution and total:
                h = 9 * counts[i] / counts.max()
                self._rect(x - 3, ly + 3, 6, h, RUG)

    def _draw_ruler(self, row, e, comp, pts):
        ly = row.line_y
        if e.kind == "linear":
            grid = np.array([e.lo, e.hi]) if e.hi > e.lo else np.array([e.lo])
            gcomp = self.view_.component(e, grid)
            gpts = self.scale.points(e.index, gcomp)
        else:
            grid, gpts = e.grid, pts
        xs = np.array([self._px(p) for p in gpts])
        row.values, row.xs = np.asarray(grid, float), xs
        row.left, row.right = float(xs.min()), float(xs.max())
        self._line(row.left, ly, row.right, ly, width=1.2)

        def x_of(v):
            if e.kind == "linear":
                return self._px(float(self.scale.points(
                    e.index, self.view_.component(e, [v]))[0]))
            return float(np.interp(v, grid, xs))

        ticks = E.nice_ticks(e.lo, e.hi, max(3, int(abs(row.right - row.left) / 70)))
        if len(ticks) < 2:
            # escala corta: al menos sus extremos
            ticks = sorted(set(ticks) | ({e.lo, e.hi} if e.hi > e.lo else {e.lo}))
        fm = QFontMetricsF(self._small)
        placed = []
        for t in ticks:
            x = x_of(t)
            self._line(x, ly - 4, x, ly + 4)
            txt = E.format_value(e.var, t)
            w = fm.horizontalAdvance(txt)
            if all(abs(x - p) > (w + pw) / 2 + 4 for p, pw in placed):
                self._text(txt, x, ly - 20, self._small, anchor="center")
                placed.append((x, w))
        if self.show_distribution and e.counts is not None and e.edges is not None \
                and e.counts.max() > 0:
            for c, a, b in zip(e.counts, e.edges[:-1], e.edges[1:]):
                if c <= 0:
                    continue
                xa, xb = x_of(a), x_of(b)
                xa, xb = min(xa, xb), max(xa, xb)
                h = 9 * c / e.counts.max()
                self._rect(xa, ly + 3, max(1.0, xb - xa - 0.5), h, RUG)

    def _draw_curve(self, row, e, pts):
        top, bottom = row.y + 10, row.y + row.height - 18
        xs = np.array([self._px(p) for p in pts])
        span = e.hi - e.lo if e.hi > e.lo else 1.0
        ys = bottom - (np.asarray(e.grid) - e.lo) / span * (bottom - top)
        row.xs, row.ys, row.values = xs, ys, np.asarray(e.grid, float)
        row.left, row.right = float(xs.min()), float(xs.max())
        # ejes: valor en vertical
        self._line(self._x0 - 6, top, self._x0 - 6, bottom, color=NEUTRAL)
        self._text(E.format_value(e.var, e.hi), self._x0 - 10, top - 6, self._small,
                   anchor="right")
        self._text(E.format_value(e.var, e.lo), self._x0 - 10, bottom - 8, self._small,
                   anchor="right")
        for p in (0, 25, 50, 75, 100):
            self._line(self._px(p), bottom + 4, self._px(p), bottom + 7, color=NEUTRAL)
        path = QPainterPath(QPointF(xs[0], ys[0]))
        for x, yv in zip(xs[1:], ys[1:]):
            path.lineTo(QPointF(x, yv))
        pen = QPen(self._fg, 1.6)
        pen.setCosmetic(True)
        self.scene.addPath(path, pen)
        if self.show_distribution and e.counts is not None and e.edges is not None \
                and e.counts.max() > 0:
            for c, a, b in zip(e.counts, e.edges[:-1], e.edges[1:]):
                if c <= 0:
                    continue
                ya = bottom - (a - e.lo) / span * (bottom - top)
                yb = bottom - (b - e.lo) / span * (bottom - top)
                w = 22 * c / e.counts.max()
                self._rect(self._x0 - 7 - w, min(ya, yb), w, abs(ya - yb) - 0.5, RUG)
        self._text("↕ efecto no monótono", 8, row.y + 46, self._small, color=NEUTRAL,
                   tooltip="El efecto sube y baja con el valor: la curva da los puntos "
                           "(en horizontal) de cada valor (en vertical). Arrastra el "
                           "marcador en vertical para cambiar el valor.")

    def _draw_hidden_row(self, y, x0):
        self._hidden_y = y
        return y + 30

    # ------------------------------------------------------------- parte dinámica
    def _clear_dynamic(self):
        for item in self._dynamic:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        self._dynamic = []

    def _current(self):
        """Componentes, puntos y puntuación de la instancia actual."""
        view = self.view_
        comps = {}
        for e in view.effects:
            comps[e.index] = float(view.component(e, [self.values[e.index]])[0])
        hidden = sum(comps[e.index] for e in self.hidden)
        points = {e.index: float(self.scale.points(e.index, comps[e.index]))
                  for e in self.shown}
        total = sum(points.values())
        score = view.base + sum(comps.values())
        return comps, hidden, points, total, score

    def _contrib_text(self, comp):
        if self.scale_index == POINTS:
            return f"{'+' if comp >= 0 else '−'}{E.format_number(abs(comp) * self.scale.k, 3)}"
        return f"{'+' if comp >= 0 else '−'}{E.format_number(abs(comp), 3)}"

    def _draw_dynamic(self):
        self._clear_dynamic()
        view = self.view_
        if view is None or not self.rows and not self.hidden:
            return
        comps, hidden, points, total, score = self._current()
        # escala de las barras: la mayor contribución actual, con un mínimo del 10 %
        # del mayor efecto posible para que no se amplíe el ruido
        widest = max([self.scale.maxs[e.index] - self.scale.mins[e.index]
                      for e in self.shown] + [0.0])
        bar_max = max([abs(comps[e.index]) for e in self.shown]
                      + [abs(hidden), 0.1 * widest, 1e-300])
        for row in self.rows:
            e = row.effect
            value = self.values[e.index]
            comp = comps[e.index]
            color = POSITIVE if comp > 1e-12 else (NEGATIVE if comp < -1e-12 else NEUTRAL)
            label = self._effect_label(e, value)
            self._text(f"= {label}", 8, row.y + 24, self._font, color=color, dynamic=True)
            x, y = self._marker_pos(row, value, points[e.index])
            unknown = value is None or np.isnan(value)
            r = 7
            item = self.scene.addEllipse(
                QRectF(x - r, y - r, 2 * r, 2 * r),
                QPen(QColor("white") if not unknown else color, 2),
                QBrush(color) if not unknown else QBrush(Qt.BrushStyle.NoBrush))
            item.setZValue(10)
            item.setToolTip(f"{e.name} = {label}: {self._contrib_text(comp)} "
                            f"{'puntos' if self.scale_index == POINTS else ''}")
            self._dynamic.append(item)
            if unknown:
                self._text("?", x, y - 8, self._bold, color=color, anchor="center",
                           dynamic=True)
            if self._contrib:
                self._draw_bar(row.line_y, comp, bar_max, color)
        if self.hidden:
            y = self._hidden_y
            self._text(f"Otros {len(self.hidden)} atributos", 8, y + 6, self._bold,
                       color=NEUTRAL, dynamic=True)
            self._text("suman su contribución a los valores actuales", self._x0,
                       y + 8, self._small, color=NEUTRAL, dynamic=True)
            if self._contrib:
                color = POSITIVE if hidden > 1e-12 else (
                    NEGATIVE if hidden < -1e-12 else NEUTRAL)
                self._draw_bar(y + 14, hidden, bar_max, color)
        self._draw_footer(self._footer_y, total, score, hidden)
        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(QRectF(0, 0, self._width, rect.bottom() + 12))

    def _draw_bar(self, y, comp, bar_max, color):
        """Barra centrada en cero y, a su derecha, el valor de la contribución."""
        c0, c1 = self._contrib
        b1 = c1 - 52
        xc = (c0 + b1) / 2
        half = (b1 - c0) / 2
        w = half * min(1.0, abs(comp) / bar_max) if bar_max > 0 else 0
        self._line(xc, y - 9, xc, y + 9, color=NEUTRAL, dynamic=True)
        x = xc if comp >= 0 else xc - w
        self._rect(x, y - 6, max(w, 0.5), 12, color, dynamic=True)
        item = self._text(self._contrib_text(comp), c1, y, self._small, color=color,
                          anchor="right", dynamic=True)
        item.setY(y - item.boundingRect().height() / 2)

    def _marker_pos(self, row, value, pts):
        e = row.effect
        if row.curve:
            top, bottom = row.y + 10, row.y + row.height - 18
            span = e.hi - e.lo if e.hi > e.lo else 1.0
            v = e.lo if value is None or np.isnan(value) else min(e.hi, max(e.lo, value))
            return self._px(pts), bottom - (v - e.lo) / span * (bottom - top)
        x = self._px(pts)
        x = min(max(x, row.left - 10), row.right + 10)
        return x, row.line_y

    def _draw_footer(self, y, total, score, hidden):
        view = self.view_
        a = self.analysis
        x0, x1 = self._x0, self._x1
        k = self.scale.k
        offset = self.scale.offset + hidden
        tmax = sum(self.scale.maxs[e.index] - self.scale.mins[e.index]
                   for e in self.shown) * k
        tmax = max(tmax, 1e-9)
        dyn = True
        self._line(0, y, self._width, y, color=NEUTRAL, dynamic=dyn)
        y += 12
        # puntos totales
        ya = y + 26
        self._text("Puntos totales", 8, ya - 10, self._bold, dynamic=dyn)
        self._dyn_axis(ya, 0, tmax, lambda t: E.format_number(t))
        xt = x0 + min(max(total, 0), tmax) / tmax * (x1 - x0)
        self._dyn_marker(xt, ya, POSITIVE, f"{E.format_number(total, 4)} puntos")
        y = ya + 24
        # predicción
        s_lo, s_hi = offset, offset + tmax / k

        def x_of_score(s):
            return x0 + (s - s_lo) / (s_hi - s_lo) * (x1 - x0) if s_hi > s_lo else x0

        link = view.link
        ya = y + 26
        if a.kind == "regression":
            title = f"Predicción de {a.class_var.name}"
            ticks = E.nice_ticks(s_lo, s_hi, 8)
            self._dyn_axis(ya, s_lo, s_hi, lambda t: E.format_number(t), ticks=ticks)
            pred_add = score
        elif link.has_probability:
            title = f"P({a.class_var.name} = {a.class_var.values[view.target]})"
            probs = [p for p in (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 0.95, 0.99, 0.999)]
            self._line(x0, ya, x1, ya, width=1.2, dynamic=dyn)
            last = -1e9
            fm = QFontMetricsF(self._small)
            for p in probs:
                s = float(link.inverse(p))
                if not np.isfinite(s) or s < s_lo - 1e-12 or s > s_hi + 1e-12:
                    continue
                x = x_of_score(s)
                self._line(x, ya - 4, x, ya + 4, dynamic=dyn)
                txt = E.format_number(p, 3)
                w = fm.horizontalAdvance(txt)
                if x - w / 2 > last + 4:
                    self._text(txt, x, ya - 20, self._small, anchor="center", dynamic=dyn)
                    last = x + w / 2
            pred_add = float(link(score))
        else:
            title = f"Puntuación de «{a.class_var.values[view.target]}»"
            ticks = E.nice_ticks(s_lo, s_hi, 8)
            self._dyn_axis(ya, s_lo, s_hi, lambda t: E.format_number(t), ticks=ticks)
            if link.kind == "decision" and s_lo <= link.threshold <= s_hi:
                xt0 = x_of_score(link.threshold)
                self._line(xt0, ya - 12, xt0, ya + 12, color=NEGATIVE, width=1.5,
                           dynamic=dyn, style=Qt.PenStyle.DashLine)
                self._text("frontera de decisión", xt0 + 4, ya + 6, self._small,
                           color=NEGATIVE, dynamic=dyn)
            pred_add = score
        self._text(title, 8, ya - 10, self._bold, dynamic=dyn)
        # banda de incertidumbre
        band = self._band()
        if band is not None:
            lo, hi, _ = band
            if a.kind == "classification" and link.has_probability:
                slo, shi = float(link.inverse(lo)), float(link.inverse(hi))
            elif a.kind == "classification":
                slo = shi = np.nan
            else:
                slo, shi = lo, hi
            if np.isfinite(slo) and np.isfinite(shi):
                xa = x_of_score(max(min(slo, shi), s_lo))
                xb = x_of_score(min(max(slo, shi), s_hi))
                if xb > xa:
                    self._rect(xa, ya - 7, xb - xa, 14, BAND, dynamic=dyn)
        xs = x_of_score(min(max(score, s_lo), s_hi))
        self._dyn_marker(xs, ya, POSITIVE, "suma aditiva")
        real_val = self._real_value()
        if real_val is not None and np.isfinite(real_val):
            s_real = real_val if a.kind == "regression" else (
                float(link.inverse(real_val)) if link.has_probability else np.nan)
            if np.isfinite(s_real):
                xr = x_of_score(min(max(s_real, s_lo), s_hi))
                poly = QPolygonF([QPointF(xr, ya - 9), QPointF(xr + 7, ya),
                                  QPointF(xr, ya + 9), QPointF(xr - 7, ya)])
                item = self.scene.addPolygon(poly, QPen(QColor("white"), 1.5),
                                             QBrush(REAL))
                item.setZValue(11)
                item.setToolTip("Predicción del modelo real")
                self._dynamic.append(item)
        y = ya + 22
        for line, color in self._summary_lines(pred_add, real_val):
            self._text(line, 8, y, self._font, color=color, dynamic=dyn)
            y += 20

    def _dyn_axis(self, y, lo, hi, label_of, ticks=None):
        x0, x1 = self._x0, self._x1
        self._line(x0, y, x1, y, width=1.2, dynamic=True)
        ticks = E.nice_ticks(lo, hi, 10) if ticks is None else ticks
        last = -1e9
        fm = QFontMetricsF(self._small)
        for t in ticks:
            x = x0 + (t - lo) / (hi - lo) * (x1 - x0) if hi > lo else x0
            self._line(x, y - 4, x, y + 4, dynamic=True)
            txt = label_of(t)
            w = fm.horizontalAdvance(txt)
            if x - w / 2 > last + 4:
                self._text(txt, x, y - 20, self._small, anchor="center", dynamic=True)
                last = x + w / 2

    def _dyn_marker(self, x, y, color, tooltip):
        r = 7
        item = self.scene.addEllipse(QRectF(x - r, y - r, 2 * r, 2 * r),
                                     QPen(QColor("white"), 2), QBrush(color))
        item.setZValue(10)
        item.setToolTip(tooltip)
        self._dynamic.append(item)

    # ------------------------------------------------------------- predicción real
    def _real_value(self):
        """Probabilidad de la clase objetivo o valor que predice el modelo real."""
        if self.real is None:
            return None
        if self.analysis.kind == "classification":
            return float(self.real[self.view_.target])
        return float(self.real)

    def _band(self):
        a = self.analysis
        if self.spread is not None:
            return self.spread
        if a.kind == "regression" and a.residuals is not None and self.real is not None:
            lo, hi = np.percentile(a.residuals, (5, 95))
            return float(self.real + lo), float(self.real + hi), None
        return None

    def _update_real(self, redraw=True):
        a = self.analysis
        self.real = None
        self.spread = None
        if a is None or self.values is None:
            return
        try:
            self.real = E.predict_instance(a.model, a.domain, self.values)
        except Exception:  # el modelo no admite la instancia (p. ej. desconocidos)
            self.real = None
        target = self._target_index() if a.kind == "classification" else None
        self.spread = E.ensemble_spread(a.model, a.domain, self.values, target)
        if redraw and self.view_ is not None and self.rows:
            self._draw_dynamic()

    def _summary_lines(self, pred_add, real_val):
        a = self.analysis
        view = self.view_
        lines = []
        self.Warning.difference.clear()
        if a.kind == "classification":
            cls = a.class_var.values[view.target]
            if real_val is not None:
                pred_cls = a.class_var.values[int(np.argmax(self.real))]
                lines.append((f"◆ Modelo real: P({cls}) = {E.format_number(real_val * 100, 3)} %"
                              f" · clase predicha «{pred_cls}»", REAL))
            if view.link.has_probability:
                lines.append((f"● Suma de los puntos: P({cls}) ≈ "
                              f"{E.format_number(pred_add * 100, 3)} %", POSITIVE))
                if real_val is not None:
                    tol = max(0.05, 2 * view.fidelity.get("mae", 0)) \
                        if a.mode == E.APPROXIMATE else 0.02
                    if abs(pred_add - real_val) > tol:
                        self.Warning.difference(f"{pred_add * 100:.1f} %",
                                                f"{real_val * 100:.1f} %")
            else:
                lines.append((f"● Puntuación de «{cls}»: {E.format_number(pred_add, 4)}",
                              POSITIVE))
        else:
            if real_val is not None:
                lines.append((f"◆ Modelo real: {a.class_var.name} = "
                              f"{E.format_number(real_val, 5)}", REAL))
            lines.append((f"● Suma de los puntos: {E.format_number(pred_add, 5)}",
                          POSITIVE))
            if real_val is not None:
                scale = view.fidelity.get("scale", 1.0) or 1.0
                tol = max(0.1 * scale, 2 * view.fidelity.get("mae", 0)) \
                    if a.mode == E.APPROXIMATE else 1e-6 * max(1.0, abs(real_val))
                if abs(pred_add - real_val) > tol:
                    self.Warning.difference(E.format_number(pred_add, 4),
                                            E.format_number(real_val, 4))
        band = self._band()
        if band is not None:
            lo, hi, n = band
            fmt = (lambda v: f"{E.format_number(v * 100, 3)} %") \
                if a.kind == "classification" else (lambda v: E.format_number(v, 4))
            if n:
                txt = f"Dispersión entre los {n} árboles (percentiles 5–95): {fmt(lo)} – {fmt(hi)}"
            else:
                where = "de entrenamiento (optimista)" if a.residuals_on_training \
                    else "de referencia"
                txt = (f"Error típico del modelo en los datos {where} (percentiles 5–95 de "
                       f"los residuos): {fmt(lo)} – {fmt(hi)}")
            lines.append(("▬ " + txt, POSITIVE.darker(120)))
        return lines

    # ------------------------------------------------------------- interacción
    def row_at(self, pos):
        for row in self.rows:
            if row.y <= pos.y() < row.y + row.height:
                return row
        return None

    def set_from_position(self, row, pos, dragging=False):
        if row.xs is None:
            return
        value = row.value_at(pos)
        if self.sort_index == SORT_INSTANCE and self._order_frozen is None:
            self._order_frozen = [r.effect.index for r in self.rows] + \
                [e.index for e in self.hidden]
        if self.values[row.effect.index] == value:
            return
        self.values[row.effect.index] = value
        self._values_changed(final=not dragging)

    def drag_finished(self):
        self._values_changed(final=True)

    def edit_value(self, row):
        e = row.effect
        var = e.var
        current = self.values[e.index]
        if e.kind == "discrete":
            items = list(var.values)
            idx = int(current) if np.isfinite(current) else 0
            text, ok = QInputDialog.getItem(self, "Nomogram Pro", f"{var.name}:", items,
                                            idx, False)
            if ok:
                self.values[e.index] = float(items.index(text))
        else:
            start = float(current) if np.isfinite(current) else (e.lo + e.hi) / 2
            text, ok = QInputDialog.getText(
                self, "Nomogram Pro", f"{var.name} (vacío o ? = desconocido):",
                text=E.format_value(var, start).replace("−", "-"))
            if ok:
                text = text.strip()
                if text in ("", "?"):
                    self.values[e.index] = np.nan
                else:
                    try:
                        self.values[e.index] = float(var.to_val(text.replace(",", ".")))
                    except (ValueError, TypeError):
                        try:
                            self.values[e.index] = float(var.to_val(text))
                        except (ValueError, TypeError):
                            return
        self._order_frozen = None
        self.relayout()
        self._values_changed(final=True)

    def _values_changed(self, final):
        if self.view_ is None:
            return
        if final:
            self._order_frozen = None if self.sort_index != SORT_INSTANCE \
                else self._order_frozen
            self._store_values()
            self._update_real(redraw=False)
            needs_layout = self._needs_relayout()
            if needs_layout:
                self._order_frozen = None
                self.relayout()
            else:
                self._draw_dynamic()
            self.commit.deferred()
        else:
            self._draw_dynamic()
            self._predict_timer.start()

    def _needs_relayout(self):
        if self.sort_index == SORT_INSTANCE:
            return True
        # una lineal con un valor fuera del rango amplía su escala
        for e in self.shown:
            v = self.values[e.index]
            if e.kind == "linear" and np.isfinite(v) and not e.lo <= v <= e.hi:
                return True
        return False

    def _store_values(self):
        a = self.analysis
        stored = {}
        for var, v in zip(a.domain.attributes, self.values):
            if v is None or not np.isfinite(v):
                stored[var.name] = None
            elif var.is_discrete:
                stored[var.name] = var.values[int(v)]
            else:
                stored[var.name] = float(v)
        self.saved_values = stored

    # ------------------------------------------------------------------ salidas
    @gui.deferred
    def commit(self):
        a = self.analysis
        if a is None or self.view_ is None or self.values is None:
            self.Outputs.features.send(None)
            self.Outputs.instance.send(None)
            self.Outputs.contributions.send(None)
            return
        shown = self.shown or self._ordered_effects()
        self.Outputs.features.send(AttributeList([e.var for e in shown]))
        self.Outputs.instance.send(self._instance_table())
        self.Outputs.contributions.send(self._contributions_table())

    def _instance_table(self):
        a = self.analysis
        cv = a.class_var
        domain = Domain(a.domain.attributes, cv)
        if self.real is None:
            y = np.nan
        elif a.kind == "classification":
            y = float(np.argmax(self.real))
        else:
            y = float(self.real)
        table = Table.from_numpy(domain, self.values[None, :].copy(), np.array([y]))
        table.name = "Instancia del nomograma"
        return table

    def _contributions_table(self):
        view = self.view_
        scale = self.scale or E.point_scale(view)
        order = self._ordered_effects()
        unit = "log-odds" if view.variant.space in ("score", "logit") else (
            "probabilidad" if view.variant.space == "prob" else "valor")
        cols = [ContinuousVariable("Puntos"),
                ContinuousVariable(f"Contribución ({unit})"),
                ContinuousVariable("Rango de puntos"),
                ContinuousVariable("Importancia en los datos (puntos)")]
        metas = [StringVariable("Atributo"), StringVariable("Valor")]
        domain = Domain(cols, metas=metas)
        X, M = [], []
        for e in order:
            comp = float(view.component(e, [self.values[e.index]])[0])
            gcomp = view.grid_component(e)
            lo = scale.mins.get(e.index, float(np.nanmin(gcomp)))
            X.append([(comp - lo) * scale.k, comp, np.ptp(gcomp) * scale.k,
                      view.importance(e) * scale.k])
            M.append([e.name, self._effect_label(e, self.values[e.index])])
        table = Table.from_numpy(domain, np.array(X, dtype=float).reshape(-1, 4),
                                 metas=np.array(M, dtype=object).reshape(-1, 2))
        table.name = "Contribuciones"
        return table

    # ------------------------------------------------------------------ informe
    def send_report(self):
        a = self.analysis
        if a is None or self.view_ is None:
            return
        html, _ = fidelity_summary(self.view_)
        items = [("Modelo", describe_model(a)),
                 ("Modo", "exacto" if a.mode == E.EXACT else "aproximado")]
        if a.kind == "classification":
            items.append(("Clase objetivo", a.class_var.values[self.view_.target]))
        else:
            items.append(("Objetivo", a.class_var.name))
        self.report_items(items)
        self.report_raw(f"<p>{html}</p>")
        self.report_plot()

    def sizeHint(self):
        return QSize(1150, 720)


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner
    _data = Table("heart_disease")
    WidgetPreview(OWNomogramPro).run(set_model=LogisticRegressionLearner()(_data),
                                     set_data=_data)
