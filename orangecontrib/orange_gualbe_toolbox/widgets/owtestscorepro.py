# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.
# Portions adapted from Orange Data Mining (Orange/widgets/evaluate/owtestandscore.py),
# Copyright (C) Bioinformatics Laboratory, University of Ljubljana, GPL-3.0-or-later.

"""Widget Test & Score Pro: Test & Score con comparaciones estadísticas rigurosas.

La lógica vive en _tsp_engine.py, _tsp_stats.py, _tsp_plots.py y _tsp_report.py, sin Qt
salvo la exportación a PDF. Aquí están la interfaz, el estado y el reparto de trabajo:
evaluar es caro y se hace una sola vez por modelo; analizar y dibujar no reentrenan.
"""
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Any, Optional

import numpy as np

from AnyQt.QtCore import (QAbstractTableModel, QDateTime, QModelIndex, QSize, QSortFilterProxyModel, Qt,
                          QTimer, QTimeZone)
from AnyQt.QtGui import QBrush, QColor, QFont, QImage, QPalette, QPixmap
from AnyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableView,
    QTabWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from Orange.base import Learner
from Orange.data import DiscreteVariable, Domain, Table, TimeVariable
from Orange.data.filter import HasClass
from Orange.data.table import DomainTransformationError
from Orange.evaluation import Results
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import gui
from Orange.widgets.evaluate.utils import learner_name, usable_scorers
from Orange.widgets.settings import ContextSetting, PerfectDomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Msg, MultiInput, Output, OWWidget

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_engine as engine
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_window as tw

N_FOLDS = [2, 3, 5, 10, 20]
N_REPEATS = [2, 3, 5, 10, 20, 50, 100]
SAMPLE_SIZES = [5, 10, 20, 25, 30, 33, 40, 50, 60, 66, 70, 75, 80, 90, 95]
ALPHAS = [0.01, 0.05, 0.1]
TARGET_AVERAGE = "(Promedio de las clases)"
DEBOUNCE_MS = 150
WINDOW_DEBOUNCE_MS = 500
# Por encima de estos entrenamientos, la ventana deslizante espera a que se pulse Evaluar.
LARGE_TRAININGS = 2000
NAN = float("nan")


# ------------------------------------------------------------------- tareas ----

@dataclass
class EvalRequest:
    data: Table
    test_data: Optional[Table]
    learners: list
    sampling: engine.Sampling
    preprocessor: Any
    workers: int


@dataclass
class AnalysisRequest:
    keys: list
    names: list
    raw_names: list
    results: list           # por objetivo: una lista de Results por modelo
    learners: list
    scorers: list
    sampling: engine.Sampling
    target: Optional[int]
    options: engine.TestOptions
    n_rows: int
    cached: Optional[list]  # ScoreSet de cada objetivo, si ya están calculados
    position: int = 0       # objetivo que se analiza
    target_names: tuple = ()
    send_results: bool = True


@dataclass
class EvalDone:
    outcomes: dict


@dataclass
class AnalysisDone:
    scoreset: engine.ScoreSet
    report: engine.TestReport
    predictions: Optional[Table]
    tables: dict
    fresh_scores: bool
    memory_error: bool = False
    scoresets: Optional[list] = None


@dataclass
class ReportDone:
    path: str
    layout: Any = None


def task_evaluate(request, state):
    state.set_status("Evaluando modelos…")
    outcomes = engine.evaluate(
        request.data, request.test_data, request.learners, request.sampling, request.preprocessor,
        request.workers, state.set_progress_value, state.is_interruption_requested)
    return EvalDone(outcomes)


def task_analyse(request, state):
    def check():
        if state.is_interruption_requested():
            raise engine.Cancelled()

    state.set_status("Calculando comparaciones…")
    scoresets = request.cached
    fresh = scoresets is None
    if fresh:
        scoresets = [engine.compute_scores(request.keys, request.names, results, request.scorers,
                                           request.sampling, request.n_rows, request.target, check,
                                           request.raw_names)
                     for results in request.results]
    scoreset = scoresets[request.position]
    report = engine.run_tests(scoreset, request.sampling, request.options, check)
    window = request.sampling.kind == engine.WINDOW
    plan = request.sampling.plan if window else None
    tables = dict(scores=engine.scores_table(scoreset, report),
                  folds=(engine.window_folds_table(scoresets, plan, request.target_names) if window
                         else engine.folds_table(scoreset)),
                  pairs=engine.pairs_table(report), ranks=engine.ranks_table(report),
                  horizons=engine.horizon_table(scoresets, request.target_names) if window else None)
    predictions, memory_error = None, False
    send = fresh or request.send_results
    if send:
        try:
            predictions = scoreset.combined.get_augmented_data(request.raw_names)
            if window:
                predictions = engine.add_window_columns(predictions, scoreset.combined, plan,
                                                        request.target_names[request.position])
        except MemoryError:
            memory_error = True
    return AnalysisDone(scoreset, report, predictions, tables, send, memory_error, scoresets)


def task_report(context, scoreset, report, path, state):
    state.set_status("Generando el informe…")
    from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_report
    document = _tsp_report.build_blocks(context, scoreset, report)
    if path.lower().endswith(".pdf"):
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_pdf
        return ReportDone(path, _tsp_pdf.render_pdf(document, path))
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(_tsp_report.render_html(document))
    return ReportDone(path)


# ------------------------------------------------------------------ modelos ----

class RowsModel(QAbstractTableModel):
    """Tabla genérica de solo lectura: textos, valores para ordenar y resaltado."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.headers = []
        self.rows = []
        self.sort_values = []
        self.highlight = set()
        self.tooltips = {}

    def set_content(self, headers, rows, sort_values, highlight=(), tooltips=None):
        self.beginResetModel()
        self.headers, self.rows, self.sort_values = headers, rows, sort_values
        self.highlight = set(highlight)
        self.tooltips = tooltips or {}
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role == Qt.DisplayRole:
            return self.rows[row][column]
        if role == Qt.UserRole:
            return self.sort_values[row][column]
        if role == Qt.TextAlignmentRole and column > 0:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if (row, column) in self.highlight:
            if role == Qt.FontRole:
                font = QFont()
                font.setBold(True)
                return font
            if role == Qt.BackgroundRole:
                color = QColor("#2a78d6")
                color.setAlphaF(0.20)
                return QBrush(color)
        if role == Qt.ToolTipRole:
            return self.tooltips.get((row, column))
        return None


class NumericSort(QSortFilterProxyModel):
    def lessThan(self, left, right):
        a, b = left.data(Qt.UserRole), right.data(Qt.UserRole)
        if isinstance(a, str) or isinstance(b, str):
            return str(a) < str(b)
        a = float("inf") if a is None or a != a else a
        b = float("inf") if b is None or b != b else b
        return a < b


class ChartPane(QWidget):
    """Muestra una figura de matplotlib como imagen y la redibuja al cambiar de tamaño."""

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.dirty = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.save_button = QToolButton(self, text="Guardar imagen…")
        self.save_button.clicked.connect(self.save_image)
        toolbar.addWidget(self.save_button)
        layout.addLayout(toolbar)
        self.scroll = QScrollArea(self)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll, 1)
        self._timer = QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.redraw)

    def invalidate(self):
        self.dirty = True
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self.dirty:
            self._timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.dirty = True
        self._timer.start()

    def theme(self):
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        base = self.palette().color(QPalette.Window)
        if base.lightness() < 110:
            return plots.Theme(base.name(), "#f0efec", "#c3c2b7", "#3b3b38", "#3987e5", "#d95926", "#6f6e68")
        return plots.Theme(background=base.name())

    def figure(self, width=None, height=None, theme=None):
        size = self.scroll.viewport().size()
        return self.builder(width or max(size.width(), 320) / 100,
                            height or max(size.height(), 200) / 100, theme or self.theme())

    def redraw(self):
        self.dirty = False
        figure = self.figure()
        if figure is None:
            self.canvas.clear()
            return
        ratio = self.devicePixelRatioF()
        figure.set_dpi(100 * ratio)
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height(physical=True)
        buffer = figure.canvas.buffer_rgba()
        image = QImage(buffer, width, height, QImage.Format_RGBA8888).copy()
        image.setDevicePixelRatio(ratio)
        self.canvas.setPixmap(QPixmap.fromImage(image))
        self.canvas.resize(int(width / ratio), int(height / ratio))

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", "grafico.png",
                                              "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.write_image(path)

    def write_image(self, path):
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        figure = self.figure(9, 5, plots.LIGHT)
        if figure is not None:
            plots.save_figure(figure, path)


def _utc():
    return QTimeZone(b"UTC")


class DateEdit(QDateTimeEdit):
    """Fecha y hora en UTC (como los valores de TimeVariable), limitada al rango de los datos."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCalendarPopup(True)
        self.setKeyboardTracking(False)
        if hasattr(self, "setTimeZone"):
            self.setTimeZone(_utc())
        else:
            self.setTimeSpec(Qt.UTC)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @staticmethod
    def to_qt(seconds):
        return QDateTime.fromMSecsSinceEpoch(int(round(seconds * 1000)), _utc())

    def seconds(self):
        return self.dateTime().toMSecsSinceEpoch() / 1000.0

    def configure(self, var, low, high):
        if var is None or not np.isfinite(low):
            return
        if var.have_date and var.have_time:
            self.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        elif var.have_time:
            self.setDisplayFormat("HH:mm:ss")
        else:
            self.setDisplayFormat("yyyy-MM-dd")
        self.blockSignals(True)
        self.setDateTimeRange(self.to_qt(low), self.to_qt(high))
        self.blockSignals(False)

    def set_seconds(self, seconds):
        self.blockSignals(True)
        self.setDateTime(self.to_qt(seconds))
        self.blockSignals(False)


@dataclass
class Slot:
    learner: Learner
    outcome: Optional[engine.Outcome] = None


class OWTestScorePro(OWWidget, ConcurrentWidgetMixin):
    name = "Test & Score Pro"
    description = ("Evalúa modelos como Test & Score y los compara con tests estadísticos "
                   "rigurosos, gráficos e informe exportable.")
    icon = "icons/owtestscorepro.svg"
    priority = 30
    keywords = ["test", "score", "cross validation", "validación cruzada", "comparar", "evaluar",
                "bayes", "wilcoxon", "friedman", "nemenyi", "mcnemar", "ventana deslizante",
                "sliding window", "series temporales", "time series", "diebold-mariano", "horizonte"]

    class Inputs:
        train_data = Input("Data", Table, default=True)
        test_data = Input("Test Data", Table)
        learner = MultiInput("Learner", Learner, filter_none=True)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        predictions = Output("Predictions", Table)
        evaluations_results = Output("Evaluation Results", Results)
        scores = Output("Scores", Table)
        fold_scores = Output("Fold Scores", Table)
        comparisons = Output("Pairwise Comparisons", Table)
        ranks = Output("Ranks", Table)
        best_learner = Output("Best Learner", Learner)
        horizon_scores = Output("Horizon Scores", Table)

    class Error(OWWidget.Error):
        train_data = Msg("{}")
        test_data = Msg("{}")
        sampling = Msg("{}")
        analysis = Msg("No se pudieron calcular las comparaciones: {}")
        memory = Msg("No hay memoria suficiente para crear la tabla de predicciones.")
        report = Msg("No se pudo guardar el informe: {}")

    class Warning(OWWidget.Warning):
        test_missing = Msg("Falta conectar los datos de test.")
        test_unused = Msg("Hay datos de test conectados que no se usan: elige «Sobre datos de test».")
        models_failed = Msg("{}")
        notes = Msg("{}")

    class Information(OWWidget.Information):
        missing_class = Msg("{}")
        test_transformed = Msg("Los datos de test se han transformado para coincidir con los de "
                               "entrenamiento.")
        report_saved = Msg("Informe guardado en {}")

    settings_version = 2
    settingsHandler = PerfectDomainContextHandler()

    KFOLD, FEATURE, SHUFFLE, LOO, ON_TRAIN, ON_TEST, WINDOW = range(7)

    # Ojo: ningún Setting puede llamarse como un método de QWidget/OWWidget.
    resampling = Setting(KFOLD)
    n_folds = Setting(3)  # 10 particiones
    cv_stratified = Setting(True)
    n_repeats = Setting(3)  # 10 repeticiones
    sample_size = Setting(9)  # 66 %
    shuffle_stratified = Setting(True)
    random_seed = Setting(42)
    parallel_models = Setting(True)
    fold_feature = ContextSetting(None)
    fold_feature_selected = ContextSetting(False)
    class_selection = ContextSetting(TARGET_AVERAGE)
    comparison_metric = Setting("")
    alpha_index = Setting(1)
    correction_method = Setting("holm")
    rope_auto = Setting(True)
    rope_value = Setting(0.01)
    hidden_metrics = Setting([])
    matrix_test = Setting("")
    details_test = Setting("")
    advanced_open = Setting(False)
    current_tab = Setting(0)
    dm_horizon = Setting(0)  # 0: automático
    # Ventana deslizante. Las fechas y la variable de tiempo dependen de los datos.
    window_time_var = ContextSetting(None)
    window_start_mode = Setting(tw.START_FIRST)
    window_start_row = Setting(1)
    window_start_date = ContextSetting(NAN)
    window_initial_mode = Setting(tw.INITIAL_PERCENT)
    window_initial_percent = Setting(50)
    window_initial_rows = Setting(100)
    window_initial_date = ContextSetting(NAN)
    window_advance = Setting(tw.EXPANDING)
    window_unit = Setting(tw.UNIT_ROWS)
    window_time_unit = Setting(tw.DEFAULT_TIME_UNIT)
    window_test = Setting(10)
    window_gap = Setting(0)
    window_step = Setting(10)
    window_step_same = Setting(True)
    window_end_mode = Setting(tw.END_ALL)
    window_end_date = ContextSetting(NAN)
    window_last = Setting(tw.LAST_DISCARD)
    window_min_train = Setting(10)
    window_max_iterations = Setting(200)
    horizon_target = ContextSetting("")

    @classmethod
    def migrate_settings(cls, settings, version):
        # La 0.4.0 inserta las pestañas Ventanas, Iteraciones y Horizontes tras Puntuaciones.
        if version is not None and version < 2 and settings.get("current_tab", 0) >= 1:
            settings["current_tab"] = settings["current_tab"] + 3

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self.test_data = None
        self.preprocessor = None
        self.scorers = []
        self.learners = {}
        self._learner_inputs = []
        self._id_generator = count()
        self._need_refresh = False
        self._running = None
        self._scoreset = None
        self._analysis = None
        self._sampling_used = None
        self._sampling_notes = []
        self._pending_analysis = False
        self._report_layout = None
        self._dropped_rows = ""
        self._scoresets = None
        self._preview_plan = None
        self._large_confirmed = False
        self._send_results = True
        self._window_timer = QTimer(self, singleShot=True, interval=WINDOW_DEBOUNCE_MS)
        self._window_timer.timeout.connect(self._window_evaluate)
        self._after_task = QTimer(self, singleShot=True, interval=0)
        self._after_task.timeout.connect(self._start_analysis)
        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._debounce.timeout.connect(self._start_analysis)
        self._build_controls()
        self._build_main_area()
        self._sync_advanced()
        self._sync_window_controls()

    # ------------------------------------------------------ construcción ----

    def _build_controls(self):
        box = gui.vBox(self.controlArea, "Muestreo")
        radio = gui.radioButtons(box, self, "resampling", callback=self._sampling_chosen)
        gui.appendRadioButton(radio, "Validación cruzada")
        inner = gui.indentedBox(radio)
        gui.comboBox(inner, self, "n_folds", label="Particiones:", items=[str(x) for x in N_FOLDS],
                     orientation=Qt.Horizontal, callback=self._kfold_changed)
        gui.checkBox(inner, self, "cv_stratified", "Estratificada", callback=self._kfold_changed)
        gui.appendRadioButton(radio, "Por característica")
        inner = gui.indentedBox(radio)
        self.feature_model = DomainModel(order=DomainModel.METAS, valid_types=DiscreteVariable)
        self.features_combo = gui.comboBox(inner, self, "fold_feature", model=self.feature_model,
                                           orientation=Qt.Horizontal, searchable=True,
                                           callback=self._feature_changed)
        gui.appendRadioButton(radio, "Muestreo aleatorio")
        inner = gui.indentedBox(radio)
        gui.comboBox(inner, self, "n_repeats", label="Repeticiones:",
                     items=[str(x) for x in N_REPEATS], orientation=Qt.Horizontal,
                     callback=self._shuffle_changed)
        gui.comboBox(inner, self, "sample_size", label="Entrenamiento:",
                     items=[f"{x} %" for x in SAMPLE_SIZES], orientation=Qt.Horizontal,
                     callback=self._shuffle_changed)
        gui.checkBox(inner, self, "shuffle_stratified", "Estratificado", callback=self._shuffle_changed)
        gui.appendRadioButton(radio, "Dejar uno fuera")
        gui.appendRadioButton(radio, "Sobre datos de entrenamiento")
        gui.appendRadioButton(radio, "Sobre datos de test")
        button = gui.appendRadioButton(radio, "Ventana deslizante (series temporales)")
        button.setToolTip("Entrena con el pasado y evalúa con el bloque siguiente, respetando el orden "
                          "temporal. Admite varias variables objetivo (multi-horizonte).")
        self._build_window_box()

        box = gui.vBox(self.controlArea, "Comparar")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.horizon_label = QLabel("Objetivo:")
        self.horizon_combo = QComboBox(self)
        self.horizon_combo.setToolTip("Variable objetivo (horizonte) que se analiza en las pestañas y "
                                      "en las salidas Predictions y Evaluation Results.")
        self.horizon_combo.activated[int].connect(self._horizon_target_chosen)
        self.metric_combo = QComboBox(self)
        self.metric_combo.activated[int].connect(self._metric_chosen)
        self.target_label = QLabel("Clase objetivo:")
        self.target_combo = QComboBox(self)
        self.target_combo.activated[int].connect(self._target_chosen)
        self.alpha_combo = QComboBox(self)
        self.alpha_combo.addItems([f"{a:g}".replace(".", ",") for a in ALPHAS])
        self.alpha_combo.setCurrentIndex(self.alpha_index)
        self.alpha_combo.activated[int].connect(self._alpha_chosen)
        grid.addWidget(self.horizon_label, 0, 0)
        grid.addWidget(self.horizon_combo, 0, 1)
        grid.addWidget(QLabel("Métrica:"), 1, 0)
        grid.addWidget(self.metric_combo, 1, 1)
        grid.addWidget(self.target_label, 2, 0)
        grid.addWidget(self.target_combo, 2, 1)
        grid.addWidget(QLabel("Significación α:"), 3, 0)
        grid.addWidget(self.alpha_combo, 3, 1)
        grid.setColumnStretch(1, 1)
        self.horizon_label.setVisible(False)
        self.horizon_combo.setVisible(False)

        self.advanced_toggle = QToolButton(self, text="Opciones avanzadas", checkable=True,
                                           toolButtonStyle=Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.setChecked(bool(self.advanced_open))
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.controlArea.layout().addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget(self)
        grid = QGridLayout(self.advanced_panel)
        grid.setContentsMargins(4, 0, 0, 0)
        self.controlArea.layout().addWidget(self.advanced_panel)
        self.correction_combo = QComboBox(self)
        for key, label in st.CORRECTIONS.items():
            self.correction_combo.addItem(label, key)
        self.correction_combo.setCurrentIndex(max(self.correction_combo.findData(self.correction_method), 0))
        self.correction_combo.activated[int].connect(self._correction_chosen)
        self.rope_check = gui.checkBox(None, self, "rope_auto", "ROPE automático",
                                       callback=self._rope_changed, addToLayout=False)
        self.rope_spin = QDoubleSpinBox(self, minimum=0, maximum=1e12, decimals=4, singleStep=0.005)
        self.rope_spin.setValue(self.rope_value)
        self.rope_spin.setKeyboardTracking(False)
        self.rope_spin.valueChanged.connect(self._rope_spin_changed)
        self.seed_spin = QSpinBox(self, minimum=0, maximum=999999)
        self.seed_spin.setValue(self.random_seed)
        self.seed_spin.setKeyboardTracking(False)
        self.seed_spin.valueChanged.connect(self._seed_changed)
        self.parallel_check = gui.checkBox(None, self, "parallel_models", "Entrenar en paralelo",
                                           callback=self._parallel_changed, addToLayout=False)
        grid.addWidget(QLabel("Corrección múltiple:"), 0, 0)
        grid.addWidget(self.correction_combo, 0, 1)
        grid.addWidget(self.rope_check, 1, 0)
        grid.addWidget(self.rope_spin, 1, 1)
        grid.addWidget(QLabel("Semilla:"), 2, 0)
        grid.addWidget(self.seed_spin, 2, 1)
        self.dm_spin = QSpinBox(self, minimum=0, maximum=1000)
        self.dm_spin.setSpecialValueText("Automático")
        self.dm_spin.setValue(self.dm_horizon)
        self.dm_spin.setKeyboardTracking(False)
        self.dm_spin.setToolTip("Horizonte h de Diebold-Mariano en la ventana deslizante. Automático: 1 con "
                                "un objetivo; con varios, la posición del objetivo analizado.")
        self.dm_spin.valueChanged.connect(self._dm_horizon_changed)
        grid.addWidget(QLabel("Horizonte DM:"), 3, 0)
        grid.addWidget(self.dm_spin, 3, 1)
        grid.addWidget(self.parallel_check, 4, 0, 1, 2)

        gui.rubber(self.controlArea)
        self.report_button2 = QPushButton("Exportar informe…", self)
        self.report_button2.clicked.connect(self.export_report_dialog)
        self.report_button2.setEnabled(False)
        self.controlArea.layout().addWidget(self.report_button2)
        self.cancel_button = QPushButton("Cancelar", self)
        self.cancel_button.clicked.connect(self.cancel_evaluation)
        self.cancel_button.setVisible(False)
        self.controlArea.layout().addWidget(self.cancel_button)

    def _build_window_box(self):
        """Opciones de la ventana deslizante, en el orden en que se piensan."""
        self.window_box = gui.vBox(self.controlArea, "Ventana deslizante")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(4)
        self.window_box.layout().addLayout(grid)

        def combo(items, setting, tip):
            widget = QComboBox(self)
            widget.addItems(items)
            widget.setCurrentIndex(min(getattr(self, setting), len(items) - 1))
            widget.setToolTip(tip)
            widget.activated[int].connect(lambda index: self._window_option(setting, index))
            return widget

        def spin(low, high, setting, tip, suffix=""):
            widget = QSpinBox(self, minimum=low, maximum=high)
            widget.setValue(int(getattr(self, setting)))
            widget.setSuffix(suffix)
            widget.setKeyboardTracking(False)
            widget.setToolTip(tip)
            widget.valueChanged.connect(lambda value: self._window_option(setting, value))
            return widget

        def date(setting, tip):
            widget = DateEdit(self)
            widget.setToolTip(tip)
            widget.dateTimeChanged.connect(lambda _: self._window_option(setting, widget.seconds()))
            return widget

        def label(text, tip=""):
            widget = QLabel(text)
            widget.setToolTip(tip)
            return widget

        self.time_model = DomainModel(order=DomainModel.MIXED, valid_types=TimeVariable,
                                      placeholder="Orden de las filas")
        self.time_combo = QComboBox(self)
        self.time_combo.setModel(self.time_model)
        self.time_combo.setToolTip("Variable que da el orden temporal. Sin ella se usa el orden de las filas "
                                   "y no se pueden usar fechas.")
        self.time_combo.activated[int].connect(self._time_var_chosen)
        tip = "Primera fila del estudio: las anteriores quedan fuera."
        self.start_combo = combo(["Primera fila", "Desde la fila N", "Desde la fecha"], "window_start_mode", tip)
        self.start_row_spin = spin(1, 10 ** 9, "window_start_row", "Número de fila, en el orden temporal.")
        self.start_date = date("window_start_date", "Primera fecha del estudio.")
        tip = "Tamaño del primer entrenamiento."
        self.initial_combo = combo(["Porcentaje", "Filas", "Hasta la fecha"], "window_initial_mode", tip)
        self.initial_percent_spin = spin(1, 99, "window_initial_percent", "Porcentaje de las filas del estudio.",
                                         " %")
        self.initial_rows_spin = spin(1, 10 ** 9, "window_initial_rows", "Filas del primer entrenamiento.",
                                      " filas")
        self.initial_date = date("window_initial_date", "Última fecha (incluida) del primer entrenamiento.")
        self.advance_combo = combo(["Creciente: inicio fijo", "Deslizante: tamaño fijo"], "window_advance",
                                   "Creciente: el entrenamiento empieza siempre en el mismo sitio y crece.\n"
                                   "Deslizante: conserva su tamaño y su inicio avanza con cada iteración.")
        self.unit_combo = combo(["Filas", "Tiempo"], "window_unit",
                                "Unidad del test, el hueco y el paso.")
        self.time_unit_combo = combo([name for name, _ in tw.TIME_UNITS], "window_time_unit",
                                     "Unidad de tiempo. Meses y años siguen el calendario.")
        self.test_spin = spin(1, 10 ** 7, "window_test", "Tamaño de cada bloque de test, justo después del "
                                                         "entrenamiento (o del hueco).")
        self.gap_spin = spin(0, 10 ** 7, "window_gap", "Separación entre el final del entrenamiento y el "
                                                       "test, para que el modelo no vea datos demasiado "
                                                       "cercanos.")
        self.step_spin = spin(1, 10 ** 7, "window_step", "Cuánto avanza la ventana en cada iteración.")
        self.step_same_check = QCheckBox("igual al test", self)
        self.step_same_check.setChecked(bool(self.window_step_same))
        self.step_same_check.toggled.connect(lambda value: self._window_option("window_step_same", value))
        self.end_combo = combo(["Hasta el final", "Hasta la fecha"], "window_end_mode",
                               "Última fila que puede entrar en un test.")
        self.end_date = date("window_end_date", "Última fecha (incluida) que puede entrar en un test.")
        self.last_combo = combo(["Descartar", "Usar"], "window_last",
                                "Si el último test no llega a su tamaño: descartarlo (todos los tests iguales) "
                                "o usarlo más corto.")
        self.min_train_spin = spin(1, 10 ** 7, "window_min_train",
                                   "Las iteraciones con menos filas de entrenamiento se omiten.", " filas")
        self.max_iterations_spin = spin(1, 100000, "window_max_iterations",
                                        "Si salen más iteraciones, el widget lo indica y no evalúa.")

        rows = [
            (label("Ordenar por:"), self.time_combo, None),
            (label("Inicio:"), self.start_combo, (self.start_row_spin, self.start_date)),
            (label("Entrenamiento:"), self.initial_combo,
             (self.initial_percent_spin, self.initial_rows_spin, self.initial_date)),
            (label("Avance:"), self.advance_combo, None),
            (label("Medir en:"), self.unit_combo, (self.time_unit_combo,)),
            (label("Test:"), self.test_spin, None),
            (label("Hueco:"), self.gap_spin, None),
            (label("Paso:"), self.step_spin, (self.step_same_check,)),
            (label("Fin:"), self.end_combo, (self.end_date,)),
            (label("Test incompleto:"), self.last_combo, None),
            (label("Mín. entrenamiento:"), self.min_train_spin, None),
            (label("Máx. iteraciones:"), self.max_iterations_spin, None),
        ]
        for row, (text, main, extras) in enumerate(rows):
            grid.addWidget(text, row, 0)
            if extras is None:
                grid.addWidget(main, row, 1, 1, 2)
            else:
                grid.addWidget(main, row, 1)
                for extra in extras:
                    grid.addWidget(extra, row, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.window_summary = QLabel(self)
        self.window_summary.setWordWrap(True)
        self.window_box.layout().addWidget(self.window_summary)
        self.evaluate_button = QPushButton("Evaluar", self)
        self.evaluate_button.setToolTip("La evaluación es grande: pulsa para empezarla.")
        self.evaluate_button.clicked.connect(self._confirm_large)
        self.evaluate_button.setVisible(False)
        self.window_box.layout().addWidget(self.evaluate_button)

    def _build_main_area(self):
        layout = self.mainArea.layout()
        self.banner = QLabel("Conecta datos y al menos un modelo.")
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.RichText)
        self.banner.setMargin(8)
        self.banner.setStyleSheet("background: rgba(42,120,214,0.12); border-left: 4px solid #2a78d6;")
        layout.addWidget(self.banner)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        scores_page = QWidget(self)
        page_layout = QVBoxLayout(scores_page)
        bar = QHBoxLayout()
        bar.addStretch(1)
        self.metrics_button = QToolButton(self, text="Métricas visibles", popupMode=QToolButton.InstantPopup)
        self.metrics_menu = QMenu(self.metrics_button)
        self.metrics_button.setMenu(self.metrics_menu)
        bar.addWidget(self.metrics_button)
        page_layout.addLayout(bar)
        self.scores_model = RowsModel(self)
        self.scores_proxy = NumericSort(self)
        self.scores_proxy.setSourceModel(self.scores_model)
        self.scores_view = self._make_table(self.scores_proxy)
        page_layout.addWidget(self.scores_view)
        self.tabs.addTab(scores_page, "Puntuaciones")

        self.windows_page = QWidget(self)
        page_layout = QVBoxLayout(self.windows_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_windows = ChartPane(self._draw_windows, self)
        page_layout.addWidget(self.chart_windows, 3)
        self.iterations_model = RowsModel(self)
        self.iterations_proxy = NumericSort(self)
        self.iterations_proxy.setSourceModel(self.iterations_model)
        self.iterations_view = self._make_table(self.iterations_proxy)
        self.iterations_view.sortByColumn(0, Qt.AscendingOrder)
        page_layout.addWidget(self.iterations_view, 2)
        self.tabs.addTab(self.windows_page, "Ventanas")
        self.chart_iterations = ChartPane(self._draw_iterations, self)
        self.tabs.addTab(self.chart_iterations, "Iteraciones")
        self.chart_horizons = ChartPane(self._draw_horizons, self)
        self.tabs.addTab(self.chart_horizons, "Horizontes")

        self.chart_scores = ChartPane(self._draw_scores, self)
        self.tabs.addTab(self.chart_scores, "Comparación")
        self.chart_cd = ChartPane(self._draw_cd, self)
        self.tabs.addTab(self.chart_cd, "Rangos")

        matrix_page = QWidget(self)
        page_layout = QVBoxLayout(matrix_page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Test:"))
        self.matrix_combo = QComboBox(self)
        self.matrix_combo.activated[int].connect(self._matrix_test_chosen)
        row.addWidget(self.matrix_combo)
        row.addStretch(1)
        page_layout.addLayout(row)
        self.chart_matrix = ChartPane(self._draw_matrix, self)
        page_layout.addWidget(self.chart_matrix, 1)
        self.tabs.addTab(matrix_page, "Matriz")

        self.chart_bayes = ChartPane(self._draw_bayes, self)
        self.tabs.addTab(self.chart_bayes, "Bayesiano")
        self.chart_times = ChartPane(self._draw_times, self)
        self.tabs.addTab(self.chart_times, "Tiempos")

        details_page = QWidget(self)
        page_layout = QVBoxLayout(details_page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Test:"))
        self.details_combo = QComboBox(self)
        self.details_combo.activated[int].connect(self._details_test_chosen)
        row.addWidget(self.details_combo)
        row.addStretch(1)
        page_layout.addLayout(row)
        self.details_model = RowsModel(self)
        self.details_proxy = NumericSort(self)
        self.details_proxy.setSourceModel(self.details_model)
        self.details_view = self._make_table(self.details_proxy)
        page_layout.addWidget(self.details_view, 2)
        self.notes_browser = QTextBrowser(self)
        page_layout.addWidget(self.notes_browser, 1)
        self.tabs.addTab(details_page, "Tests y supuestos")

        self.tabs.setCurrentIndex(min(self.current_tab, self.tabs.count() - 1))
        self.tabs.currentChanged.connect(self._tab_changed)
        self.charts = (self.chart_scores, self.chart_cd, self.chart_matrix, self.chart_bayes,
                       self.chart_times, self.chart_windows, self.chart_iterations, self.chart_horizons)

    def _make_table(self, model):
        view = QTableView(self)
        view.setModel(model)
        view.setSortingEnabled(True)
        view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setWordWrap(False)
        view.verticalHeader().setVisible(False)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        view.horizontalHeader().setStretchLastSection(False)
        return view

    def sizeHint(self):
        return QSize(1120, 700)

    def _sync_advanced(self):
        self.advanced_panel.setVisible(bool(self.advanced_open))
        self.advanced_toggle.setArrowType(Qt.DownArrow if self.advanced_open else Qt.RightArrow)
        self.rope_spin.setEnabled(not self.rope_auto)

    # ---------------------------------------------------------------- entradas ----

    @Inputs.train_data
    def set_train_data(self, data):
        self.cancel_evaluation(quiet=True)
        self.Error.train_data.clear()
        self.Information.missing_class.clear()
        self._dropped_rows = ""
        if data is not None:
            problem = self._data_problem(data)
            if problem:
                self.Error.train_data(problem)
                data = None
        if data is not None:
            # Como el nativo: fuera las filas sin valor en la clase, aquí en cualquiera de los objetivos,
            # antes de formar particiones o ventanas. Así todos los objetivos usan las mismas filas.
            before = len(data)
            data, counts = tw.drop_missing_targets(data)
            if counts:
                self._dropped_rows = tw.dropped_message(before - len(data), counts)
                self.Information.missing_class(self._dropped_rows)
            if not len(data):
                self.Error.train_data("Ninguna fila tiene valor en " + (
                    "todas las variables objetivo." if len(data.domain.class_vars) > 1
                    else "la variable objetivo."))
                data = None
        self.closeContext()
        self.data = data
        self._update_scorers()
        self._update_target_combo()
        self.feature_model.set_domain(data.domain if data is not None else None)
        self.fold_feature = self.feature_model[0] if self.feature_model else None
        self.time_model.set_domain(data.domain if data is not None else None)
        self.window_time_var = self._default_time_var(data)
        self.window_start_date = self.window_initial_date = self.window_end_date = NAN
        self.horizon_target = ""
        if data is not None:
            self.openContext(data.domain)
            if self.fold_feature_selected and self.feature_model:
                self.resampling = self.FEATURE
        self._select_target_from_setting()
        self._update_horizon_combo()
        self._sync_time_combo()
        self._set_date_ranges()
        self._sync_window_values()
        self._invalidate()
        self._sync_window_controls()
        self._update_window_preview()

    @staticmethod
    def _data_problem(data):
        if not len(data):
            return "La tabla de entrenamiento está vacía."
        if not data.domain.class_vars:
            return "Los datos no tienen variable objetivo."
        if not data.domain.attributes:
            return "Los datos no tienen atributos con los que aprender."
        if len(data.domain.class_vars) > 1:
            # Solo se evalúan con la ventana deslizante, que revisa cada objetivo al formar las ventanas.
            if len({var.is_discrete for var in data.domain.class_vars}) > 1:
                return tw.check_targets(data)
            return ""
        y = data.Y
        if np.isnan(y).all():
            return "La variable objetivo no tiene ningún valor."
        if data.domain.has_discrete_class and len(np.unique(y[~np.isnan(y)])) < 2:
            return "La variable objetivo solo tiene un valor."
        if not data.domain.attributes:
            return "Los datos no tienen atributos con los que aprender."
        return ""

    @Inputs.test_data
    def set_test_data(self, data):
        self.cancel_evaluation(quiet=True)
        self.Error.test_data.clear()
        if data is not None and not len(data):
            self.Error.test_data("La tabla de test está vacía.")
            data = None
        elif data is not None and not data.domain.class_vars:
            self.Error.test_data("Los datos de test no tienen variable objetivo.")
            data = None
        if data is not None and np.isnan(data.Y).any():
            data = HasClass()(data)
            if not len(data):
                self.Error.test_data("Ninguna fila de test tiene valor en la variable objetivo.")
                data = None
        self.test_data = data
        if self.resampling == self.ON_TEST:
            self._invalidate()
        else:
            self._need_refresh = True

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        self._invalidate()

    @Inputs.learner
    def set_learner(self, index, learner):
        key, _ = self._learner_inputs[index]
        self._learner_inputs[index] = (key, learner)
        self.learners[key] = Slot(learner)
        self._invalidate([key])

    @Inputs.learner.insert
    def insert_learner(self, index, learner):
        key = next(self._id_generator)
        self._learner_inputs.insert(index, (key, learner))
        self.learners[key] = Slot(learner)
        self.learners = {k: self.learners[k] for k, _ in self._learner_inputs}
        self._invalidate([key])

    @Inputs.learner.remove
    def remove_learner(self, index):
        key, _ = self._learner_inputs.pop(index)
        self.learners.pop(key)
        self._invalidate([])

    def handleNewSignals(self):
        if self._need_refresh:
            self._refresh()

    # ----------------------------------------------------------------- controles ----

    def _single_target_domain(self):
        domain = self.data.domain
        if len(domain.class_vars) <= 1:
            return domain
        return Domain(domain.attributes, domain.class_vars[0], domain.metas)

    def _update_scorers(self):
        self.scorers = usable_scorers(self._single_target_domain()) if self.data is not None else []
        self.metric_combo.clear()
        for scorer in self.scorers:
            self.metric_combo.addItem(scorer.long_name or scorer.name, scorer.__name__)
        names = [scorer.__name__ for scorer in self.scorers]
        if names and self.comparison_metric not in names:
            self.comparison_metric = names[0]
        if names:
            self.metric_combo.setCurrentIndex(names.index(self.comparison_metric))
        self.metrics_menu.clear()
        for scorer in self.scorers:
            action = self.metrics_menu.addAction(scorer.long_name or scorer.name)
            action.setCheckable(True)
            action.setChecked(scorer.__name__ not in self.hidden_metrics)
            action.setData(scorer.__name__)
            action.toggled.connect(self._metric_visibility_changed)

    def _update_target_combo(self):
        self.target_combo.clear()
        discrete = self.data is not None and self.data.domain.has_discrete_class
        self.target_combo.setVisible(bool(discrete))
        self.target_label.setVisible(bool(discrete))
        if discrete:
            self.target_combo.addItems((TARGET_AVERAGE,) + tuple(self.data.domain.class_var.values))

    def _select_target_from_setting(self):
        if self.data is None or not self.data.domain.has_discrete_class:
            return
        index = self.target_combo.findText(self.class_selection)
        if index < 0:
            index = 0
            self.class_selection = TARGET_AVERAGE
        self.target_combo.setCurrentIndex(index)

    def _target_index(self):
        if self.data is None or not self.data.domain.has_discrete_class \
                or self.class_selection == TARGET_AVERAGE:
            return None
        return self.data.domain.class_var.values.index(self.class_selection)

    def _kfold_changed(self):
        self.resampling = self.KFOLD
        self._parameters_changed()

    def _shuffle_changed(self):
        self.resampling = self.SHUFFLE
        self._parameters_changed()

    def _feature_changed(self):
        self.resampling = self.FEATURE
        self._parameters_changed()

    def _parameters_changed(self):
        self.fold_feature_selected = self.resampling == self.FEATURE
        self._sync_window_controls()
        self._update_window_preview()
        self._invalidate()
        self._refresh()

    def _sampling_chosen(self):
        self._parameters_changed()
        if self.resampling == self.WINDOW:
            self.tabs.setCurrentWidget(self.windows_page)

    def _seed_changed(self, value):
        self.random_seed = value
        self._parameters_changed()

    def _parallel_changed(self):
        pass  # solo afecta a las próximas evaluaciones

    def _toggle_advanced(self, opened):
        self.advanced_open = bool(opened)
        self._sync_advanced()

    def _metric_chosen(self, index):
        self.comparison_metric = self.metric_combo.itemData(index)
        self._options_changed()

    def _target_chosen(self, index):
        self.class_selection = self.target_combo.itemText(index)
        self._scoreset = None
        self._scoresets = None
        self._options_changed()

    def _alpha_chosen(self, index):
        self.alpha_index = index
        self._options_changed()

    def _correction_chosen(self, index):
        self.correction_method = self.correction_combo.itemData(index)
        self._options_changed()

    def _rope_changed(self):
        self.rope_spin.setEnabled(not self.rope_auto)
        self._options_changed()

    def _rope_spin_changed(self, value):
        self.rope_value = value
        self._options_changed()

    def _options_changed(self):
        if self._successful_slots():
            self._debounce.start()

    def _metric_visibility_changed(self, checked):
        name = self.sender().data()
        hidden = set(self.hidden_metrics)
        if checked:
            hidden.discard(name)
        else:
            hidden.add(name)
        self.hidden_metrics = sorted(hidden)
        self._fill_scores_table()

    def _tab_changed(self, index):
        self.current_tab = index

    def _matrix_test_chosen(self, index):
        self.matrix_test = self.matrix_combo.itemData(index)
        self.chart_matrix.invalidate()

    def _details_test_chosen(self, index):
        self.details_test = self.details_combo.itemData(index)
        self._fill_details()

    # -------------------------------------------------------- ventana deslizante ----

    @staticmethod
    def _default_time_var(data):
        if data is None:
            return None
        chosen = getattr(data, "time_variable", None)  # tablas de Orange3-Timeseries
        if isinstance(chosen, TimeVariable):
            return chosen
        domain = data.domain
        return next((var for var in domain.attributes + domain.metas + domain.class_vars
                     if isinstance(var, TimeVariable)), None)

    def _sync_time_combo(self):
        if self.window_time_var not in self.time_model:
            self.window_time_var = None
        self.time_combo.setCurrentIndex(self.time_model.indexOf(self.window_time_var))

    def _time_var_chosen(self, index):
        self.window_time_var = self.time_model[index]
        self.window_start_date = self.window_initial_date = self.window_end_date = NAN
        self._set_date_ranges()
        self._sync_window_controls()
        self._window_changed()

    def _time_values(self):
        if self.data is None or self.window_time_var is None:
            return None
        values = tw.column(self.data, self.window_time_var)
        values = np.sort(values[np.isfinite(values)])
        return values if len(values) else None

    def _set_date_ranges(self):
        """Limita las fechas al rango de los datos y elige valores razonables si faltan."""
        values = self._time_values()
        if values is None:
            return
        low, high = float(values[0]), float(values[-1])
        defaults = (("window_start_date", self.start_date, low),
                    ("window_initial_date", self.initial_date, float(values[len(values) // 2])),
                    ("window_end_date", self.end_date, high))
        for setting, editor, default in defaults:
            editor.configure(self.window_time_var, low, high)
            value = getattr(self, setting)
            if not isinstance(value, float) or not np.isfinite(value):
                value = default
            value = min(max(value, low), high)
            setattr(self, setting, value)
            editor.set_seconds(value)

    def _sync_window_values(self):
        """Lleva los ajustes (p. ej. los de un workflow guardado) a los controles."""
        for widget, setting in ((self.start_combo, "window_start_mode"), (self.initial_combo, "window_initial_mode"),
                                (self.advance_combo, "window_advance"), (self.unit_combo, "window_unit"),
                                (self.time_unit_combo, "window_time_unit"), (self.end_combo, "window_end_mode"),
                                (self.last_combo, "window_last")):
            widget.setCurrentIndex(int(getattr(self, setting)))
        for widget, setting in ((self.start_row_spin, "window_start_row"),
                                (self.initial_percent_spin, "window_initial_percent"),
                                (self.initial_rows_spin, "window_initial_rows"), (self.test_spin, "window_test"),
                                (self.gap_spin, "window_gap"), (self.step_spin, "window_step"),
                                (self.min_train_spin, "window_min_train"),
                                (self.max_iterations_spin, "window_max_iterations"),
                                (self.dm_spin, "dm_horizon")):
            widget.blockSignals(True)
            widget.setValue(int(getattr(self, setting)))
            widget.blockSignals(False)
        self.step_same_check.blockSignals(True)
        self.step_same_check.setChecked(bool(self.window_step_same))
        self.step_same_check.blockSignals(False)

    def _window_spec(self):
        return tw.WindowSpec(
            time_var=self.window_time_var, start_mode=self.window_start_mode, start_row=self.window_start_row,
            start_date=self.window_start_date, initial_mode=self.window_initial_mode,
            initial_percent=self.window_initial_percent, initial_rows=self.window_initial_rows,
            initial_date=self.window_initial_date, advance=self.window_advance, unit=self.window_unit,
            time_unit=self.window_time_unit, test_size=self.window_test, gap=self.window_gap,
            step=self.window_step, step_same=bool(self.window_step_same), end_mode=self.window_end_mode,
            end_date=self.window_end_date, last=self.window_last, min_train=self.window_min_train,
            max_iterations=self.window_max_iterations)

    def _window_option(self, setting, value):
        setattr(self, setting, value)
        self._sync_window_controls()
        self._window_changed()

    def _window_changed(self):
        """La vista previa cambia al momento; la evaluación, al dejar de tocar opciones."""
        if self.resampling != self.WINDOW:
            return
        self._update_window_preview()
        self._invalidate()
        if self.data is not None and self.learners:
            self.banner.setText("Opciones de la ventana cambiadas: se evaluará en un momento…")
        self._window_timer.start()

    def _window_evaluate(self):
        self._refresh()

    def _confirm_large(self):
        self._large_confirmed = True
        self.evaluate_button.setVisible(False)
        self._refresh()

    def _sync_window_controls(self):
        window = self.resampling == self.WINDOW
        self.window_box.setVisible(window)
        has_time = self.window_time_var is not None
        self.start_row_spin.setVisible(self.window_start_mode == tw.START_ROW)
        self.start_date.setVisible(self.window_start_mode == tw.START_DATE)
        self.initial_percent_spin.setVisible(self.window_initial_mode == tw.INITIAL_PERCENT)
        self.initial_rows_spin.setVisible(self.window_initial_mode == tw.INITIAL_ROWS)
        self.initial_date.setVisible(self.window_initial_mode == tw.INITIAL_DATE)
        self.end_date.setVisible(self.window_end_mode == tw.END_DATE)
        self.time_unit_combo.setVisible(self.window_unit == tw.UNIT_TIME)
        self.step_spin.setEnabled(not self.window_step_same)
        for combo, index in ((self.start_combo, tw.START_DATE), (self.initial_combo, tw.INITIAL_DATE),
                             (self.end_combo, tw.END_DATE), (self.unit_combo, tw.UNIT_TIME)):
            item = combo.model().item(index)
            if item is not None:
                item.setEnabled(has_time)
                item.setToolTip("" if has_time else "Necesita una variable de tiempo en «Ordenar por».")
        if self.window_unit == tw.UNIT_TIME:
            name = tw.TIME_UNITS[self.window_time_unit][0]
            suffix = f" {name}"
        else:
            suffix = " filas"
        for spin in (self.test_spin, self.gap_spin, self.step_spin):
            spin.setSuffix(suffix)
        if self.window_step_same and self.step_spin.value() != self.window_test:
            self.step_spin.blockSignals(True)
            self.step_spin.setValue(self.window_test)
            self.step_spin.blockSignals(False)
        multi = self.data is not None and len(self.data.domain.class_vars) > 1
        for page, visible in ((self.windows_page, window), (self.chart_iterations, window),
                              (self.chart_horizons, window and multi)):
            index = self.tabs.indexOf(page)
            if hasattr(self.tabs, "setTabVisible"):
                self.tabs.setTabVisible(index, visible)
            else:
                self.tabs.setTabEnabled(index, visible)
        visible_multi = window and multi
        self.horizon_label.setVisible(visible_multi)
        self.horizon_combo.setVisible(visible_multi)

    def _update_window_preview(self):
        plan = None
        if self.data is not None and self.resampling == self.WINDOW:
            plan = tw.plan_windows(self.data, self._window_spec())
        self._show_plan(plan)

    def _show_plan(self, plan):
        self._preview_plan = plan
        if plan is None:
            self.window_summary.setText("")
            self.iterations_model.set_content([], [], [])
        else:
            n_targets = len(self.data.domain.class_vars) if self.data is not None else 1
            text = plan.summary(len(self.learners), n_targets)
            color = "#c0392b" if plan.error else "palette(text)"
            self.window_summary.setText(f"<span style='color:{color}'>{text}</span>")
            headers = ["Iteración", "Entrenamiento", "Filas", "Hueco", "Test", "Filas test", "Completa"]
            rows, sort_values = [], []
            for it in plan.iterations[:5000]:
                gap = it.test_lo - it.train_hi
                rows.append([str(it.index + 1), plan.span(it.train_lo, it.train_hi), str(it.n_train), str(gap),
                             plan.span(it.test_lo, it.test_hi), str(it.n_test), "sí" if it.complete else "no"])
                sort_values.append([it.index, it.train_lo, it.n_train, gap, it.test_lo, it.n_test, int(it.complete)])
            self.iterations_model.set_content(headers, rows, sort_values)
            self.iterations_view.resizeColumnsToContents()
        self.chart_windows.invalidate()

    def _target_names(self):
        return [var.name for var in self.data.domain.class_vars] if self.data is not None else []

    def _target_position(self):
        names = self._target_names()
        return names.index(self.horizon_target) if self.horizon_target in names else 0

    def _update_horizon_combo(self):
        self.horizon_combo.clear()
        names = self._target_names()
        if len(names) > 1:
            self.horizon_combo.addItems([f"{name} (h = {position + 1})" for position, name in enumerate(names)])
            if self.horizon_target not in names:
                self.horizon_target = names[0]
            self.horizon_combo.setCurrentIndex(self._target_position())

    def _horizon_target_chosen(self, index):
        names = self._target_names()
        if 0 <= index < len(names) and names[index] != self.horizon_target:
            self.horizon_target = names[index]
            self._send_results = True
            self._options_changed()

    def _dm_horizon_changed(self, value):
        self.dm_horizon = value
        self._options_changed()

    def _window_plan_used(self):
        sampling = self._sampling_used
        if sampling is None or sampling.kind != engine.WINDOW:
            return None
        return sampling.plan

    def _draw_windows(self, width, height, theme):
        if self._preview_plan is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.windows_figure(self._preview_plan, width, height, theme)

    def _draw_iterations(self, width, height, theme):
        plan = self._window_plan_used()
        if self._analysis is None or plan is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.iterations_figure(self._analysis, self._scoreset, plan, width, height, theme)

    def _horizon_values(self):
        if not self._scoresets or self._analysis is None:
            return None
        metric = self._analysis.metric
        return np.column_stack([scoreset.overall[metric] for scoreset in self._scoresets])

    def _draw_horizons(self, width, height, theme):
        values = self._horizon_values()
        if values is None or len(self._scoresets) < 2:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.horizons_figure(self._analysis.names, self._target_names(), values,
                                     self._analysis.metric_label, self._analysis.lower_better,
                                     width, height, theme)

    # --------------------------------------------------------------- evaluación ----

    def _invalidate(self, which=None):
        self.cancel_evaluation(quiet=True)
        keys = self.learners.keys() if which is None else which
        for key in keys:
            self.learners[key] = Slot(self.learners[key].learner)
        self._scoreset = None
        self._scoresets = None
        self._large_confirmed = False
        self._need_refresh = True

    def _set_busy(self, busy):
        self.cancel_button.setVisible(busy)
        self.report_button2.setEnabled(not busy and self._analysis is not None)

    def cancel_evaluation(self, quiet=False):
        self._debounce.stop()
        self._window_timer.stop()
        self._after_task.stop()
        was_running = self.task is not None
        self.cancel()
        self._set_busy(False)
        if was_running and not quiet:
            self.banner.setText("Evaluación cancelada. Cambia una opción o reconecta para repetirla.")

    def _sampling_from_settings(self):
        stratified = self.shuffle_stratified if self.resampling == self.SHUFFLE else self.cv_stratified
        return engine.Sampling(
            kind=self.resampling, k=N_FOLDS[self.n_folds], stratified=stratified,
            n_repeats=N_REPEATS[self.n_repeats], train_fraction=SAMPLE_SIZES[self.sample_size] / 100,
            feature=self.fold_feature, seed=self.random_seed, window=self._window_spec())

    def _clear_results(self, message):
        self._analysis = self._scoreset = None
        self.scores_model.set_content([], [], [])
        self.details_model.set_content([], [], [])
        self.notes_browser.clear()
        self.matrix_combo.clear()
        self.details_combo.clear()
        self.banner.setText(message)
        self.report_button2.setEnabled(False)
        for chart in self.charts:
            chart.invalidate()
        self.Outputs.predictions.send(None)
        self.Outputs.evaluations_results.send(None)
        self.Outputs.scores.send(None)
        self.Outputs.fold_scores.send(None)
        self.Outputs.comparisons.send(None)
        self.Outputs.ranks.send(None)
        self.Outputs.best_learner.send(None)
        self.Outputs.horizon_scores.send(None)

    def _refresh(self):
        self._need_refresh = False
        self.cancel_evaluation(quiet=True)
        self.Error.sampling.clear()
        self.Error.analysis.clear()
        self.Warning.test_missing.clear()
        self.Warning.test_unused.clear()
        self.Warning.models_failed.clear()
        self.Warning.notes.clear()
        self.Information.test_transformed.clear()
        self.Information.report_saved.clear()
        if self.data is None or not self.learners:
            self._clear_results("Conecta datos y al menos un modelo.")
            return
        sampling, notes, problem = engine.check_sampling(self.data, self.test_data,
                                                         self._sampling_from_settings())
        self._sampling_notes = notes
        if self.resampling == self.WINDOW:
            self._show_plan(sampling.plan)
        if problem:
            self.Error.sampling(problem)
            if self._sampling_notes:
                self.Warning.notes("\n".join(dict.fromkeys(self._sampling_notes)))
            self._clear_results("Revisa el muestreo: no se ha podido evaluar.")
            return
        if self.resampling == self.ON_TEST and self.test_data is None:
            self.Warning.test_missing()
            self._clear_results("Falta conectar los datos de test.")
            return
        if self.resampling != self.ON_TEST and self.test_data is not None:
            self.Warning.test_unused()
        if self.resampling == self.ON_TEST and \
                self.data.domain.attributes != self.test_data.domain.attributes:
            self.Information.test_transformed()
        self._sampling_used = sampling
        pending = [(key, slot.learner) for key, slot in self.learners.items() if slot.outcome is None]
        if not pending:
            self._show_failures()
            self._after_task.start()
            return
        workers = 1
        work_rows = len(self.data)
        if self.resampling == self.WINDOW:
            trainings = len(sampling.plan.iterations) * len(pending) * len(self.data.domain.class_vars)
            if trainings > LARGE_TRAININGS and not self._large_confirmed:
                self.evaluate_button.setVisible(True)
                self._clear_results(f"Con estas opciones se harán {trainings:,} entrenamientos ".replace(",", ".")
                                    + "(iteraciones × modelos × objetivos). Revisa la vista previa y pulsa "
                                      "<b>Evaluar</b> para empezar.")
                return
            work_rows = tw.training_rows(sampling.plan)
        self.evaluate_button.setVisible(False)
        if self.parallel_models:
            workers = engine.default_workers(len(pending), self.preprocessor, work_rows)
        self.banner.setText("Evaluando modelos…")
        self._set_busy(True)
        self._running = "evaluate"
        self.Outputs.predictions.invalidate()
        self.Outputs.evaluations_results.invalidate()
        self.start(task_evaluate, EvalRequest(self.data, self.test_data, pending, sampling,
                                              self.preprocessor, workers))

    def _successful_slots(self):
        """Modelos evaluados sin error; con varios objetivos, en todos ellos."""
        out = []
        for key, slot in self.learners.items():
            outcome = slot.outcome
            if outcome is None:
                continue
            parts = outcome.per_target if outcome.per_target is not None else [outcome]
            if all(part.results is not None for part in parts):
                out.append((key, slot))
        return out

    def _show_failures(self):
        self.Warning.models_failed.clear()
        lines = []
        names = self._target_names()
        for slot in self.learners.values():
            if slot.outcome is None:
                continue
            parts = slot.outcome.per_target or [slot.outcome]
            for position, part in enumerate(parts):
                error = part.error
                if error is None:
                    continue
                name = learner_name(slot.learner)
                if len(parts) > 1 and position < len(names):
                    name += f" con «{names[position]}»"
                if isinstance(error, DomainTransformationError):
                    lines.append(f"{name}: los datos de test no son compatibles con los de entrenamiento.")
                elif isinstance(error, tw.IterationError):
                    lines.append(f"{name} ha fallado {error}")
                else:
                    lines.append(f"{name} ha fallado: {type(error).__name__}: {error}")
        if lines:
            self.Warning.models_failed("\n".join(lines))
        return lines

    def _options(self):
        horizon = self.dm_horizon or self._target_position() + 1
        return engine.TestOptions(metric=self.comparison_metric, alpha=ALPHAS[self.alpha_index],
                                  correction=self.correction_method, rope_auto=bool(self.rope_auto),
                                  rope=self.rope_value, horizon=horizon)

    def _start_analysis(self):
        self._debounce.stop()
        self.Error.analysis.clear()
        slots = self._successful_slots()
        if not slots or not self.scorers:
            self._clear_results("Ningún modelo se ha podido evaluar: revisa los avisos del widget.")
            return
        if self._sampling_used is None:
            return
        raw_names = [learner_name(slot.learner) for _, slot in slots]
        names = engine.unique_labels(raw_names)
        n_targets = len(self.data.domain.class_vars) if self._sampling_used.kind == engine.WINDOW else 1
        results = [[slot.outcome.part(t).results for _, slot in slots] for t in range(n_targets)]
        position = self._target_position() if n_targets > 1 else 0
        request = AnalysisRequest([key for key, _ in slots], names, raw_names, results,
                                  [slot.learner for _, slot in slots], self.scorers,
                                  self._sampling_used, self._target_index(), self._options(),
                                  len(self.data), self._scoresets if self._scoreset is not None else None,
                                  position,
                                  tuple(self._target_names()[:n_targets]), self._send_results)
        self._send_results = False
        self._set_busy(True)
        self._running = "analyse"
        self.start(task_analyse, request)

    def on_done(self, result):
        self._set_busy(False)
        if isinstance(result, EvalDone):
            for key, outcome in result.outcomes.items():
                if key in self.learners:
                    self.learners[key] = Slot(self.learners[key].learner, outcome)
            self._show_failures()
            self._scoreset = None
            # Orange prohíbe arrancar una tarea desde on_done.
            self._after_task.start()
        elif isinstance(result, AnalysisDone):
            self._on_analysis_done(result)
        elif isinstance(result, ReportDone):
            self._on_report_done(result)

    def on_exception(self, ex):
        self._set_busy(False)
        if isinstance(ex, engine.Cancelled):
            return
        if self._running == "report":
            self.Error.report(f"{type(ex).__name__}: {ex}")
            return
        self.Error.analysis(f"{type(ex).__name__}: {ex}")
        if self._running == "evaluate":
            self._clear_results("La evaluación ha fallado.")

    def _on_analysis_done(self, done):
        self._scoreset = done.scoreset
        self._scoresets = done.scoresets
        self._analysis = done.report
        self.Error.memory(shown=done.memory_error)
        self._show_notes()
        self._fill_banner()
        self._fill_scores_table()
        self._fill_test_combos()
        self._fill_details()
        for chart in self.charts:
            chart.invalidate()
        self.report_button2.setEnabled(True)
        report = done.report
        if done.fresh_scores:
            self.Outputs.predictions.send(done.predictions)
            self.Outputs.evaluations_results.send(done.scoreset.combined)
        self.Outputs.scores.send(done.tables["scores"])
        self.Outputs.fold_scores.send(done.tables["folds"])
        self.Outputs.comparisons.send(done.tables["pairs"])
        self.Outputs.ranks.send(done.tables["ranks"])
        self.Outputs.horizon_scores.send(done.tables.get("horizons"))
        best = None
        if report.best >= 0:
            best = self.learners[done.scoreset.keys[report.best]].learner
        self.Outputs.best_learner.send(best)

    def _show_notes(self):
        report = self._analysis
        notes = list(self._sampling_notes) + list(report.notes)
        missing = [engine.metric_label(s) for s in self._scoreset.scorers if s.__name__ in self._scoreset.errors]
        if missing:
            notes.append("No se han podido calcular: " + ", ".join(missing) + ".")
        self.Warning.notes.clear()
        if notes:
            self.Warning.notes("\n".join(dict.fromkeys(notes)))

    # --------------------------------------------------------------- resultados ----

    def _fill_banner(self):
        lines = self._analysis.recommendation
        if lines:
            self.banner.setText("<b>" + lines[0] + "</b><br>" + "<br>".join(lines[1:]))

    def _fill_scores_table(self):
        if self._analysis is None:
            return
        scoreset, report = self._scoreset, self._analysis
        visible = [s for s in scoreset.scorers if s.__name__ not in self.hidden_metrics]
        headers = ["Modelo"] + [engine.metric_label(s) for s in visible] + [
            f"IC {1 - report.alpha:.0%} ({report.metric_label})", "Rango medio", "Entrena (s)", "Predice (s)"]
        rows, sort_values = [], []
        highlight = set()
        for column, scorer in enumerate(visible, 1):
            values = scoreset.overall[scorer.__name__]
            if np.isfinite(values).any():
                best = np.nanmin(values) if engine.is_lower_better(scorer) else np.nanmax(values)
                highlight.update((row, column) for row in np.flatnonzero(np.isclose(values, best, rtol=1e-12)))
        tooltips = {}
        for model, name in enumerate(report.names):
            row, sort_row = [name], [name]
            for scorer in visible:
                value = scoreset.overall[scorer.__name__][model]
                row.append("—" if not np.isfinite(value) else f"{value:.4g}" if abs(value) >= 1000 else f"{value:.3f}")
                sort_row.append(float(value))
            low, high = report.ci_low[model], report.ci_high[model]
            row.append("—" if not np.isfinite(low) else f"{low:.3f} – {high:.3f}")
            sort_row.append(float(low))
            rank = report.mean_ranks[model]
            row += ["—" if not np.isfinite(rank) else f"{rank:.2f}", f"{scoreset.train_time[model]:.3f}",
                    f"{scoreset.test_time[model]:.3f}"]
            sort_row += [float(rank), float(scoreset.train_time[model]), float(scoreset.test_time[model])]
            rows.append(row)
            sort_values.append(sort_row)
        self.scores_model.set_content(headers, rows, sort_values, highlight, tooltips)
        self.scores_view.resizeColumnsToContents()
        names = [scorer.__name__ for scorer in visible]
        if report.metric in names:
            ascending = report.lower_better
            self.scores_view.sortByColumn(names.index(report.metric) + 1,
                                          Qt.AscendingOrder if ascending else Qt.DescendingOrder)

    def _fill_test_combos(self):
        report = self._analysis
        for combo, stored in ((self.matrix_combo, self.matrix_test), (self.details_combo, self.details_test)):
            combo.clear()
            for test in report.available:
                combo.addItem(st.TEST_NAMES[test], test)
            wanted = stored if stored in report.available else report.primary
            combo.setCurrentIndex(max(combo.findData(wanted), 0))
        self.matrix_test = self.matrix_combo.currentData() or ""
        self.details_test = self.details_combo.currentData() or ""

    def _fill_details(self):
        report = self._analysis
        if report is None:
            return
        from orangecontrib.orange_gualbe_toolbox.widgets._tsp_report import EXPLANATIONS, format_number, format_p
        test = self.details_combo.currentData()
        pairs = report.pairs_of(test) if test else []
        bayes = test == st.BAYES
        if bayes:
            headers = ["A", "B", "Diferencia (A − B)", "P(A mejor)", "P(equiv.)", "P(B mejor)", "Decisión"]
        else:
            effect_header = "Razón A/B" if test in (st.MCNEMAR, st.DIEBOLD_MARIANO) else "Efecto (A − B)"
            headers = ["A", "B", "Estadístico", "p", "p ajustado", "Diferencia (A − B)", "IC", effect_header,
                       "Decisión"]
        rows, sort_values = [], []
        tooltips = {}
        for position, pair in enumerate(pairs):
            a, b = report.names[pair.a], report.names[pair.b]
            for column in range(len(headers)):
                tooltips[(position, column)] = st.interpret(pair, report.names, report.metric_label, report.alpha)
            if bayes:
                rows.append([a, b, format_number(pair.mean_diff), format_number(pair.p_a, 3),
                             format_number(pair.p_rope, 3), format_number(pair.p_b, 3), pair.decision])
                sort_values.append([a, b, pair.mean_diff, pair.p_a, pair.p_rope, pair.p_b, pair.decision])
            else:
                interval = "—" if not np.isfinite(pair.ci_low) else (
                    f"{format_number(pair.ci_low)} – {format_number(pair.ci_high)}")
                effect = "—" if not np.isfinite(pair.effect) else (
                    f"{format_number(pair.effect, 3)} ({pair.effect_label})")
                rows.append([a, b, format_number(pair.statistic), format_p(pair.p_value).replace("&lt;", "<"),
                             format_p(pair.p_adjusted).replace("&lt;", "<"), format_number(pair.mean_diff),
                             interval, effect, pair.decision])
                sort_values.append([a, b, pair.statistic, pair.p_value, pair.p_adjusted, pair.mean_diff,
                                    pair.ci_low, pair.effect, pair.decision])
        self.details_model.set_content(headers, rows, sort_values, tooltips=tooltips)
        self.details_view.resizeColumnsToContents()

        text = []
        if test:
            text.append(f"<p><b>{st.TEST_NAMES[test]}.</b> {EXPLANATIONS[test]}</p>")
            if pairs:
                text.append(f"<p><b>Aporta evidencia sobre:</b> {pairs[0].evidence}. La diferencia es "
                            f"A − B, en {pairs[0].units}: en métricas de error y en rangos, un valor "
                            "negativo significa que A es mejor.</p>")
            assumption = next((p.assumption for p in pairs if p.assumption), "")
            if assumption:
                text.append(f"<p><b>Supuesto:</b> {assumption}</p>")
            seen = dict.fromkeys(line for p in pairs for line in p.warnings)
            text.extend(f"<p>⚠ {line}</p>" for line in seen)
        if report.friedman is not None and report.friedman.n_blocks:
            fried = report.friedman
            text.append(f"<p><b>Friedman:</b> χ² = {format_number(fried.chi2)}, p = "
                        f"{format_p(fried.p_value)}; Iman-Davenport F = "
                        f"{format_number(fried.iman_davenport)}, p = {format_p(fried.p_iman_davenport)}; "
                        f"diferencia crítica = {format_number(fried.critical_difference)}.</p>")
            text.extend(f"<p>⚠ {line}</p>" for line in fried.warnings)
        self.notes_browser.setHtml("".join(text))

    def _draw_scores(self, width, height, theme):
        if self._analysis is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.scores_figure(self._analysis, self._scoreset, width, height, theme)

    def _draw_cd(self, width, height, theme):
        if self._analysis is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.critical_difference_figure(self._analysis, width, height, theme)

    def _draw_matrix(self, width, height, theme):
        if self._analysis is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        test = self.matrix_combo.currentData() or self._analysis.primary
        return plots.matrix_figure(self._analysis, test, width, height, theme)

    def _draw_bayes(self, width, height, theme):
        if self._analysis is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.bayes_figure(self._analysis, width, height, theme)

    def _draw_times(self, width, height, theme):
        if self._analysis is None:
            return None
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        return plots.times_figure(self._analysis, self._scoreset, width, height, theme)

    # ------------------------------------------------------------------- informe ----

    def _report_context(self):
        data = self.data
        class_vars = data.domain.class_vars
        class_var = class_vars[self._target_position() if len(class_vars) > 1 else 0]
        targets = ", ".join(var.name for var in class_vars)
        setup = [("Conjunto de datos", data.name), ("Filas", len(data)),
                 ("Atributos", len(data.domain.attributes)), ("Metas", len(data.domain.metas)),
                 ("Variables objetivo" if len(class_vars) > 1 else "Variable objetivo", targets),
                 ("Tipo de problema", "Clasificación" if class_var.is_discrete else "Regresión"),
                 ("Muestreo", self._sampling_used.describe()), ("Semilla", self.random_seed),
                 ("Métrica de comparación", self._analysis.metric_label),
                 ("Significación α", self._analysis.alpha)]
        if len(class_vars) > 1:
            setup.append(("Objetivo analizado", f"{class_var.name} (h = {self._target_position() + 1})"))
        elif class_var.is_discrete:
            counts = np.bincount(data.Y.astype(int), minlength=len(class_var.values))
            setup.append(("Distribución de la clase",
                          ", ".join(f"{v}: {c}" for v, c in zip(class_var.values, counts))))
            setup.append(("Clase objetivo", self.class_selection))
        if self._dropped_rows:
            setup.append(("Filas descartadas", self._dropped_rows))
        setup.append(("Valores perdidos en los atributos", "sí" if data.has_missing_attribute() else "no"))
        if self.preprocessor is not None:
            setup.append(("Preprocesador", str(self.preprocessor)))
        if self.test_data is not None and self.resampling == self.ON_TEST:
            setup.append(("Filas de test", len(self.test_data)))
        from orangecontrib.orange_gualbe_toolbox.widgets._tsp_report import learner_parameters
        models = [(learner_name(slot.learner), learner_parameters(slot.learner, data.domain))
                  for slot in self.learners.values()]
        context = dict(title=f"Evaluación de modelos: {data.name}", setup=setup, models=models,
                       failures=self._show_failures(),
                       notes=list(self._sampling_notes) + list(self._analysis.notes))
        plan = self._window_plan_used()
        if plan is not None:
            context["window"] = dict(config=self._window_config(plan), plan=plan,
                                     target=class_var.name if len(class_vars) > 1 else "",
                                     target_names=self._target_names() if len(class_vars) > 1 else [],
                                     horizon_values=self._horizon_values())
        return context

    def _window_config(self, plan):
        spec = plan.spec
        order = spec.time_var.name if spec.time_var is not None else "Orden de las filas"
        start = {tw.START_FIRST: "Primera fila", tw.START_ROW: f"Desde la fila {spec.start_row}",
                 tw.START_DATE: f"Desde {plan.label(plan.start)}"}[spec.start_mode]
        initial = {tw.INITIAL_PERCENT: f"{spec.initial_percent} % de las filas del estudio",
                   tw.INITIAL_ROWS: f"{spec.initial_rows} filas",
                   tw.INITIAL_DATE: "Hasta la fecha indicada"}[spec.initial_mode]
        first = plan.iterations[0] if plan.iterations else None
        if first is not None:
            initial += f" ({first.n_train} filas: {plan.span(first.train_lo, first.train_hi)})"
        unused = plan.unused_counts()
        config = [("Ordenar por", order), ("Inicio", start), ("Entrenamiento inicial", initial),
                  ("Avance", "Creciente: inicio fijo" if spec.advance == tw.EXPANDING
                   else "Deslizante: tamaño fijo"),
                  ("Test", spec.amount(spec.test_size)), ("Hueco", spec.amount(spec.gap)),
                  ("Paso", spec.amount(spec.effective_step)),
                  ("Fin", "Hasta el final" if spec.end_mode == tw.END_ALL else f"Hasta {plan.label(plan.end - 1)}"),
                  ("Último test incompleto", "Se usa" if spec.last == tw.LAST_KEEP else "Se descarta"),
                  ("Entrenamiento mínimo", f"{spec.min_train} filas"),
                  ("Iteraciones", len(plan.iterations)),
                  ("Filas evaluadas", int(plan.tested.sum())),
                  ("Filas fuera del estudio", unused["before"] + unused["after"]),
                  ("Filas sin evaluar dentro del estudio", unused["between"] + unused["tail"]),
                  ("Horizonte de Diebold-Mariano", self._options().horizon)]
        return config

    def export_report_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exportar informe", "informe_modelos.html",
                                              "HTML (*.html);;PDF (*.pdf)")
        if path:
            self.export_report(path)

    def export_report(self, path):
        if self._analysis is None:
            return
        self.Error.report.clear()
        self.Information.report_saved.clear()
        try:
            context = self._report_context()
        except Exception as error:  # pylint: disable=broad-except
            self.Error.report(f"{type(error).__name__}: {error}")
            return
        self._set_busy(True)
        self._running = "report"
        self.start(task_report, context, self._scoreset, self._analysis, path)

    def _on_report_done(self, done):
        self.Error.report.clear()
        self._report_layout = done.layout
        self.Information.report_saved(done.path)

    def send_report(self):
        if self.data is None or self._analysis is None:
            return
        report = self._analysis
        self.report_items("Evaluación", [("Muestreo", self._sampling_used.describe()),
                                         ("Métrica", report.metric_label), ("α", report.alpha)])
        self.report_paragraph("Recomendación", " ".join(report.recommendation))
        self.report_table("Puntuaciones", self.scores_view)
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
        from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_report as builder
        import base64
        for title, figure in builder.report_figures(report, self._scoreset):
            data = base64.b64encode(plots.figure_png(figure, 110)).decode("ascii")
            self.report_raw(title, f"<img style='max-width:100%' src='data:image/png;base64,{data}'/>")
        self.report_table("Tests", self.details_view)

    def onDeleteWidget(self):
        self._debounce.stop()
        self._window_timer.stop()
        self._after_task.stop()
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner, MajorityLearner, RandomForestLearner
    WidgetPreview(OWTestScorePro).run(
        set_train_data=Table("heart_disease"),
        insert_learner=[(0, LogisticRegressionLearner()), (1, RandomForestLearner()), (2, MajorityLearner())])
