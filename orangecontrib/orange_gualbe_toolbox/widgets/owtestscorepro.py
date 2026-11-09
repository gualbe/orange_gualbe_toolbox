"""Widget Test & Score Pro: Test & Score con comparaciones estadísticas rigurosas.

La lógica vive en _tsp_engine.py, _tsp_stats.py, _tsp_plots.py y _tsp_report.py, sin Qt
salvo la exportación a PDF. Aquí están la interfaz, el estado y el reparto de trabajo:
evaluar es caro y se hace una sola vez por modelo; analizar y dibujar no reentrenan.
"""
import re
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Any, Optional

import numpy as np

from AnyQt.QtCore import QAbstractTableModel, QModelIndex, QSize, QSizeF, QSortFilterProxyModel, Qt, QTimer, QUrl
from AnyQt.QtGui import QBrush, QColor, QFont, QImage, QPalette, QPixmap, QTextDocument
from AnyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableView, QTabWidget,
    QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from Orange.base import Learner
from Orange.data import DiscreteVariable, Table
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

N_FOLDS = [2, 3, 5, 10, 20]
N_REPEATS = [2, 3, 5, 10, 20, 50, 100]
SAMPLE_SIZES = [5, 10, 20, 25, 30, 33, 40, 50, 60, 66, 70, 75, 80, 90, 95]
ALPHAS = [0.01, 0.05, 0.1]
TARGET_AVERAGE = "(Promedio de las clases)"
DEBOUNCE_MS = 150
A4_LANDSCAPE = (842.0, 595.0)
PDF_IMAGE_WIDTH = 620


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
    results: list
    learners: list
    scorers: list
    sampling: engine.Sampling
    target: Optional[int]
    options: engine.TestOptions
    n_rows: int
    cached: Optional[engine.ScoreSet]


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


@dataclass
class ReportDone:
    path: str
    html: str


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
    scoreset = request.cached
    fresh = scoreset is None
    if fresh:
        scoreset = engine.compute_scores(request.keys, request.names, request.results, request.scorers,
                                         request.sampling, request.n_rows, request.target, check,
                                         request.raw_names)
    report = engine.run_tests(scoreset, request.sampling, request.options, check)
    tables = dict(scores=engine.scores_table(scoreset, report), folds=engine.folds_table(scoreset),
                  pairs=engine.pairs_table(report), ranks=engine.ranks_table(report))
    predictions, memory_error = None, False
    if fresh:
        try:
            predictions = scoreset.combined.get_augmented_data(request.raw_names)
        except MemoryError:
            memory_error = True
    return AnalysisDone(scoreset, report, predictions, tables, fresh, memory_error)


def task_report(context, scoreset, report, path, state):
    state.set_status("Generando el informe…")
    from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_report
    return ReportDone(path, _tsp_report.build_html(context, scoreset, report))


def html_to_pdf(html, path):
    """Convierte el informe a PDF con Qt; debe llamarse desde el hilo de la interfaz."""
    import base64
    from AnyQt.QtGui import QPageLayout
    from AnyQt.QtPrintSupport import QPrinter
    images = []

    def collect(match):
        images.append(base64.b64decode(match.group(1)))
        return f"src='imagen{len(images) - 1}.png' width='{PDF_IMAGE_WIDTH}'"

    pdf_html = re.sub(r"font-size:\s*(\d+)px", lambda m: f"font-size:{int(m.group(1)) * 0.75:g}px", html)
    pdf_html = re.sub(r"src='data:image/png;base64,([^']+)'", collect, pdf_html)
    document = QTextDocument()
    # Con el tamaño de página en puntos, Qt trata 1 px como 1 pt al imprimir; sin esto
    # escala cada tabla por su cuenta y salen diminutas.
    document.setPageSize(QSizeF(A4_LANDSCAPE[0], A4_LANDSCAPE[1]))
    document.setHtml(pdf_html)
    # Después de setHtml: setHtml borra los recursos que ya tuviera el documento.
    for position, data in enumerate(images):
        image = QImage()
        image.loadFromData(data)
        document.addResource(QTextDocument.ImageResource, QUrl(f"imagen{position}.png"), image)
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setPageOrientation(QPageLayout.Landscape)
    printer.setOutputFileName(path)
    printing = getattr(document, "print", None) or document.print_
    printing(printer)


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
                "bayes", "wilcoxon", "friedman", "nemenyi", "mcnemar"]

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

    class Error(OWWidget.Error):
        train_data = Msg("{}")
        test_data = Msg("{}")
        sampling = Msg("{}")
        analysis = Msg("No se pudieron calcular las comparaciones: {}")
        memory = Msg("No hay memoria suficiente para crear la tabla de predicciones.")
        report = Msg("No se pudo guardar el informe: {}")

    class Warning(OWWidget.Warning):
        missing_class = Msg("Se han quitado {} filas sin valor en la clase.")
        test_missing = Msg("Falta conectar los datos de test.")
        test_unused = Msg("Hay datos de test conectados que no se usan: elige «Sobre datos de test».")
        models_failed = Msg("{}")
        notes = Msg("{}")

    class Information(OWWidget.Information):
        test_transformed = Msg("Los datos de test se han transformado para coincidir con los de "
                               "entrenamiento.")
        report_saved = Msg("Informe guardado en {}")

    settings_version = 1
    settingsHandler = PerfectDomainContextHandler()

    KFOLD, FEATURE, SHUFFLE, LOO, ON_TRAIN, ON_TEST = range(6)

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
        self._after_task = QTimer(self, singleShot=True, interval=0)
        self._after_task.timeout.connect(self._start_analysis)
        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._debounce.timeout.connect(self._start_analysis)
        self._build_controls()
        self._build_main_area()
        self._sync_advanced()

    # ------------------------------------------------------ construcción ----

    def _build_controls(self):
        box = gui.vBox(self.controlArea, "Muestreo")
        radio = gui.radioButtons(box, self, "resampling", callback=self._parameters_changed)
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

        box = gui.vBox(self.controlArea, "Comparar")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.metric_combo = QComboBox(self)
        self.metric_combo.activated[int].connect(self._metric_chosen)
        self.target_label = QLabel("Clase objetivo:")
        self.target_combo = QComboBox(self)
        self.target_combo.activated[int].connect(self._target_chosen)
        self.alpha_combo = QComboBox(self)
        self.alpha_combo.addItems([f"{a:g}".replace(".", ",") for a in ALPHAS])
        self.alpha_combo.setCurrentIndex(self.alpha_index)
        self.alpha_combo.activated[int].connect(self._alpha_chosen)
        grid.addWidget(QLabel("Métrica:"), 0, 0)
        grid.addWidget(self.metric_combo, 0, 1)
        grid.addWidget(self.target_label, 1, 0)
        grid.addWidget(self.target_combo, 1, 1)
        grid.addWidget(QLabel("Significación α:"), 2, 0)
        grid.addWidget(self.alpha_combo, 2, 1)
        grid.setColumnStretch(1, 1)

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
        grid.addWidget(self.parallel_check, 3, 0, 1, 2)

        gui.rubber(self.controlArea)
        self.report_button2 = QPushButton("Exportar informe…", self)
        self.report_button2.clicked.connect(self.export_report_dialog)
        self.report_button2.setEnabled(False)
        self.controlArea.layout().addWidget(self.report_button2)
        self.cancel_button = QPushButton("Cancelar", self)
        self.cancel_button.clicked.connect(self.cancel_evaluation)
        self.cancel_button.setVisible(False)
        self.controlArea.layout().addWidget(self.cancel_button)

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
                       self.chart_times)

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
        self.Warning.missing_class.clear()
        if data is not None:
            problem = self._data_problem(data)
            if problem:
                self.Error.train_data(problem)
                data = None
        if data is not None and np.isnan(data.Y).any():
            before = len(data)
            data = HasClass()(data)
            self.Warning.missing_class(before - len(data))
            if not len(data):
                self.Error.train_data("Ninguna fila tiene valor en la variable objetivo.")
                data = None
        self.closeContext()
        self.data = data
        self._update_scorers()
        self._update_target_combo()
        self.feature_model.set_domain(data.domain if data is not None else None)
        self.fold_feature = self.feature_model[0] if self.feature_model else None
        if data is not None:
            self.openContext(data.domain)
            if self.fold_feature_selected and self.feature_model:
                self.resampling = self.FEATURE
        self._select_target_from_setting()
        self._invalidate()

    @staticmethod
    def _data_problem(data):
        if not len(data):
            return "La tabla de entrenamiento está vacía."
        if not data.domain.class_vars:
            return "Los datos no tienen variable objetivo."
        if len(data.domain.class_vars) > 1:
            return "Hay varias variables objetivo: deja solo una con Select Columns."
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

    def _update_scorers(self):
        self.scorers = usable_scorers(self.data.domain) if self.data is not None else []
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
        self._invalidate()
        self._refresh()

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

    # --------------------------------------------------------------- evaluación ----

    def _invalidate(self, which=None):
        self.cancel_evaluation(quiet=True)
        keys = self.learners.keys() if which is None else which
        for key in keys:
            self.learners[key] = Slot(self.learners[key].learner)
        self._scoreset = None
        self._need_refresh = True

    def _set_busy(self, busy):
        self.cancel_button.setVisible(busy)
        self.report_button2.setEnabled(not busy and self._analysis is not None)

    def cancel_evaluation(self, quiet=False):
        self._debounce.stop()
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
            feature=self.fold_feature, seed=self.random_seed)

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
        if problem:
            self.Error.sampling(problem)
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
        if self.parallel_models:
            workers = engine.default_workers(len(pending), self.preprocessor, len(self.data))
        self.banner.setText("Evaluando modelos…")
        self._set_busy(True)
        self._running = "evaluate"
        self.Outputs.predictions.invalidate()
        self.Outputs.evaluations_results.invalidate()
        self.start(task_evaluate, EvalRequest(self.data, self.test_data, pending, sampling,
                                              self.preprocessor, workers))

    def _successful_slots(self):
        return [(key, slot) for key, slot in self.learners.items()
                if slot.outcome is not None and slot.outcome.results is not None]

    def _show_failures(self):
        self.Warning.models_failed.clear()
        lines = []
        for slot in self.learners.values():
            error = slot.outcome.error if slot.outcome is not None else None
            if error is None:
                continue
            if isinstance(error, DomainTransformationError):
                lines.append(f"{learner_name(slot.learner)}: los datos de test no son compatibles "
                             "con los de entrenamiento.")
            else:
                lines.append(f"{learner_name(slot.learner)} ha fallado: {type(error).__name__}: {error}")
        if lines:
            self.Warning.models_failed("\n".join(lines))
        return lines

    def _options(self):
        return engine.TestOptions(metric=self.comparison_metric, alpha=ALPHAS[self.alpha_index],
                                  correction=self.correction_method, rope_auto=bool(self.rope_auto),
                                  rope=self.rope_value)

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
        request = AnalysisRequest([key for key, _ in slots], names, raw_names,
                                  [slot.outcome.results for _, slot in slots],
                                  [slot.learner for _, slot in slots], self.scorers,
                                  self._sampling_used, self._target_index(), self._options(),
                                  len(self.data), self._scoreset)
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
            effect_header = "Razón A/B" if test == st.MCNEMAR else "Efecto (A − B)"
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
        class_var = data.domain.class_var
        setup = [("Conjunto de datos", data.name), ("Filas", len(data)),
                 ("Atributos", len(data.domain.attributes)), ("Metas", len(data.domain.metas)),
                 ("Variable objetivo", class_var.name),
                 ("Tipo de problema", "Clasificación" if class_var.is_discrete else "Regresión"),
                 ("Muestreo", self._sampling_used.describe()), ("Semilla", self.random_seed),
                 ("Métrica de comparación", self._analysis.metric_label),
                 ("Significación α", self._analysis.alpha)]
        if class_var.is_discrete:
            counts = np.bincount(data.Y.astype(int), minlength=len(class_var.values))
            setup.append(("Distribución de la clase",
                          ", ".join(f"{v}: {c}" for v, c in zip(class_var.values, counts))))
            setup.append(("Clase objetivo", self.class_selection))
        setup.append(("Valores perdidos en los atributos", "sí" if data.has_missing_attribute() else "no"))
        if self.preprocessor is not None:
            setup.append(("Preprocesador", str(self.preprocessor)))
        if self.test_data is not None and self.resampling == self.ON_TEST:
            setup.append(("Filas de test", len(self.test_data)))
        from orangecontrib.orange_gualbe_toolbox.widgets._tsp_report import learner_parameters
        models = [(learner_name(slot.learner), learner_parameters(slot.learner))
                  for slot in self.learners.values()]
        return dict(title=f"Evaluación de modelos: {data.name}", setup=setup, models=models,
                    failures=self._show_failures(),
                    notes=list(self._sampling_notes) + list(self._analysis.notes))

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
        self._set_busy(True)
        self._running = "report"
        self.start(task_report, self._report_context(), self._scoreset, self._analysis, path)

    def _on_report_done(self, done):
        try:
            if done.path.lower().endswith(".pdf"):
                html_to_pdf(done.html, done.path)
            else:
                with open(done.path, "w", encoding="utf-8") as stream:
                    stream.write(done.html)
        except OSError as error:
            self.Error.report(str(error))
            return
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
        self._after_task.stop()
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner, MajorityLearner, RandomForestLearner
    WidgetPreview(OWTestScorePro).run(
        set_train_data=Table("heart_disease"),
        insert_learner=[(0, LogisticRegressionLearner()), (1, RandomForestLearner()), (2, MajorityLearner())])
