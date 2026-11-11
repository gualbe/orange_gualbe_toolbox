# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.
# Portions adapted from Orange Data Mining (Orange/widgets/data/owfile.py),
# Copyright (C) Bioinformatics Laboratory, University of Ljubljana, GPL-3.0-or-later.

"""Widget File Import: reúne lo mejor de File y CSV File Import.

La lectura vive en _fileimport.py (sin Qt). Aquí solo hay interfaz, estado que se
guarda con el workflow y el reparto de trabajo en segundo plano: primero una
vista previa que lee únicamente el principio del archivo y después, si procede,
la carga completa.
"""
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from AnyQt.QtCore import QAbstractTableModel, QModelIndex, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from AnyQt.QtGui import QColor, QPainter
from AnyQt.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QRadioButton, QSizePolicy,
    QSpinBox, QSplitter, QStyle, QStyledItemDelegate, QTableView, QToolButton,
    QVBoxLayout, QWidget,
)

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, Table, TimeVariable
from Orange.data.table import get_sample_datasets_dir
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.filedialogs import (
    OWUrlDropBase, RecentPath, RecentPathsWComboMixin,
)
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Output

from orangecontrib.orange_gualbe_toolbox.widgets import _fileimport as engine
from orangecontrib.orange_gualbe_toolbox.widgets._fileimport import (
    ATTRIBUTE, AUTO, CATEGORICAL, CLASS, DATE, HEADER_AUTO, HEADER_NONE, META, NUMERIC,
    SKIP, TEXT, ColumnChoice, Downloaded, ImportCancelled, ImportFailure, Loaded,
    Options, Progress, format_int,
)

TYPE_LABELS = {NUMERIC: "Numérico", CATEGORICAL: "Categórico", TEXT: "Texto", DATE: "Fecha"}
ROLE_LABELS = {ATTRIBUTE: "Atributo", CLASS: "Clase", META: "Meta", SKIP: "Omitir"}
TYPE_COLORS = {NUMERIC: "#E0585B", CATEGORICAL: "#4E9A52", TEXT: "#8A8A8A", DATE: "#D9A441"}
TYPE_PLURALS = {NUMERIC: "numéricas", CATEGORICAL: "categóricas", TEXT: "de texto",
                DATE: "de fecha"}
TYPE_SINGULARS = {NUMERIC: "numérica", CATEGORICAL: "categórica", TEXT: "de texto",
                  DATE: "de fecha"}
FORMAT_LABELS = {
    AUTO: "Automático", "csv": "CSV", "tsv": "TSV", "tab": "TAB de Orange",
    "xlsx": "Excel (XLSX)", "xls": "Excel (XLS)", "arff": "ARFF",
}
FILE_FILTER = ("Datos (*.csv *.tsv *.tab *.txt *.xlsx *.xlsm *.xls *.arff "
               "*.gz *.bz2 *.xz *.zip);;Todos los archivos (*)")
MAX_SAVED_CONFIGS = 40
DEBOUNCE_MS = 350


def _type_icon(kind):
    variable = {NUMERIC: ContinuousVariable("x"), CATEGORICAL: DiscreteVariable("x"),
                TEXT: StringVariable("x"), DATE: TimeVariable("x")}[kind]
    return gui.attributeIconDict[variable]


def _delimiter_text(delimiter):
    return {"\t": "tabulador", " ": "espacio"}.get(delimiter, delimiter)


# ------------------------------------------------------------------- tareas ----

@dataclass
class Request:
    kind: int
    target: str
    options: Options
    temp_dir: str
    download: Optional[Downloaded]


@dataclass
class PreviewResult:
    download: Optional[Downloaded]
    path: str
    display_name: str
    loaded: Loaded
    cells: list


@dataclass
class LoadResult:
    download: Optional[Downloaded]
    path: str
    display_name: str
    loaded: Loaded
    table: Table
    notes: List[str]
    signature: tuple
    cells: list


@dataclass
class ConvertResult:
    table: Table
    notes: List[str]


def _progress(state):
    return Progress(report=state.set_progress_value,
                    is_cancelled=state.is_interruption_requested,
                    status=state.set_status)


def _resolve(request, progress):
    if request.kind == OWFileImport.LOCAL:
        return None, request.target, request.target
    download = request.download
    if download is None or not os.path.exists(download.path):
        progress.say("Descargando…")
        download = engine.download(request.target, request.temp_dir,
                                   request.options.max_download_mb, progress)
    return download, download.path, download.display_name


def _remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _output_limit(download, options):
    """Los archivos locales son del usuario: solo se limita lo descargado."""
    if download is None:
        return None
    return engine.decompression_limit(download.size, options.max_download_mb)


def _signature(path, options):
    stat = os.stat(path)
    return (path, stat.st_mtime_ns, stat.st_size, tuple(sorted(options.to_dict().items())))


def _read(request, download, path, name, progress, **kwargs):
    """Lee lo resuelto sin perder la pista de un temporal recién descargado."""
    try:
        return engine.load(path, request.options, name, progress=progress,
                           max_output=_output_limit(download, request.options), **kwargs)
    except ImportCancelled:
        if download is not None and download is not request.download:
            _remove_file(download.path)
        raise
    except ImportFailure as failure:
        failure.download = download
        raise


def task_preview(request, overrides, state):
    progress = _progress(state)
    download, path, name = _resolve(request, progress)
    progress.say("Leyendo la vista previa…")
    loaded = _read(request, download, path, name, progress,
                   limit=engine.INFERENCE_ROWS, head_only=True)
    cells = engine.preview_columns(loaded, apply_overrides(loaded.columns, overrides))
    return PreviewResult(download, path, name, loaded, cells)


def task_load(request, overrides, state):
    progress = _progress(state)
    download, path, name = _resolve(request, progress)
    progress.say("Leyendo el archivo…")
    signature = _signature(path, request.options)
    loaded = _read(request, download, path, name, progress)
    progress.say("Creando la tabla…")
    choices = apply_overrides(loaded.columns, overrides)
    try:
        table, notes = engine.frame_to_table(loaded, choices, progress)
    except ImportFailure as failure:
        failure.download = download
        raise
    cells = engine.preview_columns(loaded, choices)
    return LoadResult(download, path, name, loaded, table, notes, signature, cells)


def task_convert(loaded, choices, state):
    progress = _progress(state)
    progress.say("Creando la tabla…")
    table, notes = engine.frame_to_table(loaded, choices, progress)
    return ConvertResult(table, notes)


def column_key(meta):
    return meta.source.strip() or f"#{meta.index}"


def apply_overrides(columns, overrides):
    choices = engine.default_choices(columns)
    for meta, choice in zip(columns, choices):
        item = overrides.get(column_key(meta))
        if not item:
            continue
        choice.name = str(item.get("name", choice.name)) or choice.name
        if item.get("type") in engine.TYPES:
            choice.type = item["type"]
        if item.get("role") in engine.ROLES:
            choice.role = item["role"]
        if choice.type == TEXT and choice.role in (ATTRIBUTE, CLASS):
            choice.role = META
    return choices


# ------------------------------------------------------------------ modelos ----

class ColumnsModel(QAbstractTableModel):
    """Tabla editable de columnas: una fila por columna del archivo."""

    NAME, TYPE, ROLE, MISSING, UNIQUE, SAMPLE = range(6)
    HEADERS = ("Nombre", "Tipo", "Rol", "Perdidos", "Únicos", "Ejemplo")
    edited = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns = []
        self.choices = []
        self.n_rows = 0

    def set_columns(self, columns, choices, n_rows):
        self.beginResetModel()
        self.columns = columns
        self.choices = choices
        self.n_rows = n_rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.column() in (self.NAME, self.TYPE, self.ROLE):
            flags |= Qt.ItemIsEditable
        return flags

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        meta, choice = self.columns[index.row()], self.choices[index.row()]
        column = index.column()
        if role == Qt.ForegroundRole and choice.role == SKIP:
            return QColor(150, 150, 150)
        if role == Qt.DecorationRole and column in (self.NAME, self.TYPE):
            return _type_icon(choice.type)
        if role == Qt.ToolTipRole:
            if meta.source and meta.source.strip() != meta.name:
                return f"Nombre en el archivo: «{meta.source}»"
            return f"Inferido: {TYPE_LABELS[meta.type]}, {ROLE_LABELS[meta.role]}"
        if role == Qt.UserRole and column == self.MISSING:
            return meta.n_missing / max(self.n_rows, 1)
        if role == Qt.EditRole:
            return {self.NAME: choice.name, self.TYPE: choice.type,
                    self.ROLE: choice.role}.get(column)
        if role == Qt.DisplayRole:
            if column == self.NAME:
                return choice.name
            if column == self.TYPE:
                return TYPE_LABELS[choice.type]
            if column == self.ROLE:
                return ROLE_LABELS[choice.role]
            if column == self.MISSING:
                return f"{100 * meta.n_missing / max(self.n_rows, 1):.1f} %".replace(".", ",")
            if column == self.UNIQUE:
                return format_int(meta.n_unique)
            if column == self.SAMPLE:
                return ", ".join(meta.samples)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        return self.apply_to([index.row()], {
            self.NAME: "name", self.TYPE: "type", self.ROLE: "role",
        }[index.column()], value)

    def apply_to(self, rows, attribute, value):
        changed = []
        for row in rows:
            choice = self.choices[row]
            if attribute == "name":
                value = str(value).strip()
                if not value:
                    continue
            if getattr(choice, attribute) == value:
                continue
            if attribute == "role" and choice.type == TEXT and value in (ATTRIBUTE, CLASS):
                continue  # Orange solo admite texto como meta
            setattr(choice, attribute, value)
            if attribute == "type" and value == TEXT and choice.role in (ATTRIBUTE, CLASS):
                choice.role = META
            changed.append(row)
        if changed:
            self.dataChanged.emit(self.index(min(changed), 0),
                                  self.index(max(changed), self.columnCount() - 1))
            self.edited.emit(changed)
        return bool(changed)


class PreviewModel(QAbstractTableModel):
    """Vista previa guardada por columnas: editar una no recalcula las demás."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cells = []
        self.choices = []

    def set_cells(self, cells, choices):
        self.beginResetModel()
        self.cells = cells
        self.choices = choices
        self.endResetModel()

    def set_column(self, column, values):
        self.cells[column] = values
        self.dataChanged.emit(self.index(0, column),
                              self.index(self.rowCount() - 1, column))
        self.headerDataChanged.emit(Qt.Horizontal, column, column)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or not self.cells else len(self.cells[0])

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.choices)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical:
            return str(section + 1) if role == Qt.DisplayRole else None
        if section >= len(self.choices):
            return None
        choice = self.choices[section]
        if role == Qt.DisplayRole:
            return choice.name
        if role == Qt.DecorationRole:
            return _type_icon(choice.type)
        if role == Qt.ToolTipRole:
            return f"{TYPE_LABELS[choice.type]} · {ROLE_LABELS[choice.role]}"
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.cells[index.column()][index.row()]
        skipped = self.choices[index.column()].role == SKIP
        if role == Qt.DisplayRole:
            return "?" if value == "" else value
        if role == Qt.ForegroundRole and (value == "" or skipped):
            return QColor(160, 160, 160)
        if role == Qt.TextAlignmentRole and self.choices[index.column()].type == NUMERIC:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class ComboDelegate(QStyledItemDelegate):
    """Editor de lista que se abre con un solo clic."""

    def __init__(self, labels, icons=None, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.icons = icons

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        for value, label in self.labels.items():
            editor.addItem(label, value)
            if self.icons:
                editor.setItemIcon(editor.count() - 1, self.icons(value))
        editor.activated.connect(lambda: self.commitData.emit(editor))
        editor.activated.connect(lambda: self.closeEditor.emit(editor))
        QTimer.singleShot(0, editor.showPopup)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(max(editor.findData(index.data(Qt.EditRole)), 0))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.EditRole)


class MissingDelegate(QStyledItemDelegate):
    """Porcentaje de perdidos con una barra de fondo."""

    def paint(self, painter, option, index):
        share = index.data(Qt.UserRole) or 0
        if share > 0:
            painter.save()
            color = QColor("#E0585B")
            color.setAlphaF(0.30)
            rect = QRectF(option.rect).adjusted(2, 3, -2, -3)
            rect.setWidth(max(rect.width() * share, 2))
            painter.fillRect(rect, color)
            painter.restore()
        super().paint(painter, option, index)


class TypeBar(QWidget):
    """Barra apilada con la proporción de columnas de cada tipo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = {}
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_counts(self, counts):
        self.counts = counts
        self.update()

    def paintEvent(self, _event):
        total = sum(self.counts.values())
        if not total:
            return
        painter = QPainter(self)
        x = 0.0
        width = float(self.width())
        for kind in engine.TYPES:
            count = self.counts.get(kind, 0)
            if not count:
                continue
            share = width * count / total
            painter.fillRect(QRectF(x, 0, max(share - 2, 1), self.height()),
                             QColor(TYPE_COLORS[kind]))
            x += share


class StatTile(QWidget):
    def __init__(self, caption, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 18, 0)
        layout.setSpacing(0)
        self.value = QLabel("—")
        font = self.value.font()
        font.setPointSizeF(font.pointSizeF() * 1.55)
        font.setBold(True)
        self.value.setFont(font)
        caption_label = QLabel(caption)
        caption_label.setStyleSheet("color: #808080;")
        layout.addWidget(self.value)
        layout.addWidget(caption_label)


class OWFileImport(OWUrlDropBase, RecentPathsWComboMixin, ConcurrentWidgetMixin):
    name = "File Import"
    description = ("Importa datos desde un archivo o una URL (CSV, TSV, TAB, Excel y "
                   "ARFF) con vista previa, edición de columnas y opciones avanzadas.")
    icon = "icons/owfileimport.svg"
    priority = 20
    keywords = ["file", "import", "csv", "excel", "arff", "tsv", "url", "archivo",
                "importar", "cargar"]

    class Outputs:
        data = Output("Data", Table)

    class Error(OWUrlDropBase.Error):
        failed = Msg("{}")
        unexpected = Msg("Error inesperado al importar: {}")

    class Warning(OWUrlDropBase.Warning):
        notes = Msg("{}")
        columns_changed = Msg("{}")

    class Information(OWUrlDropBase.Information):
        details = Msg("{}")

    LOCAL, URL = range(2)
    SEARCH_PATHS = [("sample-datasets", get_sample_datasets_dir())]

    # Ojo: ningún Setting puede llamarse como un método de QWidget/OWWidget.
    recent_paths = Setting([
        RecentPath("", "sample-datasets", "iris.tab"),
        RecentPath("", "sample-datasets", "titanic.tab"),
        RecentPath("", "sample-datasets", "housing.tab"),
        RecentPath("", "sample-datasets", "heart_disease.tab"),
    ])
    recent_urls = Setting([])
    source_kind = Setting(LOCAL)
    url_text = Setting("")
    source_configs = Setting({}, schema_only=True)
    advanced_open = Setting(False)
    auto_apply = Setting(True)

    want_main_area = True

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.setAcceptDrops(True)
        self._temp_dir = None
        self._download = None
        self._display_path = None
        self._preview = None
        self._loaded = None
        self._loaded_signature = None
        self._load_notes = []
        self._conversion_notes = []
        self._last_was_failure = False
        self._restoring = False
        self._sheet_choice = ""
        self._running = None
        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._debounce.timeout.connect(self._refresh_preview)
        # Orange prohíbe arrancar una tarea desde on_done.
        self._after_preview = QTimer(self, singleShot=True, interval=0)
        self._after_preview.timeout.connect(self.commit.deferred)

        self._build_source_box()
        self._build_format_box()
        self._build_advanced_box()
        gui.rubber(self.controlArea)
        self.cancel_button = gui.button(self.buttonsArea, self, "Cancelar",
                                        callback=self.cancel_import, autoDefault=False)
        self.cancel_button.setVisible(False)
        gui.auto_apply(self.buttonsArea, self, "auto_apply")
        self._build_main_area()

        self.set_file_list()
        self.url_combo.addItems(self.recent_urls)
        self.url_combo.setEditText(self.url_text)
        self._load_config_into_controls()
        self._sync_source_controls()
        self.info.set_output_summary(self.info.NoOutput)
        QTimer.singleShot(0, self._restore_source)

    # ------------------------------------------------------ construcción ----

    def _build_source_box(self):
        box = gui.vBox(self.controlArea, "Origen")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.file_radio = QRadioButton("Archivo")
        self.url_radio = QRadioButton("URL")
        group = QButtonGroup(self)
        group.addButton(self.file_radio, self.LOCAL)
        group.addButton(self.url_radio, self.URL)
        (self.url_radio if self.source_kind == self.URL else self.file_radio).setChecked(True)
        self.file_radio.toggled.connect(self._on_source_kind)

        self.file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.file_combo.setMinimumContentsLength(14)
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.file_combo.activated[int].connect(self._on_recent_selected)
        browse = QToolButton(self, text="…", toolTip="Abrir un archivo")
        browse.clicked.connect(self.browse)
        reload_button = QToolButton(self, toolTip="Volver a leer el origen")
        reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.clicked.connect(self.reload_source)

        self.url_combo = QComboBox(self, editable=True)
        self.url_combo.setInsertPolicy(QComboBox.NoInsert)
        self.url_combo.lineEdit().setPlaceholderText("https://…")
        self.url_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.url_combo.setMinimumContentsLength(14)
        self.url_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.url_combo.lineEdit().returnPressed.connect(self._on_url_entered)
        self.url_combo.activated[int].connect(lambda _: self._on_url_entered())

        grid.addWidget(self.file_radio, 0, 0)
        grid.addWidget(self.file_combo, 0, 1)
        grid.addWidget(browse, 0, 2)
        grid.addWidget(reload_button, 0, 3)
        grid.addWidget(self.url_radio, 1, 0)
        grid.addWidget(self.url_combo, 1, 1, 1, 3)
        grid.setColumnStretch(1, 1)

    def _build_format_box(self):
        box = gui.vBox(self.controlArea, "Formato")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.format_combo = QComboBox(self)
        for key, label in FORMAT_LABELS.items():
            self.format_combo.addItem(label, key)
        self.format_combo.activated[int].connect(lambda _: self._on_options_edited())
        self.sheet_label = QLabel("Hoja:")
        self.sheet_combo = QComboBox(self)
        self.sheet_combo.activated[int].connect(self._on_sheet_selected)
        grid.addWidget(QLabel("Tipo:"), 0, 0)
        grid.addWidget(self.format_combo, 0, 1)
        grid.addWidget(self.sheet_label, 1, 0)
        grid.addWidget(self.sheet_combo, 1, 1)
        grid.setColumnStretch(1, 1)
        self._show_sheet_row(False)

    def _on_sheet_selected(self, _index):
        self._sheet_choice = self.sheet_combo.currentData() or ""
        self._on_options_edited()

    def _show_sheet_row(self, visible):
        self.sheet_label.setVisible(visible)
        self.sheet_combo.setVisible(visible)

    def _build_advanced_box(self):
        self.advanced_toggle = QToolButton(
            self, text="Importación avanzada", checkable=True,
            toolButtonStyle=Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.setChecked(bool(self.advanced_open))
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)
        self.controlArea.layout().addWidget(self.advanced_toggle)

        self.advanced_panel = QFrame(self)
        grid = QGridLayout(self.advanced_panel)
        grid.setContentsMargins(4, 0, 0, 0)
        self.controlArea.layout().addWidget(self.advanced_panel)

        def combo(items, editable=False):
            widget = QComboBox(self, editable=editable)
            if editable:
                widget.setInsertPolicy(QComboBox.NoInsert)
            for value, label in items:
                widget.addItem(label, value)
            widget.activated[int].connect(lambda _: self._on_options_edited())
            if editable:
                widget.lineEdit().editingFinished.connect(self._on_options_edited)
            return widget

        self.encoding_combo = combo(
            [(AUTO, "Automática"), ("utf-8", "UTF-8"), ("utf-8-sig", "UTF-8 con BOM"),
             ("utf-16", "UTF-16"), ("cp1252", "Windows-1252"), ("latin-1", "ISO-8859-1"),
             ("iso-8859-15", "ISO-8859-15"), ("mac_roman", "Mac Roman")], editable=True)
        self.delimiter_combo = combo(
            [(AUTO, "Automático"), (",", ","), (";", ";"), ("\t", "tabulador"),
             ("|", "|"), (" ", "espacio"), (":", ":")], editable=True)
        self.quote_combo = combo([('"', '"'), ("'", "'"), ("", "Sin comillas")])
        self.decimal_combo = combo([(AUTO, "Automático"), (".", "."), (",", ",")])
        self.thousands_combo = combo([(AUTO, "Automático"), ("", "Ninguno"), (".", "."),
                                      (",", ","), (" ", "espacio"), ("'", "'")])
        self.header_combo = combo([(HEADER_AUTO, "Automática"), (HEADER_NONE, "Sin cabecera"),
                                   (1, "En la fila…")])
        self.header_spin = QSpinBox(self, minimum=1, maximum=1000000)
        self.header_spin.setKeyboardTracking(False)
        self.header_spin.valueChanged.connect(self._on_options_edited)
        self.skip_spin = QSpinBox(self, minimum=0, maximum=100000000)
        self.skip_spin.setKeyboardTracking(False)
        self.skip_spin.valueChanged.connect(self._on_options_edited)
        self.max_rows_spin = QSpinBox(self, minimum=0, maximum=2000000000, specialValueText="Todas")
        self.max_rows_spin.setKeyboardTracking(False)
        self.max_rows_spin.setSingleStep(1000)
        self.max_rows_spin.valueChanged.connect(self._on_options_edited)
        self.na_edit = QLineEdit(self)
        self.na_edit.setToolTip("Textos que cuentan como valor perdido, separados por comas.")
        self.na_edit.editingFinished.connect(self._on_options_edited)
        self.comment_edit = QLineEdit(self, maxLength=1, placeholderText="ninguno")
        self.comment_edit.editingFinished.connect(self._on_options_edited)
        self.date_combo = combo([(AUTO, "Automático"), ("dmy", "Día/mes/año"),
                                 ("mdy", "Mes/día/año")])
        self.download_spin = QSpinBox(self, minimum=1, maximum=100000, suffix=" MB")
        self.download_spin.setKeyboardTracking(False)
        self.download_spin.valueChanged.connect(self._on_options_edited)

        rows = (("Codificación:", self.encoding_combo),
                ("Separador:", self.delimiter_combo),
                ("Comillas:", self.quote_combo),
                ("Decimal:", self.decimal_combo),
                ("Miles:", self.thousands_combo),
                ("Saltar filas:", self.skip_spin),
                ("Cabecera:", self.header_combo),
                ("Fila de cabecera:", self.header_spin),
                ("Valores perdidos:", self.na_edit),
                ("Comentarios:", self.comment_edit),
                ("Máximo de filas:", self.max_rows_spin),
                ("Orden de fechas:", self.date_combo),
                ("Descarga máxima:", self.download_spin))
        self.header_row_label = None
        for position, (text, widget) in enumerate(rows):
            label = QLabel(text)
            grid.addWidget(label, position, 0)
            grid.addWidget(widget, position, 1)
            if widget is self.header_spin:
                self.header_row_label = label
        reset = QToolButton(self, text="Restablecer opciones")
        reset.clicked.connect(self.reset_options)
        grid.addWidget(reset, len(rows), 0, 1, 2)
        grid.setColumnStretch(1, 1)
        self._on_advanced_toggled(bool(self.advanced_open))

    def _build_main_area(self):
        layout = self.mainArea.layout()
        summary = QWidget(self)
        strip = QHBoxLayout(summary)
        strip.setContentsMargins(4, 4, 4, 2)
        self.rows_tile = StatTile("filas")
        self.columns_tile = StatTile("columnas")
        self.missing_tile = StatTile("perdidos")
        for tile in (self.rows_tile, self.columns_tile, self.missing_tile):
            strip.addWidget(tile)
        self.types_label = QLabel("")
        self.types_label.setTextFormat(Qt.RichText)
        strip.addWidget(self.types_label, 1)
        layout.addWidget(summary)
        self.type_bar = TypeBar(self)
        layout.addWidget(self.type_bar)
        self.status_label = QLabel("Elige un archivo, escribe una URL o arrastra un "
                                   "archivo a esta ventana.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Vertical, self)
        layout.addWidget(splitter, 1)

        columns_panel = QWidget(self)
        columns_layout = QVBoxLayout(columns_panel)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        bulk = QHBoxLayout()
        title = QLabel("<b>Columnas</b>")
        bulk.addWidget(title)
        bulk.addStretch(1)
        bulk.addWidget(QLabel("Cambiar las seleccionadas:"))
        self.bulk_type = QComboBox(self)
        self.bulk_type.addItem("Tipo…", None)
        for kind, label in TYPE_LABELS.items():
            self.bulk_type.addItem(_type_icon(kind), label, kind)
        self.bulk_type.activated[int].connect(self._on_bulk_type)
        self.bulk_role = QComboBox(self)
        self.bulk_role.addItem("Rol…", None)
        for role, label in ROLE_LABELS.items():
            self.bulk_role.addItem(label, role)
        self.bulk_role.activated[int].connect(self._on_bulk_role)
        restore = QToolButton(self, text="Restaurar lo detectado")
        restore.clicked.connect(self.reset_columns)
        bulk.addWidget(self.bulk_type)
        bulk.addWidget(self.bulk_role)
        bulk.addWidget(restore)
        columns_layout.addLayout(bulk)

        self.columns_model = ColumnsModel(self)
        self.columns_model.edited.connect(self._on_columns_edited)
        self.columns_view = QTableView(self)
        self.columns_view.setModel(self.columns_model)
        self.columns_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.columns_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.columns_view.setEditTriggers(QAbstractItemView.CurrentChanged
                                          | QAbstractItemView.SelectedClicked)
        self.columns_view.verticalHeader().setVisible(False)
        self.columns_view.setWordWrap(False)
        self.columns_view.setItemDelegateForColumn(
            ColumnsModel.TYPE, ComboDelegate(TYPE_LABELS, _type_icon, self.columns_view))
        self.columns_view.setItemDelegateForColumn(
            ColumnsModel.ROLE, ComboDelegate(ROLE_LABELS, None, self.columns_view))
        self.columns_view.setItemDelegateForColumn(
            ColumnsModel.MISSING, MissingDelegate(self.columns_view))
        self.columns_view.horizontalHeader().setStretchLastSection(True)
        columns_layout.addWidget(self.columns_view)
        splitter.addWidget(columns_panel)

        preview_panel = QWidget(self)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_title = QLabel(f"<b>Vista previa</b> (primeras {engine.PREVIEW_ROWS} filas)")
        preview_layout.addWidget(self.preview_title)
        self.preview_model = PreviewModel(self)
        self.preview_view = QTableView(self)
        self.preview_view.setModel(self.preview_model)
        self.preview_view.setWordWrap(False)
        self.preview_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_view.verticalHeader().setDefaultSectionSize(22)
        self.preview_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        preview_layout.addWidget(self.preview_view)
        splitter.addWidget(preview_panel)
        splitter.setSizes([260, 300])

    def sizeHint(self):
        return QSize(1100, 680)

    # --------------------------------------------------------- utilidades ----

    def _on_advanced_toggled(self, opened):
        self.advanced_open = bool(opened)
        self.advanced_panel.setVisible(bool(opened))
        self.advanced_toggle.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)

    def _source_key(self):
        if self.source_kind == self.URL:
            return self.url_text.strip() or None
        path = self.last_path()
        return os.path.basename(path) if path else None

    def _config(self):
        key = self._source_key()
        if key is None:
            return {}
        return self.source_configs.get(key, {})

    def _save_config(self, config):
        key = self._source_key()
        if key is None:
            return
        self.source_configs.pop(key, None)
        self.source_configs[key] = config
        while len(self.source_configs) > MAX_SAVED_CONFIGS:
            del self.source_configs[next(iter(self.source_configs))]

    def _overrides(self):
        overrides = self._config().get("columns")
        return overrides if isinstance(overrides, dict) else {}

    def _combo_value(self, combo):
        text = combo.currentText()
        index = combo.findText(text)
        if index >= 0:
            return combo.itemData(index)
        # Valor escrito a mano, o «Automático (detectado…)».
        if combo.currentIndex() == 0 and text.startswith("Autom"):
            return AUTO
        return text

    def _set_combo_value(self, combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        elif combo.isEditable():
            combo.setEditText(str(value))

    def _options_from_controls(self):
        options = Options()
        options.format = self.format_combo.currentData()
        options.sheet = self._sheet_choice
        options.encoding = self._combo_value(self.encoding_combo)
        delimiter = self._combo_value(self.delimiter_combo)
        options.delimiter = "\t" if delimiter in ("tabulador", "\\t") else (
            " " if delimiter == "espacio" else delimiter)
        options.quotechar = self.quote_combo.currentData()
        options.decimal = self.decimal_combo.currentData()
        options.thousands = self.thousands_combo.currentData()
        options.skip_rows = self.skip_spin.value()
        header = self.header_combo.currentData()
        options.header = self.header_spin.value() if header == 1 else header
        options.na_values = self.na_edit.text()
        options.comment = self.comment_edit.text()
        options.max_rows = self.max_rows_spin.value()
        options.date_order = self.date_combo.currentData()
        options.max_download_mb = self.download_spin.value()
        return options

    def _load_config_into_controls(self):
        """Muestra en los controles las opciones guardadas para este origen."""
        options = Options.from_dict(self._config().get("options"))
        controls = (self.format_combo, self.sheet_combo, self.encoding_combo,
                    self.delimiter_combo, self.quote_combo, self.decimal_combo,
                    self.thousands_combo, self.header_combo, self.header_spin,
                    self.skip_spin, self.na_edit, self.comment_edit, self.max_rows_spin,
                    self.date_combo, self.download_spin)
        blockers = [widget.blockSignals(True) for widget in controls]
        self._set_combo_value(self.format_combo, options.format)
        self.sheet_combo.clear()
        self._sheet_choice = options.sheet
        if options.sheet:
            self.sheet_combo.addItem(options.sheet, options.sheet)
        self._set_combo_value(self.encoding_combo, options.encoding)
        self._set_combo_value(self.delimiter_combo, options.delimiter)
        self._set_combo_value(self.quote_combo, options.quotechar)
        self._set_combo_value(self.decimal_combo, options.decimal)
        self._set_combo_value(self.thousands_combo, options.thousands)
        self.skip_spin.setValue(options.skip_rows)
        self.max_rows_spin.setValue(options.max_rows)
        self.na_edit.setText(options.na_values)
        self.comment_edit.setText(options.comment)
        self._set_combo_value(self.date_combo, options.date_order)
        self.download_spin.setValue(options.max_download_mb)
        if options.header > 0:
            self._set_combo_value(self.header_combo, 1)
            self.header_spin.setValue(options.header)
        else:
            self._set_combo_value(self.header_combo, options.header)
        for widget, blocked in zip(controls, blockers):
            widget.blockSignals(blocked)
        self._sync_header_row()

    def _sync_header_row(self):
        visible = self.header_combo.currentData() == 1
        self.header_spin.setVisible(visible)
        if self.header_row_label is not None:
            self.header_row_label.setVisible(visible)

    def _sync_source_controls(self):
        local = self.source_kind == self.LOCAL
        self.file_combo.setEnabled(local)
        self.url_combo.setEnabled(not local)

    def _temporary_directory(self):
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="orange-file-import-")
        return self._temp_dir

    def _request(self):
        if self.source_kind == self.URL:
            target = self.url_text.strip()
        else:
            target = self.last_path() or ""
        if not target:
            return None
        return Request(self.source_kind, target, self._options_from_controls(),
                       self._temporary_directory(), self._download)

    def _clear_messages(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

    def _set_busy(self, busy):
        self.cancel_button.setVisible(busy)

    def _clear_views(self, message):
        self._preview = None
        self.columns_model.set_columns([], [], 0)
        self.preview_model.set_cells([], [])
        self.rows_tile.value.setText("—")
        self.columns_tile.value.setText("—")
        self.missing_tile.value.setText("—")
        self.types_label.setText("")
        self.type_bar.set_counts({})
        self.status_label.setText(message)

    # --------------------------------------------------------------- orígenes ----

    def _restore_source(self):
        if self._request() is None:
            self.Outputs.data.send(None)
            return
        self._restoring = True
        self._refresh_preview()

    def _on_source_kind(self, _checked=False):
        kind = self.LOCAL if self.file_radio.isChecked() else self.URL
        if kind == self.source_kind:
            return
        self.source_kind = kind
        self._sync_source_controls()
        self._source_changed()

    def _forget_download(self):
        download, self._download = self._download, None
        if download is not None:
            _remove_file(download.path)

    def _source_changed(self):
        self.cancel_import(quiet=True)
        self._forget_download()
        self._loaded = self._loaded_signature = None
        self._load_config_into_controls()
        self._refresh_preview()

    def _on_recent_selected(self, index):
        self.select_file(index)
        self.source_kind = self.LOCAL
        self.file_radio.setChecked(True)
        self._sync_source_controls()
        self._source_changed()

    def browse(self):
        start = os.path.dirname(self.last_path() or "") or os.path.expanduser("~/")
        path, _ = QFileDialog.getOpenFileName(self, "Abrir un archivo de datos", start, FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        self.add_path(path)
        self.source_kind = self.LOCAL
        self.file_radio.setChecked(True)
        self._sync_source_controls()
        self._source_changed()

    def _on_url_entered(self):
        self.open_url(self.url_combo.currentText())

    def open_url(self, url):
        url = url.strip()
        self.url_text = url
        self.source_kind = self.URL
        self.url_radio.setChecked(True)
        self.url_combo.setEditText(url)
        self._sync_source_controls()
        if url:
            if url in self.recent_urls:
                self.recent_urls.remove(url)
            self.recent_urls.insert(0, url)
            del self.recent_urls[20:]
            blocked = self.url_combo.blockSignals(True)
            self.url_combo.clear()
            self.url_combo.addItems(self.recent_urls)
            self.url_combo.setEditText(url)
            self.url_combo.blockSignals(blocked)
        self._source_changed()

    def reload_source(self):
        self._debounce.stop()
        self.cancel_import(quiet=True)
        self._forget_download()
        self._loaded = self._loaded_signature = None
        self._refresh_preview()

    def canDropUrl(self, url: QUrl) -> bool:
        return url.isLocalFile() or url.scheme().lower() in ("http", "https")

    def handleDroppedUrl(self, url: QUrl) -> None:
        if url.isLocalFile():
            self.open_path(url.toLocalFile())
        else:
            self.open_url(url.toString())

    def workflowEnvChanged(self, key, value, oldvalue):
        self.update_file_list(key, value, oldvalue)

    # --------------------------------------------------------------- opciones ----

    def _on_options_edited(self):
        self._sync_header_row()
        self._save_options()
        self._debounce.start()

    def _save_options(self):
        config = dict(self._config())
        options = self._options_from_controls()
        config["options"] = options.to_dict()
        self._save_config(config)

    def reset_options(self):
        config = dict(self._config())
        config.pop("options", None)
        self._save_config(config)
        self._load_config_into_controls()
        self._debounce.start()

    def _refresh_preview(self):
        self._debounce.stop()
        self._after_preview.stop()
        request = self._request()
        if request is None:
            self.cancel()
            self._clear_messages()
            self._clear_views("Elige un archivo, escribe una URL o arrastra un archivo "
                              "a esta ventana.")
            self._send(None)
            return
        self._clear_messages()
        self._last_was_failure = False
        self.status_label.setText("Leyendo…")
        self._set_busy(True)
        self._running = "preview"
        self.start(task_preview, request, dict(self._overrides()))

    def cancel_import(self, quiet=False):
        self._debounce.stop()
        self._after_preview.stop()
        was_busy = self.task is not None
        self.cancel()
        self._set_busy(False)
        if was_busy and not quiet:
            self.status_label.setText("Importación cancelada.")
            self.Information.details("Importación cancelada. Pulsa recargar para reintentarlo.")

    # --------------------------------------------------------------- columnas ----

    def _on_columns_edited(self, rows):
        config = dict(self._config())
        overrides = dict(config.get("columns") or {})
        for row in rows:
            meta = self.columns_model.columns[row]
            choice = self.columns_model.choices[row]
            changed = {}
            if choice.name != meta.name:
                changed["name"] = choice.name
            if choice.type != meta.type:
                changed["type"] = choice.type
            if choice.role != meta.role:
                changed["role"] = choice.role
            key = column_key(meta)
            if changed:
                overrides[key] = changed
            else:
                overrides.pop(key, None)
        config["columns"] = overrides
        config["known"] = [column_key(meta) for meta in self.columns_model.columns]
        self._save_config(config)
        self._refresh_preview_columns(rows)
        self._refresh_summary()
        self.commit.deferred()

    def _selected_rows(self):
        return sorted({index.row() for index in self.columns_view.selectionModel().selectedRows()})

    def _on_bulk_type(self, index):
        value = self.bulk_type.itemData(index)
        self.bulk_type.setCurrentIndex(0)
        if value:
            self.columns_model.apply_to(self._selected_rows(), "type", value)

    def _on_bulk_role(self, index):
        value = self.bulk_role.itemData(index)
        self.bulk_role.setCurrentIndex(0)
        if value:
            self.columns_model.apply_to(self._selected_rows(), "role", value)

    def reset_columns(self):
        config = dict(self._config())
        config.pop("columns", None)
        config.pop("known", None)
        self._save_config(config)
        current = self._loaded if self._loaded is not None else self._preview
        if current is None:
            return
        choices = engine.default_choices(current.columns)
        self.columns_model.set_columns(current.columns, choices, current.n_rows)
        self.preview_model.set_cells(engine.preview_columns(current, choices), list(choices))
        self._refresh_summary()
        self.commit.deferred()

    def _check_columns_changed(self, columns):
        self.Warning.columns_changed.clear()
        config = self._config()
        known = config.get("known") or []
        overrides = self._overrides()
        if not overrides or not known:
            return
        current = {column_key(meta) for meta in columns}
        gone = [key for key in overrides if key not in current]
        added = [key for key in current if key not in set(known)]
        if not gone and not added:
            return
        parts = []
        if gone:
            parts.append(f"ya no existen {len(gone)} columnas configuradas "
                         f"({', '.join(gone[:3])}{'…' if len(gone) > 3 else ''})")
        if added:
            parts.append(f"hay {len(added)} columnas nuevas "
                         f"({', '.join(sorted(added)[:3])}{'…' if len(added) > 3 else ''})")
        self.Warning.columns_changed("El archivo ha cambiado desde que se configuró: "
                                     + " y ".join(parts)
                                     + ". Revisa la tabla de columnas.")

    def _set_columns(self, loaded):
        overrides = self._overrides()
        self.columns_model.set_columns(loaded.columns,
                                       apply_overrides(loaded.columns, overrides), loaded.n_rows)
        self._check_columns_changed(loaded.columns)
        self.columns_view.resizeColumnsToContents()

    def _refresh_preview_columns(self, rows):
        current = self._preview if self._preview is not None else self._loaded
        model = self.columns_model
        if current is None or len(model.choices) != len(self.preview_model.cells):
            return
        for row in rows:
            self.preview_model.set_column(
                row, engine.preview_column(current, model.columns[row], model.choices[row]))

    def _refresh_summary(self):
        current = self._loaded if self._loaded is not None else self._preview
        if current is None:
            return
        choices = self.columns_model.choices
        pairs = [(m, c) for m, c in zip(self.columns_model.columns, choices) if c.role != SKIP]
        counts = {kind: sum(1 for _, c in pairs if c.type == kind) for kind in engine.TYPES}
        self.type_bar.set_counts(counts)
        total = current.n_rows * max(len(pairs), 1)
        missing = sum(m.n_missing for m, _ in pairs)
        share = f"{100 * missing / max(total, 1):.1f} %".replace(".", ",")
        complete = current is self._loaded
        self.rows_tile.value.setText(format_int(current.n_rows) if complete
                                     else f"≥ {format_int(current.n_rows)}")
        self.columns_tile.value.setText(format_int(len(pairs)))
        self.missing_tile.value.setText(share)
        legend = [f'<span style="color:{TYPE_COLORS[kind]}">■</span> '
                  f'{counts[kind]} {(TYPE_SINGULARS if counts[kind] == 1 else TYPE_PLURALS)[kind]}'
                  for kind in engine.TYPES if counts[kind]]
        self.types_label.setText("&nbsp;&nbsp;".join(legend))
        self._update_detected_labels(current)

    def _update_detected_labels(self, current):
        def label(combo, text):
            combo.setItemText(0, text)
        label(self.format_combo, f"Automático ({FORMAT_LABELS[current.format]})")
        if current.encoding:
            label(self.encoding_combo, f"Automática ({current.encoding})")
        if current.delimiter:
            label(self.delimiter_combo, f"Automático ({_delimiter_text(current.delimiter)})")
        label(self.decimal_combo, f"Automático ({current.decimal})")
        label(self.thousands_combo, "Automático (ninguno)" if not current.thousands
              else f"Automático ({_delimiter_text(current.thousands)})")
        label(self.header_combo, "Automática (sin cabecera)" if current.header_row == HEADER_NONE
              else f"Automática (fila {current.header_row})")

    def _update_sheets(self, loaded):
        visible = loaded.format in ("xlsx", "xls") and bool(loaded.sheets)
        self._show_sheet_row(visible)
        if not visible:
            return
        blocked = self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        for sheet in loaded.sheets:
            self.sheet_combo.addItem(sheet, sheet)
        self.sheet_combo.setCurrentIndex(max(self.sheet_combo.findData(loaded.sheet), 0))
        self.sheet_combo.blockSignals(blocked)

    def _show_notes(self, notes):
        self.Warning.notes.clear()
        if notes:
            self.Warning.notes("\n".join(dict.fromkeys(notes)))

    def _show_download_notes(self, download):
        self.Information.details.clear()
        if download is not None and download.notes:
            self.Information.details("\n".join(download.notes))

    # ---------------------------------------------------------- carga completa ----

    @gui.deferred
    def commit(self):
        self.Error.failed.clear()
        self.Error.unexpected.clear()
        request = self._request()
        if request is None or self._last_was_failure:
            self._send(None)
            return
        self._set_busy(True)
        reusable = False
        if self._loaded is not None and self._loaded_signature is not None:
            try:
                path = self._loaded_signature[0]
                reusable = _signature(path, request.options) == self._loaded_signature
            except OSError:
                reusable = False
        if reusable and len(self.columns_model.choices) == len(self._loaded.columns):
            self._running = "convert"
            self.start(task_convert, self._loaded, [ColumnChoice(c.name, c.type, c.role)
                                                    for c in self.columns_model.choices])
        else:
            self._loaded = self._loaded_signature = None
            self._running = "load"
            self.start(task_load, request, dict(self._overrides()))

    def _send(self, table):
        if table is None:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            self.info.set_output_summary(len(table), format_summary_details(table))
        self.Outputs.data.send(table)

    def on_done(self, result):
        self._set_busy(False)
        if isinstance(result, PreviewResult):
            self._on_preview_done(result)
        elif isinstance(result, LoadResult):
            self._on_load_done(result)
        elif isinstance(result, ConvertResult):
            self._conversion_notes = result.notes
            self._show_notes(self._load_notes + result.notes)
            self._send(result.table)

    def _on_preview_done(self, result):
        self._download = result.download
        self._display_path = result.path
        self._preview = result.loaded
        self._restoring = False
        self._loaded = self._loaded_signature = None
        self._update_sheets(result.loaded)
        self._set_columns(result.loaded)
        self.preview_model.set_cells(result.cells, list(self.columns_model.choices))
        self._refresh_summary()
        self._show_notes(result.loaded.notes)
        self._show_download_notes(result.download)
        self.status_label.setText(self._describe_source(result))
        self._after_preview.start()

    def _describe_source(self, result):
        loaded = result.loaded
        size = ""
        try:
            bytes_on_disk = os.path.getsize(result.path)
            if bytes_on_disk < 1024 * 1024:
                size = f" · {format_int(max(bytes_on_disk // 1024, 1))} KB"
            else:
                size = f" · {bytes_on_disk / (1024 * 1024):.1f} MB".replace(".", ",")
        except OSError:
            pass
        extra = f" · hoja «{loaded.sheet}»" if loaded.sheet else ""
        return f"{os.path.basename(result.display_name)} · {FORMAT_LABELS[loaded.format]}{size}{extra}"

    def _on_load_done(self, result):
        self._download = result.download
        self._loaded = result.loaded
        self._loaded_signature = result.signature
        self._preview = None
        self._load_notes = list(result.loaded.notes)
        self._conversion_notes = result.notes
        self._update_sheets(result.loaded)
        self._set_columns(result.loaded)
        self.preview_model.set_cells(result.cells, list(self.columns_model.choices))
        self._refresh_summary()
        self._show_notes(self._load_notes + result.notes)
        self.status_label.setText(self._describe_source(result))
        self._send(result.table)

    def on_exception(self, ex):
        self._set_busy(False)
        if isinstance(ex, ImportCancelled):
            return
        if isinstance(ex, ImportFailure):
            self.Error.failed(str(ex))
            self._download = getattr(ex, "download", self._download)
            if isinstance(ex, engine.DecompressionLimit):
                self._forget_download()
        else:
            self.Error.unexpected(f"{type(ex).__name__}: {ex}")
        if self._running == "preview":
            self._last_was_failure = True
            self._clear_views("No se ha podido leer el origen. Revisa el mensaje del "
                              "widget y las opciones.")
        elif self._running == "load":
            self._loaded = self._loaded_signature = None
        self._send(None)

    def onDeleteWidget(self):
        self._debounce.stop()
        self._after_preview.stop()
        self.shutdown()
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWFileImport).run()
