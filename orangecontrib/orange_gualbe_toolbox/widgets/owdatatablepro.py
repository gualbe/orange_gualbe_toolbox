# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Widget Data Table Pro: varias tablas en pestañas, editables, con buscar y reemplazar.

La lógica de datos vive en _dtp_engine.py (sin Qt). Aquí están la vista perezosa, que lee
cada celda del motor solo cuando hay que pintarla, las pestañas, las salidas numeradas y
el estado guardado con el workflow.
"""
from typing import Optional

import numpy as np

from AnyQt.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, QSize, Qt, QTimer, Signal
from AnyQt.QtGui import QBrush, QColor, QFont, QKeySequence
from AnyQt.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMenu, QPushButton, QShortcut, QStyledItemDelegate, QTableView,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from Orange.data import Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, MultiInput, Output, OWWidget

from orangecontrib.orange_gualbe_toolbox.widgets import _dtp_engine as engine

MAX_OUTPUTS = 5


def _comparable(entry):
    """Solo se comparan huellas del mismo algoritmo; las de otra versión cuentan como ausentes.

    Se mira la versión guardada y también el formato (longitud): la 0.5.1 guardaba un sha1 de 16
    caracteres sin versión. Un cambio de formato nunca debe tomarse por «datos distintos».
    """
    value = entry.get("fingerprint")
    return (isinstance(value, str) and len(value) == engine.FINGERPRINT_LENGTH
            and entry.get("fingerprint_version") == engine.FINGERPRINT_VERSION)


def _same_fingerprint(entry, state):
    return _comparable(entry) and entry.get("fingerprint") == state.fingerprint
BADGES = "①②③④⑤"
SCOPE_ALL, SCOPE_COLUMNS, SCOPE_ROWS = range(3)
SCOPES = ("Toda la tabla", "Columnas seleccionadas", "Filas seleccionadas")
BarRole = Qt.UserRole + 1
FILTER_DEBOUNCE_MS = 300

ROLE_TINT = {engine.TARGET: QColor(42, 120, 214, 28), engine.META: QColor(120, 118, 110, 26)}
EDITED = QColor(255, 196, 0, 70)
MATCH = QColor(235, 104, 52, 80)
CURRENT_MATCH = QColor(235, 104, 52, 190)
UNKNOWN_TEXT = QColor(140, 138, 132)


# ------------------------------------------------------------------- modelo ----

class StateModel(QAbstractTableModel):
    """Modelo perezoso sobre un TableState: no guarda celdas, las lee al pintarlas."""

    edit_failed = Signal(str)
    edited = Signal()

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.rows = np.arange(0)
        self.matches = {}          # cid → vector booleano por fila original
        self.current = None        # (fila original, cid)
        self._bar_cache = {}

    @property
    def state(self):
        return self.page.state

    def reset(self):
        self.beginResetModel()
        self.rows = self.page.visible_rows()
        self._bar_cache = {}
        self.endResetModel()

    def repaint(self):
        """Los datos han cambiado pero no las filas: repintar sin reiniciar el modelo."""
        self._bar_cache = {}
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))
        if self.columnCount():
            self.headerDataChanged.emit(Qt.Horizontal, 0, self.columnCount() - 1)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.state is None else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.state is None else len(self.state.columns)

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def cell(self, index):
        return int(self.rows[index.row()]), self.state.columns[index.column()]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.state is None:
            return None
        row, column = self.cell(index)
        if role == Qt.DisplayRole:
            return column.text(row)
        if role == Qt.EditRole:
            return column.edit_text(row)  # con toda la precisión: confirmar sin cambios no redondea
        if role == Qt.TextAlignmentRole:
            if column.kind in (engine.NUMERIC, engine.DATETIME):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole:
            return self._background(row, column)
        if role == Qt.ForegroundRole:
            color = self.page.user_color(row, column.cid, 1)
            if color is not None:
                return QBrush(color)
            if self._unknown(row, column):
                return QBrush(UNKNOWN_TEXT)
            return None
        if role == Qt.FontRole and self._unknown(row, column):
            font = QFont()
            font.setItalic(True)
            return font
        if role == BarRole:
            return self._bar(row, column)
        if role == Qt.ToolTipRole:
            if column.edited is not None and column.edited[row] and self.page.widget.highlight_edits:
                return "Celda modificada en este widget."
        return None

    @staticmethod
    def _unknown(row, column):
        value = column.values[row]
        return value == "" if column.kind == engine.STRING else value != value

    def _background(self, row, column):
        if self.current == (row, column.cid):
            return QBrush(CURRENT_MATCH)
        mask = self.matches.get(column.cid)
        if mask is not None and mask[row]:
            return QBrush(MATCH)
        color = self.page.user_color(row, column.cid, 0)
        if color is not None:
            return QBrush(color)
        if self.page.widget.highlight_edits and column.edited is not None and column.edited[row]:
            return QBrush(EDITED)
        if self.page.widget.color_categories and column.kind == engine.CATEGORICAL:
            value = column.values[row]
            if value == value:
                colors = column.format_var().colors
                if int(value) < len(colors):
                    r, g, b = colors[int(value)][:3]
                    return QBrush(QColor(int(r), int(g), int(b), 70))
        tint = ROLE_TINT.get(column.role)
        return QBrush(tint) if tint is not None else None

    def _bar(self, row, column):
        if not self.page.widget.show_bars or column.kind != engine.NUMERIC:
            return None
        key = (column.cid, self.state.version)
        if key not in self._bar_cache:
            finite = column.values[~np.isnan(column.values)]
            self._bar_cache[key] = (float(finite.min()), float(finite.max())) if len(finite) else None
        bounds = self._bar_cache[key]
        value = column.values[row]
        if bounds is None or value != value:
            return None
        low, high = bounds
        return 1.0 if high == low else (value - low) / (high - low)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if self.state is None:
            return None
        if orientation == Qt.Vertical:
            if role == Qt.DisplayRole and section < len(self.rows):
                return str(int(self.rows[section]) + 1)
            return None
        if section >= len(self.state.columns):
            return None
        column = self.state.columns[section]
        if role == Qt.DisplayRole:
            return column.name
        if role == Qt.DecorationRole:
            return gui.attributeIconDict[column.format_var()]
        if role == Qt.ToolTipRole:
            lines = column.stats()
            if column.name != column.cid:
                lines.append(f"Nombre original: {column.cid}")
            return "\n".join(lines)
        if role == Qt.BackgroundRole:
            tint = ROLE_TINT.get(column.role)
            return QBrush(tint) if tint is not None else None
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        row, column = self.cell(index)
        try:
            changed = self.state.edit(row, column.cid, value)
        except engine.OperationError as error:
            self.edit_failed.emit(str(error))
            return False
        if changed:
            self.headerDataChanged.emit(Qt.Horizontal, index.column(), index.column())
            self.dataChanged.emit(self.index(0, index.column()), self.index(self.rowCount() - 1, index.column()))
            self.edited.emit()
        return changed


class BarDelegate(QStyledItemDelegate):
    """Pinta una barra proporcional al valor en las columnas numéricas, si se pide."""

    def paint(self, painter, option, index):
        fraction = index.data(BarRole)
        if fraction is not None:
            rect = option.rect.adjusted(1, option.rect.height() - 5, -1, -1)
            rect.setWidth(max(int(rect.width() * fraction), 1))
            painter.fillRect(rect, QColor(42, 120, 214, 150))
        super().paint(painter, option, index)


# -------------------------------------------------------------------- pestaña ----

class Page(QWidget):
    """Una pestaña: una tabla de entrada con su estado, su vista y sus colores."""

    def __init__(self, widget, key):
        super().__init__(widget)
        self.widget = widget
        self.key = key
        self.table: Optional[Table] = None
        self.state: Optional[engine.TableState] = None
        self.identifier = ""
        self.colors = {"cells": {}, "rows": {}, "cols": {}}
        self.filter_text = ""
        self._filter_cache = None
        self.matches = []
        self.match_index = -1
        self._output = None
        self._selected = None
        self._ranges = None
        self.saved_key = None   # clave con la que se guardaron sus cambios en el workflow
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.empty_label = QLabel("Sin datos en esta entrada.", self)
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)
        self.model = StateModel(self, self)
        self.view = QTableView(self)
        self.view.setModel(self.model)
        self.view.setItemDelegate(BarDelegate(self.view))
        self.view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                  | QAbstractItemView.AnyKeyPressed)
        self.view.setWordWrap(False)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(widget.cell_menu)
        vertical = self.view.verticalHeader()
        vertical.setSectionResizeMode(QHeaderView.Fixed)
        vertical.setDefaultSectionSize(max(self.fontMetrics().height() + 6, 20))
        header = self.view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.AscendingOrder)
        header.setHighlightSections(True)
        header.sectionClicked.connect(self._header_clicked)
        header.sectionMoved.connect(lambda *_: widget.changed(self, data=False))
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(lambda pos: widget.header_menu(self, pos))
        layout.addWidget(self.view)
        self.view.selectionModel().selectionChanged.connect(self._selection_moved)
        self.model.modelReset.connect(self._selection_moved)
        self.model.edit_failed.connect(widget.show_edit_error)
        self.model.edited.connect(lambda: widget.changed(self))
        self._last_sort = (None, None)
        self.set_table(None)

    # ------------------------------------------------------------ datos ----

    def set_table(self, table, identifier=""):
        self.table = table
        self.identifier = identifier
        self.state = None
        self.matches, self.match_index = [], -1
        self.model.matches, self.model.current = {}, None
        self._output = None
        self._filter_cache = None
        problem = ""
        if table is not None:
            try:
                self.state = engine.TableState(table, identifier)
            except engine.OperationError as error:
                problem = str(error)
        self.view.setVisible(self.state is not None)
        self.empty_label.setVisible(self.state is None)
        self.empty_label.setText(problem or "Sin datos en esta entrada.")
        self.model.reset()
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        if self.state is not None:
            self.view.resizeColumnsToContents()
            for column in range(self.model.columnCount()):
                self.view.setColumnWidth(column, min(self.view.columnWidth(column), 260))
        return problem

    def visible_rows(self):
        if self.state is None:
            return np.arange(0)
        if not self.filter_text:
            return self.state.order
        key = (self.filter_text, self.state.version)
        if self._filter_cache is None or self._filter_cache[0] != key:
            needle = self.filter_text.lower()
            mask = np.zeros(self.state.n_rows, dtype=bool)
            for column in self.state.columns:
                texts = self.state.column_texts(column.cid)
                mask |= np.fromiter((needle in t.lower() for t in texts), dtype=bool, count=len(texts))
            self._filter_cache = (key, mask)
        mask = self._filter_cache[1]
        return self.state.order[mask[self.state.order]]

    def refresh(self, keep_matches=False, layout=False):
        """Vuelve a leer el estado sin perder la celda actual ni la selección.

        Si solo han cambiado datos (no el orden ni las filas visibles), basta con repintar: el
        modelo no se reinicia y la selección se conserva sola, aunque tenga miles de rangos.
        """
        if not keep_matches:
            self.matches, self.match_index = [], -1
            self.model.matches, self.model.current = {}, None
        # Las salidas se guardan por versión del estado: no hace falta tirarlas (y tirarlas haría
        # que se enviaran dos veces tablas idénticas).
        if not layout and not self.filter_text and len(self.model.rows) == (self.state.n_rows if self.state else 0):
            self.model.repaint()
            return
        current, selected = self._remember()
        self.model.reset()
        self._recall(current, selected)

    def _remember(self):
        model = self.view.selectionModel()
        rows = self.model.rows
        index = self.view.currentIndex()
        current = (int(rows[index.row()]), index.column()) if index.isValid() and index.row() < len(rows) else None
        # Por rangos, no por celdas: seleccionar una columna de 300.000 filas es un solo rango.
        selected = [(rows[top:bottom + 1].copy(), left, right) for top, bottom, left, right in self.ranges()]
        return current, selected

    def _recall(self, current, selected):
        from AnyQt.QtCore import QItemSelection
        if self.state is None or not len(self.model.rows):
            return
        position = np.full(self.state.n_rows, -1)
        position[self.model.rows] = np.arange(len(self.model.rows))
        selection = QItemSelection()
        last = self.model.columnCount() - 1
        groups = {}
        for orig_rows, left, right in selected:  # una sola pasada por grupo de columnas
            if left <= last:
                groups.setdefault((left, min(right, last)), []).append(orig_rows)
        for (left, right), parts in groups.items():
            rows = position[np.concatenate(parts)]
            rows = np.unique(rows[rows >= 0])
            if not len(rows):
                continue
            breaks = np.flatnonzero(np.diff(rows) > 1)
            starts, stops = rows[np.r_[0, breaks + 1]], rows[np.r_[breaks, len(rows) - 1]]
            for start, stop in zip(starts.tolist(), stops.tolist()):
                selection.select(self.model.index(start, left), self.model.index(stop, right))
        model = self.view.selectionModel()
        if not selection.isEmpty():
            model.select(selection, QItemSelectionModel.ClearAndSelect)
        if current is not None and position[current[0]] >= 0 and current[1] < self.model.columnCount():
            model.setCurrentIndex(self.model.index(int(position[current[0]]), current[1]), QItemSelectionModel.NoUpdate)

    # ------------------------------------------------------------ orden ----

    def column_order(self):
        if self.state is None:
            return []
        header = self.view.horizontalHeader()
        return [self.state.columns[header.logicalIndex(v)].cid for v in range(header.count())]

    def apply_column_order(self, cids):
        header = self.view.horizontalHeader()
        index = {c.cid: i for i, c in enumerate(self.state.columns)}
        for visual, cid in enumerate(c for c in cids if c in index):
            header.moveSection(header.visualIndex(index[cid]), visual)

    def _header_clicked(self, logical):
        if self.state is None:
            return
        cid = self.state.columns[logical].cid
        descending = self._last_sort == (cid, False)
        self.sort(cid, descending)

    def sort(self, cid, descending):
        self.state.sort(cid, descending)
        self._last_sort = (cid, descending)
        self.view.horizontalHeader().setSortIndicator(self.state.index_of(cid),
                                                      Qt.DescendingOrder if descending else Qt.AscendingOrder)
        self._sort_matches()
        self.refresh(keep_matches=True, layout=True)
        self.widget.changed(self)

    def restore_order(self):
        self.state.restore_order()
        self._last_sort = (None, None)
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self._sort_matches()
        self.refresh(keep_matches=True, layout=True)
        self.widget.changed(self)

    def _sort_matches(self):
        """Las coincidencias siguen resaltadas tras ordenar; se recorren en el orden nuevo."""
        if not self.matches:
            return
        position = np.empty(self.state.n_rows, dtype=int)
        position[self.state.order] = np.arange(self.state.n_rows)
        index = {c.cid: i for i, c in enumerate(self.state.columns)}
        current = self.matches[self.match_index] if 0 <= self.match_index < len(self.matches) else None
        self.matches.sort(key=lambda hit: (position[hit[0]], index[hit[1]]))
        self.match_index = self.matches.index(current) if current in self.matches else -1

    # -------------------------------------------------------- selección ----

    def ranges(self):
        """Rangos de la selección, calculados una vez por cada cambio de selección."""
        model = self.view.selectionModel()
        if model is None:
            return []
        if self._ranges is None:
            self._ranges = [(r.top(), r.bottom(), r.left(), r.right()) for r in model.selection() if r.isValid()]
        return self._ranges

    def _selection_moved(self, *_):
        self._ranges = None

    def selected_view_rows(self):
        ranges = self.ranges()
        if not ranges:
            return np.arange(0, dtype=int)
        return np.unique(np.concatenate([np.arange(top, bottom + 1) for top, bottom, _, _ in ranges]))

    def selection(self):
        """Filas originales (en orden de vista) y cids con alguna celda seleccionada."""
        if self.state is None:
            return np.arange(0), []
        ranges = self.ranges()
        if not ranges:
            return np.arange(0), []
        rows = self.selected_view_rows()
        columns = sorted({c for _, _, left, right in ranges for c in range(left, right + 1)})
        cids = [self.state.columns[c].cid for c in columns]
        order = {cid: k for k, cid in enumerate(self.column_order())}
        cids.sort(key=lambda cid: order.get(cid, 0))
        return self.model.rows[rows], cids

    def full_columns_selected(self):
        """cids de las columnas seleccionadas enteras, en el orden de la vista."""
        model = self.view.selectionModel()
        if model is None or self.state is None or not self.model.rowCount():
            return []
        # A partir de los rangos: isColumnSelected de Qt recorre todas las filas y es muy lento.
        header = self.view.horizontalHeader()
        n_rows = self.model.rowCount()
        spans = {}
        for top, bottom, left, right in self.ranges():
            for column in range(left, right + 1):
                spans.setdefault(column, []).append((top, bottom))
        columns = []
        for column, intervals in spans.items():
            reached = -1
            for top, bottom in sorted(intervals):
                if top > reached + 1:
                    break
                reached = max(reached, bottom)
            if reached >= n_rows - 1:
                columns.append(column)
        return [self.state.columns[c].cid for c in sorted(columns, key=header.visualIndex)]

    def selected_columns(self, full_rows):
        """Columnas sobre las que actúan el ámbito «columnas seleccionadas», el tipo y el rol.

        Las columnas enteras seleccionadas; si no hay, con filas completas la columna de la celda
        actual (una fila seleccionada no selecciona sus columnas) y, si no, las de las celdas.
        """
        full = self.full_columns_selected()
        if full:
            return full
        if full_rows:
            index = self.view.currentIndex()
            return [self.state.columns[index.column()].cid] if index.isValid() and self.state else []
        return self.selection()[1]

    def select_column(self, logical):
        model = self.view.selectionModel()
        if self.model.rowCount():
            top, bottom = self.model.index(0, logical), self.model.index(self.model.rowCount() - 1, logical)
            from AnyQt.QtCore import QItemSelection
            model.select(QItemSelection(top, bottom), QItemSelectionModel.ClearAndSelect)
            model.setCurrentIndex(top, QItemSelectionModel.NoUpdate)

    # ---------------------------------------------------------- colores ----

    def user_color(self, row, cid, which):
        colors = self.colors
        for value in (colors["cells"].get(f"{row}|{cid}"), colors["rows"].get(str(row)), colors["cols"].get(cid)):
            if value and value[which]:
                return QColor(value[which])
        return None

    # -------------------------------------------------------- salidas ----

    def output(self):
        if self.state is None:
            return None
        key = (self.state.version, tuple(self.column_order()), self.identifier)
        if self._output is None or self._output[0] != key:
            self._output = (key, self.state.output(self.column_order(), self.identifier))
        return self._output[1]

    def selected_output(self, full_rows):
        table = self.output()
        if table is None:
            return None
        key = (id(table), tuple(self.ranges()), bool(full_rows), len(self.model.rows),
               self.filter_text, self.state.version)
        if self._selected is not None and self._selected[0] == key:
            return self._selected[1]
        result = self._selected_output(table, full_rows)
        self._selected = (key, result)
        return result

    def _selected_output(self, table, full_rows):
        rows, cids = self.selection()
        if not len(rows):
            return None
        position = np.empty(self.state.n_rows, dtype=int)
        position[self.state.order] = np.arange(self.state.n_rows)
        subset = table[position[rows]]
        if full_rows:
            return subset
        names = {self.state.column(cid).name for cid in cids}
        domain = subset.domain
        reduced = Domain([v for v in domain.attributes if v.name in names],
                         [v for v in domain.class_vars if v.name in names],
                         [v for v in domain.metas if v.name in names])
        out = subset.transform(reduced)
        out.name = table.name
        return out

    # -------------------------------------------------------- guardado ----

    def has_changes(self):
        """¿Tiene estado propio? Datos, orden de filas, orden de columnas o colores."""
        if self.state is None:
            return False
        return bool(self.state.log or self.state.sort_keys or any(self.colors.values())
                    or self.column_order() != [c.cid for c in self.state.columns])

    def stored(self):
        return dict(signature=self.state.signature, fingerprint=self.state.fingerprint,
                    fingerprint_version=engine.FINGERPRINT_VERSION, log=list(self.state.log),
                    sort=list(self.state.sort_keys), columns=self.column_order(), colors=self.colors)


# -------------------------------------------------------------------- widget ----

class OWDataTablePro(OWWidget):
    name = "Data Table Pro"
    description = ("Muestra y edita varias tablas en pestañas: ordenar, reordenar columnas, editar celdas, "
                   "buscar y reemplazar, prefijos, tipos y roles, con salidas numeradas.")
    icon = "icons/owdatatablepro.svg"
    priority = 40
    keywords = ["data table", "tabla", "editar", "edit", "buscar", "reemplazar", "find", "replace",
                "spreadsheet", "hoja de cálculo", "renombrar", "tipo", "rol"]

    class Inputs:
        data = MultiInput("Data", Table)

    class Outputs:
        selected_data_1 = Output("Selected Data 1", Table, default=True)
        data_1 = Output("Data 1", Table)
        selected_data_2 = Output("Selected Data 2", Table)
        data_2 = Output("Data 2", Table)
        selected_data_3 = Output("Selected Data 3", Table)
        data_3 = Output("Data 3", Table)
        selected_data_4 = Output("Selected Data 4", Table)
        data_4 = Output("Data 4", Table)
        selected_data_5 = Output("Selected Data 5", Table)
        data_5 = Output("Data 5", Table)

    class Error(OWWidget.Error):
        operation = Msg("{}")
        load = Msg("{}")

    class Warning(OWWidget.Warning):
        too_many = Msg("Hay {} tablas de entrada: solo las 5 primeras tienen salida (Data 1-5 y Selected Data 1-5).")
        edit = Msg("{}")
        restore = Msg("{}")
        orphans = Msg("{}")

    class Information(OWWidget.Information):
        result = Msg("{}")
        notes = Msg("{}")

    # Ojo: ningún ajuste puede llamarse como un método de QWidget/OWWidget.
    stored_changes = Setting({}, schema_only=True)
    find_regex = Setting(False)
    find_case = Setting(False)
    find_scope = Setting(SCOPE_ALL)
    affix_scope = Setting(SCOPE_COLUMNS)
    select_full_rows = Setting(True)
    show_bars = Setting(False)
    color_categories = Setting(False)
    highlight_edits = Setting(True)
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        self.pages = []
        self._keys = iter(range(10 ** 9))
        self._load_errors = {}      # clave de la pestaña → mensaje (se acumulan, no se pisan)
        self._restore_notes = {}
        self._sent = {}             # salida → último objeto enviado
        self._filter_page = None
        self._filter_pending = None
        self._save_timer = QTimer(self, singleShot=True, interval=0)
        self._save_timer.timeout.connect(self._save_all)
        self._filter_timer = QTimer(self, singleShot=True, interval=FILTER_DEBOUNCE_MS)
        self._filter_timer.timeout.connect(self._apply_filter)
        # Hijo del widget: si el widget se borra, el temporizador muere con él.
        self._edit_timer = QTimer(self, singleShot=True, interval=6000)
        self._edit_timer.timeout.connect(self.Warning.edit.clear)
        self._build_controls()
        self.tabs = QTabWidget(self, documentMode=True)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.mainArea.layout().addWidget(self.tabs)
        self.placeholder = QLabel("Conecta una o varias tablas a la entrada Data.", self)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.mainArea.layout().addWidget(self.placeholder)
        self._shortcuts()
        self._sync_controls()

    # ------------------------------------------------------- construcción ----

    def _build_controls(self):
        box = gui.vBox(self.controlArea, "Tabla")
        self.info_label = QLabel("Sin datos.", self, wordWrap=True)
        box.layout().addWidget(self.info_label)

        box = gui.vBox(self.controlArea, "Buscar y reemplazar")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.find_edit = QLineEdit(self, placeholderText="Texto o regex")
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.returnPressed.connect(self.find_all)
        self.replace_edit = QLineEdit(self, placeholderText="Con regex: \\1, \\g<nombre>")
        self.replace_edit.setClearButtonEnabled(True)
        grid.addWidget(QLabel("Buscar:"), 0, 0)
        grid.addWidget(self.find_edit, 0, 1, 1, 3)
        grid.addWidget(QLabel("Reemplazar:"), 1, 0)
        grid.addWidget(self.replace_edit, 1, 1, 1, 3)
        self.regex_check = gui.checkBox(None, self, "find_regex", "Expresión regular", addToLayout=False)
        self.case_check = gui.checkBox(None, self, "find_case", "Distinguir mayúsculas", addToLayout=False)
        checks = QVBoxLayout()
        checks.setSpacing(2)
        checks.addWidget(self.regex_check)
        checks.addWidget(self.case_check)
        grid.addLayout(checks, 2, 0, 1, 4)
        grid.addWidget(QLabel("Ámbito:"), 3, 0)
        self.find_scope_combo = QComboBox(self)
        self.find_scope_combo.addItems(SCOPES)
        self.find_scope_combo.setCurrentIndex(self.find_scope)
        self.find_scope_combo.activated[int].connect(lambda i: setattr(self, "find_scope", i))
        grid.addWidget(self.find_scope_combo, 3, 1, 1, 3)
        self.previous_button = QPushButton("◀", self, toolTip="Coincidencia anterior (Mayús+F3)")
        self.next_button = QPushButton("▶", self, toolTip="Coincidencia siguiente (F3)")
        for button in (self.previous_button, self.next_button):
            button.setMaximumWidth(44)
        self.find_button = QPushButton("Buscar", self, toolTip="Resalta todas las coincidencias (Ctrl+F)")
        self.replace_button = QPushButton("Reemplazar todo", self)
        self.previous_button.clicked.connect(lambda: self.step_match(-1))
        self.next_button.clicked.connect(lambda: self.step_match(1))
        self.find_button.clicked.connect(self.find_all)
        self.replace_button.clicked.connect(self.replace_all)
        row = QHBoxLayout()
        for button in (self.find_button, self.previous_button, self.next_button):
            row.addWidget(button)
        grid.addLayout(row, 4, 0, 1, 4)
        grid.addWidget(self.replace_button, 5, 0, 1, 4)
        self.find_status = QLabel("", self, wordWrap=True)
        grid.addWidget(self.find_status, 6, 0, 1, 4)
        grid.setColumnStretch(1, 1)

        box = gui.vBox(self.controlArea, "Prefijo y sufijo")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.prefix_edit = QLineEdit(self)
        self.suffix_edit = QLineEdit(self)
        grid.addWidget(QLabel("Prefijo:"), 0, 0)
        grid.addWidget(self.prefix_edit, 0, 1)
        grid.addWidget(QLabel("Sufijo:"), 1, 0)
        grid.addWidget(self.suffix_edit, 1, 1)
        grid.addWidget(QLabel("Ámbito:"), 2, 0)
        self.affix_scope_combo = QComboBox(self)
        self.affix_scope_combo.addItems(SCOPES)
        self.affix_scope_combo.setCurrentIndex(self.affix_scope)
        self.affix_scope_combo.activated[int].connect(lambda i: setattr(self, "affix_scope", i))
        grid.addWidget(self.affix_scope_combo, 2, 1)
        self.affix_button = QPushButton("Añadir", self, toolTip="Solo cambia columnas de texto y categóricas: "
                                                                "las numéricas y de fecha no se tocan.")
        self.affix_button.clicked.connect(self.apply_affix)
        grid.addWidget(self.affix_button, 3, 0, 1, 2)

        box = gui.vBox(self.controlArea, "Columna")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.column_label = QLabel("Selecciona una celda o una columna.", self, wordWrap=True)
        grid.addWidget(self.column_label, 0, 0, 1, 3)
        self.name_edit = QLineEdit(self)
        self.name_edit.returnPressed.connect(self.rename_column)
        self.rename_button = QPushButton("Aplicar", self, toolTip="Renombrar la columna (o pulsa Intro)")
        self.rename_button.clicked.connect(self.rename_column)
        grid.addWidget(QLabel("Nombre:"), 1, 0)
        grid.addWidget(self.name_edit, 1, 1)
        grid.addWidget(self.rename_button, 1, 2)
        self.kind_combo = QComboBox(self)
        for kind in engine.KINDS:
            self.kind_combo.addItem(engine.KIND_NAMES[kind], kind)
        self.kind_combo.activated[int].connect(self.change_kind)
        self.role_combo = QComboBox(self)
        for role in engine.ROLES:
            self.role_combo.addItem(engine.ROLE_NAMES[role], role)
        self.role_combo.activated[int].connect(self.change_role)
        grid.addWidget(QLabel("Tipo:"), 2, 0)
        grid.addWidget(self.kind_combo, 2, 1, 1, 2)
        grid.addWidget(QLabel("Rol:"), 3, 0)
        grid.addWidget(self.role_combo, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)

        box = gui.vBox(self.controlArea, "Vista")
        self.filter_edit = QLineEdit(self, placeholderText="Filtrar filas…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip("Muestra solo las filas que contienen este texto en alguna columna. "
                                    "No cambia las salidas Data, pero mientras está activo, buscar, "
                                    "reemplazar y el prefijo actúan solo sobre las filas visibles.")
        self.filter_edit.textChanged.connect(self._filter_typed)
        box.layout().addWidget(self.filter_edit)
        gui.checkBox(box, self, "select_full_rows", "Seleccionar filas completas", callback=self._selection_mode)
        gui.checkBox(box, self, "show_bars", "Barras en las columnas numéricas", callback=self._repaint)
        gui.checkBox(box, self, "color_categories", "Colores de las categóricas", callback=self._repaint)
        gui.checkBox(box, self, "highlight_edits", "Resaltar celdas editadas", callback=self._repaint)

        box = gui.vBox(self.controlArea, "Cambios")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)
        self.undo_button = QPushButton("Deshacer", self)
        self.redo_button = QPushButton("Rehacer", self)
        self.order_button = QPushButton("Orden original", self, toolTip="Vuelve al orden de filas de la entrada")
        self.reset_button = QPushButton("Restablecer", self, toolTip="Descarta todos los cambios de esta pestaña")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.order_button.clicked.connect(self.restore_order)
        self.reset_button.clicked.connect(self.reset_page)
        grid.addWidget(self.undo_button, 0, 0)
        grid.addWidget(self.redo_button, 0, 1)
        grid.addWidget(self.order_button, 1, 0)
        grid.addWidget(self.reset_button, 1, 1)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def _shortcuts(self):
        for sequence, slot in ((QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo),
                               (QKeySequence("Ctrl+Y"), self.redo), (QKeySequence.Copy, self.copy_selection),
                               (QKeySequence.Find, self._focus_find), (QKeySequence("F3"), lambda: self.step_match(1)),
                               (QKeySequence("Shift+F3"), lambda: self.step_match(-1))):
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    def sizeHint(self):
        return QSize(1150, 680)

    # ----------------------------------------------------------- entradas ----

    @Inputs.data
    def set_data(self, index, table):
        self.pages[index].set_table(None)
        self._load(self.pages[index], table)

    @Inputs.data.insert
    def insert_data(self, index, table):
        page = Page(self, next(self._keys))
        page.view.selectionModel().selectionChanged.connect(lambda *_: self._selection_changed(page))
        page.view.selectionModel().currentChanged.connect(lambda *_: self._sync_controls())
        self.pages.insert(index, page)
        self.tabs.insertTab(index, page, "")
        self._load(page, table)

    @Inputs.data.remove
    def remove_data(self, index):
        for other in self.pages:
            if getattr(other, "pending_restore", False):
                other.pending_restore = False  # quitar un enlace nunca traslada cambios a otra pestaña
                self._restore_notes.pop(other.key, None)  # ni deja colgado el aviso de ambigüedad
        page = self.pages.pop(index)
        self._load_errors.pop(page.key, None)
        self._restore_notes.pop(page.key, None)
        self.tabs.removeTab(self.tabs.indexOf(page))
        page.deleteLater()
        self._relabel()
        self.commit.deferred()

    def _load(self, page, table):
        self._relabel()
        identifier = self._identifier(self.pages.index(page), table)
        problem = page.set_table(table, identifier)
        self._load_errors.pop(page.key, None)
        self._restore_notes.pop(page.key, None)
        if problem:
            self._load_errors[page.key] = f"«{identifier}»: {problem}"
        if page.state is not None:
            self._restore(page, identifier)
        self._show_messages()
        self._relabel()
        self._sync_controls()
        self.commit.deferred()

    def _restore(self, page, identifier):
        """Repite los cambios guardados con el nombre exacto de esta pestaña.

        Si no hay, la pestaña queda pendiente solo durante la ronda de entradas en que se carga:
        al final de esa ronda (`_restore_by_content`) se buscan por contenido, cuando ya están
        las demás pestañas. Una pestaña que ya estaba cargada nunca hereda cambios ajenos.
        """
        stored = dict(self.stored_changes or {})
        saved = stored.get(identifier)
        page.pending_restore = saved is None and bool(stored)
        if saved is not None:
            self._apply_saved(page, identifier, identifier, saved, stored)

    def _restore_by_content(self):
        """Nodo renombrado (o «X (2)» que pasa a «X»): cambios sin dueño con la misma huella.

        Solo para las pestañas recién cargadas y sin estado propio. Si el caso es ambiguo (varios
        candidatos), la pestaña sigue pendiente hasta la siguiente ronda de entradas, por si llega
        la que falta; en cuanto deja de serlo, o el usuario cambia algo, deja de estar pendiente.
        """
        stored = dict(self.stored_changes or {})
        names = {p.identifier for p in self.pages}
        for page in self.pages:
            if not getattr(page, "pending_restore", False):
                continue
            if page.state is None or page.has_changes():
                page.pending_restore = False  # ya tiene estado propio: no se pisa
                continue
            candidates = [key for key, entry in stored.items()
                          if key not in names and entry.get("signature") == page.state.signature
                          and _same_fingerprint(entry, page.state)]
            if len(candidates) > 1:
                self._restore_notes[page.key] = (
                    f"Hay varios cambios guardados ({', '.join(f'«{k}»' for k in candidates[:3])}) que podrían ser "
                    f"de «{page.identifier}», con la misma tabla: no se aplica ninguno para no equivocarse.")
                continue  # sigue pendiente: quizá llega la otra entrada en la siguiente ronda
            page.pending_restore = False
            if candidates:
                self._apply_saved(page, page.identifier, candidates[0], stored[candidates[0]], stored)
                stored = dict(self.stored_changes or {})

    def _apply_saved(self, page, identifier, key, saved, stored):
        state = page.state
        if saved.get("signature") != state.signature:
            self._restore_notes[page.key] = (f"La tabla «{identifier}» ha cambiado (columnas o número de filas): se "
                                             "descartan los cambios que se guardaron con el workflow.")
            stored.pop(key, None)
        elif _comparable(saved) and not _same_fingerprint(saved, state):
            self._restore_notes[page.key] = (f"Los datos de «{identifier}» han cambiado (misma forma, otros valores): "
                                             "se descartan los cambios guardados para no aplicarlos a filas "
                                             "equivocadas.")
            stored.pop(key, None)
        else:
            problems = state.replay(saved.get("log", []), saved.get("sort", []))
            page.colors = saved.get("colors") or page.colors
            page.refresh(layout=True)
            page.apply_column_order(saved.get("columns", []))
            page.saved_key = identifier
            if key != identifier:
                stored[identifier] = stored.pop(key)
                self._restore_notes[page.key] = (f"Los cambios guardados como «{key}» se han aplicado a «{identifier}» "
                                                 "(misma tabla, otro nombre).")
            if problems:
                self._restore_notes[page.key] = (f"«{identifier}»: no se han podido repetir {len(problems)} cambios "
                                                 f"guardados ({problems[0]}).")
            self._relabel()
        self.stored_changes = stored

    def _show_messages(self):
        """Avisos acumulados por pestaña: una entrada nueva no borra los de las demás."""
        self.Error.load("\n".join(self._load_errors.values()), shown=bool(self._load_errors))
        self.Warning.restore("\n".join(self._restore_notes.values()), shown=bool(self._restore_notes))
        names = {p.identifier for p in self.pages}
        orphans = [key for key in (self.stored_changes or {}) if key not in names]
        self.Warning.orphans(
            "Hay cambios guardados que no corresponden a ninguna entrada conectada: "
            + ", ".join(f"«{k}»" for k in orphans[:5]) + ("…" if len(orphans) > 5 else "")
            + ". Se conservan por si vuelve a conectarse esa tabla.", shown=bool(orphans) and bool(self.pages))

    def _source_titles(self):
        """Títulos de los nodos conectados a Data, en el orden de las entradas (si se pueden obtener)."""
        signals = self.__dict__.get("_OWBaseWidget__process_signal_input", [])
        titles = []
        for signal in signals:
            try:
                if signal.channel.name == "Data":
                    titles.append(signal.link.source_node.title)
            except AttributeError:
                titles.append(None)
        return titles

    def _identifier(self, index, table):
        titles = self._source_titles()
        title = titles[index] if index < len(titles) and titles[index] else None
        name = title or (getattr(table, "name", "") if table is not None else "") or f"Tabla {index + 1}"
        used = [p.identifier for i, p in enumerate(self.pages) if i != index and p.identifier]
        unique, count = name, 2
        while unique in used:
            unique, count = f"{name} ({count})", count + 1
        return unique

    def handleNewSignals(self):
        self._restore_by_content()
        self._show_messages()
        self.commit.now()

    # ----------------------------------------------------------- pestañas ----

    def _relabel(self):
        titles = self._source_titles()
        for index, page in enumerate(self.pages):
            if page.table is not None and index < len(titles) and titles[index] and page.state is not None \
                    and not page.identifier.startswith(titles[index]):
                page.identifier = self._identifier(index, page.table)
                page.state.name = page.identifier
            name = page.identifier or "Sin datos"
            modified = " ✎" if page.state is not None and page.state.is_modified() else ""
            badge = BADGES[index] if index < MAX_OUTPUTS else "—"
            self.tabs.setTabText(self.tabs.indexOf(page), f"{badge} {name}{modified}")
            self.tabs.setTabToolTip(self.tabs.indexOf(page),
                                    f"Salidas: Data {index + 1} y Selected Data {index + 1}" if index < MAX_OUTPUTS
                                    else "Sin salida: solo las 5 primeras tablas tienen salida.")
        self.Warning.too_many(len(self.pages), shown=len(self.pages) > MAX_OUTPUTS)
        self.tabs.setVisible(bool(self.pages))
        self.placeholder.setVisible(not self.pages)

    def current_page(self) -> Optional[Page]:
        page = self.tabs.currentWidget()
        return page if isinstance(page, Page) else None

    def _tab_changed(self, _):
        self.find_status.setText("")
        if self._filter_timer.isActive():  # lo escrito en el filtro no se pierde al cambiar de pestaña
            self._filter_pending = self.filter_edit.text().strip()
            self._apply_filter()
        page = self.current_page()
        self.filter_edit.blockSignals(True)
        self.filter_edit.setText(page.filter_text if page is not None else "")
        self.filter_edit.blockSignals(False)
        self._scope_labels()
        self._sync_controls()

    # --------------------------------------------------------- controles ----

    def _sync_controls(self):
        page = self.current_page()
        state = page.state if page is not None else None
        enabled = state is not None
        for widget in (self.find_button, self.previous_button, self.next_button, self.replace_button,
                       self.affix_button, self.order_button, self.reset_button, self.filter_edit):
            widget.setEnabled(enabled)
        self.undo_button.setEnabled(enabled and state.can_undo())
        self.redo_button.setEnabled(enabled and state.can_redo())
        self.undo_button.setToolTip(f"Deshacer: {state.undo_text()} (Ctrl+Z)" if enabled and state.can_undo() else "")
        self.redo_button.setToolTip(f"Rehacer: {state.redo_text()} (Ctrl+Y)" if enabled and state.can_redo() else "")
        if not enabled:
            self.info_label.setText("Sin datos." if not self.pages else "Esta entrada no tiene datos.")
            for widget in (self.name_edit, self.rename_button, self.kind_combo, self.role_combo):
                widget.setEnabled(False)
            return
        index = self.pages.index(page)
        outputs = (f"Salidas: Data {index + 1} y Selected Data {index + 1}." if index < MAX_OUTPUTS
                   else "Sin salida (solo las 5 primeras tablas).")
        shown = len(page.model.rows)
        filtered = f" ({shown} visibles con el filtro)" if shown != state.n_rows else ""
        notes = " ".join(state.notes)
        self.info_label.setText(f"<b>{page.identifier}</b><br>{state.n_rows} filas{filtered}, "
                                f"{len(state.columns)} columnas. {outputs}"
                                + (f"<br>{len(state.log)} cambios." if state.log else "")
                                + (f"<br><i>{notes}</i>" if notes else ""))
        column = self._current_column(page)
        for widget in (self.name_edit, self.rename_button, self.kind_combo, self.role_combo):
            widget.setEnabled(column is not None)
        if column is None:
            self.column_label.setText("Selecciona una celda o una columna.")
            return
        cids = self._target_columns(page)
        extra = f" (y {len(cids) - 1} más para tipo y rol)" if len(cids) > 1 else ""
        self.column_label.setText(f"<b>{column.name}</b>{extra}")
        if not self.name_edit.hasFocus():
            self.name_edit.setText(column.name)
        self.kind_combo.setCurrentIndex(engine.KINDS.index(column.kind))
        self.role_combo.setCurrentIndex(engine.ROLES.index(column.role))

    def _current_column(self, page):
        index = page.view.currentIndex()
        if index.isValid():
            return page.state.columns[index.column()]
        cids = page.selected_columns(self.select_full_rows)
        return page.state.column(cids[0]) if cids else None

    def _target_columns(self, page):
        cids = page.selected_columns(self.select_full_rows)
        current = self._current_column(page)
        if not cids and current is not None:
            cids = [current.cid]
        return cids

    def _selection_mode(self):
        for page in self.pages:
            page.view.setSelectionBehavior(QTableView.SelectRows if self.select_full_rows else QTableView.SelectItems)
        self.commit.deferred()

    def _repaint(self):
        for page in self.pages:
            page.view.viewport().update()

    def _selection_changed(self, page):
        self._sync_controls()
        if self.pages.index(page) < MAX_OUTPUTS:
            self.commit.deferred()

    def _filter_typed(self, _):
        self._filter_page = self.current_page()  # lo escrito es de esta pestaña aunque se cambie pronto
        self._filter_timer.start()

    def _apply_filter(self):
        self._filter_timer.stop()
        page = getattr(self, "_filter_page", None) or self.current_page()
        self._filter_page = None
        if page is None or page not in self.pages:
            return
        page.filter_text = self._filter_pending if self._filter_pending is not None else self.filter_edit.text().strip()
        self._filter_pending = None
        page.refresh(keep_matches=True, layout=True)
        self._scope_labels()
        self._sync_controls()
        self.commit.deferred()  # la selección visible puede cambiar

    def _focus_find(self):
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    # ----------------------------------------------------- operaciones ----

    def _scope(self, page, scope):
        """Filas y columnas del ámbito. Con el filtro activo, solo las filas visibles."""
        visible = page.model.rows if page.filter_text else None
        if scope == SCOPE_ALL:
            return visible, None
        if scope == SCOPE_COLUMNS:
            cids = page.selected_columns(self.select_full_rows)
            if not cids:
                raise engine.OperationError("No hay columnas seleccionadas: selecciona alguna celda de las "
                                            "columnas que quieres cambiar.")
            return visible, cids
        rows, _ = page.selection()
        if not len(rows):
            raise engine.OperationError("No hay filas seleccionadas.")
        return rows, None

    def _scope_labels(self):
        """Con el filtro activo, los ámbitos dicen que solo actúan sobre las filas visibles."""
        page = self.current_page()
        filtered = page is not None and bool(page.filter_text)
        labels = (("Filas visibles (filtro)", "Columnas seleccionadas, filas visibles", SCOPES[2])
                  if filtered else SCOPES)
        for combo in (self.find_scope_combo, self.affix_scope_combo):
            for index, label in enumerate(labels):
                combo.setItemText(index, label)

    def _run(self, action):
        """Ejecuta una operación sobre la pestaña actual y muestra su error, si lo hay."""
        page = self.current_page()
        if page is None or page.state is None:
            return None
        self.Error.operation.clear()
        try:
            return action(page)
        except engine.OperationError as error:
            self.Error.operation(str(error))
            return None

    def find_all(self):
        def action(page):
            rows, cids = self._scope(page, self.find_scope)
            page.matches = page.state.find(self.find_edit.text(), self.find_regex, self.find_case, rows, cids)
            masks = {}
            for row, cid in page.matches:
                masks.setdefault(cid, np.zeros(page.state.n_rows, dtype=bool))[row] = True
            page.model.matches = masks
            page.match_index = -1
            page.model.current = None
            page.view.viewport().update()
            count = len(page.matches)
            self.find_status.setText(f"{count} coincidencia{'s' if count != 1 else ''}." if count
                                     else "Ninguna coincidencia.")
            if count:
                self.step_match(1)
            return count
        return self._run(action)

    def step_match(self, direction):
        page = self.current_page()
        if page is None or page.state is None:
            return
        if not page.matches:
            if self.find_edit.text():
                self.find_all()
            return
        page.match_index = (page.match_index + direction) % len(page.matches)
        row, cid = page.matches[page.match_index]
        page.model.current = (row, cid)
        position = np.flatnonzero(page.model.rows == row)
        if len(position):
            index = page.model.index(int(position[0]), page.state.index_of(cid))
            page.view.scrollTo(index, QAbstractItemView.PositionAtCenter)
            page.view.selectionModel().setCurrentIndex(index, QItemSelectionModel.NoUpdate)
        page.view.viewport().update()
        self.find_status.setText(f"Coincidencia {page.match_index + 1} de {len(page.matches)}"
                                 + ("" if len(position) else " (oculta por el filtro)") + ".")

    def replace_all(self):
        def action(page):
            rows, cids = self._scope(page, self.find_scope)
            total, cells, rejected = page.state.replace(self.find_edit.text(), self.replace_edit.text(),
                                                        self.find_regex, self.find_case, rows, cids)
            text = (f"{total} reemplazo{'s' if total != 1 else ''} en {cells} celda{'s' if cells != 1 else ''}."
                    if cells else "No se ha reemplazado nada.")
            if rejected:
                text += " Rechazados por no ser válidos para su tipo: " + ", ".join(
                    f"{name}: {count}" for name, count in rejected.items()) + "."
            rounded = page.state.last_rounded
            if rounded:
                text += (" Atención: " + ", ".join(f"{name}: {count}" for name, count in rounded.items())
                         + " celdas se veían redondeadas y se han reescrito a partir del texto visible: pierden "
                           "los decimales que no se veían (Ctrl+Z lo deshace).")
            self.find_status.setText(text)
            self.Information.result(text)
            if cells:
                page.refresh()
                self.changed(page)
            return total, cells, rejected
        return self._run(action)

    def apply_affix(self):
        def action(page):
            rows, cids = self._scope(page, self.affix_scope)
            cells, skipped = page.state.affix(self.prefix_edit.text(), self.suffix_edit.text(), rows, cids)
            text = f"Prefijo y sufijo añadidos en {cells} celdas."
            if skipped:
                text += (" No se han tocado las columnas numéricas o de fecha (" + ", ".join(skipped[:5])
                         + ("…" if len(skipped) > 5 else "") + "): conviértelas antes a texto si quieres cambiarlas.")
            self.Information.result(text)
            if cells:
                page.refresh()
                self.changed(page)
            return cells, skipped
        return self._run(action)

    def rename_column(self):
        def action(page):
            column = self._current_column(page)
            if column is not None and page.state.rename(column.cid, self.name_edit.text()):
                page.model.headerDataChanged.emit(Qt.Horizontal, 0, page.model.columnCount() - 1)
                self.changed(page)
        return self._run(action)

    def change_kind(self, index):
        kind = self.kind_combo.itemData(index)

        def action(page):
            # Atómico en el motor: si una columna no puede convertirse, no cambia ninguna.
            lost = page.state.convert(self._target_columns(page), kind)
            page.refresh()
            self.changed(page)
            if lost:
                self.Information.result("Valores que no se han podido convertir y pasan a desconocidos: "
                                        + ", ".join(f"{name}: {count}" for name, count in lost.items()) + ".")
            else:
                self.Information.result.clear()
            return lost
        result = self._run(action)
        self._sync_controls()
        return result

    def change_role(self, index):
        role = self.role_combo.itemData(index)

        def action(page):
            page.state.set_role(self._target_columns(page), role)  # atómico
            page.refresh()
            self.changed(page)
        self._run(action)
        self._sync_controls()

    def undo(self):
        page = self.current_page()
        if page is not None and page.state is not None and page.state.can_undo():
            self.Information.result(f"Deshecho: {page.state.undo()}.")
            page.refresh()
            self.changed(page)

    def redo(self):
        page = self.current_page()
        if page is not None and page.state is not None and page.state.can_redo():
            self.Information.result(f"Rehecho: {page.state.redo()}.")
            page.refresh()
            self.changed(page)

    def restore_order(self):
        page = self.current_page()
        if page is not None and page.state is not None:
            page.restore_order()

    def reset_page(self):
        page = self.current_page()
        if page is None or page.table is None:
            return
        self.stored_changes.pop(page.identifier, None)
        index = self.pages.index(page)
        page.set_table(page.table, page.identifier)
        page.colors = {"cells": {}, "rows": {}, "cols": {}}
        page.filter_text = ""
        self.filter_edit.clear()
        header = page.view.horizontalHeader()
        for logical in range(header.count()):
            header.moveSection(header.visualIndex(logical), logical)
        self.Information.result(f"Se han descartado los cambios de «{page.identifier}».")
        self.changed(self.pages[index])

    def show_edit_error(self, message):
        self.Warning.edit(message)
        self._edit_timer.start()

    def changed(self, page, data=True):
        """Tras cualquier cambio: rótulos, controles, guardado y salidas."""
        if data:
            self.Warning.edit.clear()
        self._relabel()
        self._sync_controls()
        self._save_timer.start()
        if self.pages.index(page) < MAX_OUTPUTS:
            self.commit.deferred()

    def _save_all(self):
        stored = dict(self.stored_changes or {})
        for page in self.pages:
            if page.state is None:
                continue
            if page.saved_key and page.saved_key != page.identifier:
                stored.pop(page.saved_key, None)  # el nodo se ha renombrado: se guarda con el nombre nuevo
            page.saved_key = page.identifier
            if page.has_changes():
                stored[page.identifier] = page.stored()
            else:
                stored.pop(page.identifier, None)
        self.stored_changes = stored

    # ------------------------------------------------------------ menús ----

    def cell_menu(self, position):
        page = self.current_page()
        if page is None or page.state is None:
            return
        menu = QMenu(self)
        menu.addAction("Copiar", self.copy_selection)
        menu.addSeparator()
        menu.addAction("Color de fondo…", lambda: self.choose_color(0))
        menu.addAction("Color del texto…", lambda: self.choose_color(1))
        menu.addAction("Quitar colores", lambda: self.choose_color(None))
        menu.exec(page.view.viewport().mapToGlobal(position))

    def header_menu(self, page, position):
        if page.state is None:
            return
        logical = page.view.horizontalHeader().logicalIndexAt(position)
        if logical < 0:
            return
        cid = page.state.columns[logical].cid
        menu = QMenu(self)
        menu.addAction("Ordenar ascendente", lambda: page.sort(cid, False))
        menu.addAction("Ordenar descendente", lambda: page.sort(cid, True))
        menu.addAction("Orden original", page.restore_order)
        menu.addSeparator()
        menu.addAction("Seleccionar la columna", lambda: page.select_column(logical))
        menu.exec(page.view.horizontalHeader().mapToGlobal(position))

    def choose_color(self, which, color=None):
        """Colorea la selección: filas completas, columnas completas o celdas sueltas."""
        page = self.current_page()
        if page is None or page.state is None:
            return
        if which is not None and color is None:
            chosen = QColorDialog.getColor(QColor("#fff3b0" if which == 0 else "#c0392b"), self,
                                           "Color de fondo" if which == 0 else "Color del texto")
            if not chosen.isValid():
                return
            color = chosen.name()
        n_rows, n_cols = page.model.rowCount(), page.model.columnCount()
        targets = []
        for top, bottom, left, right in page.ranges():
            if top == 0 and bottom == n_rows - 1 and not (left == 0 and right == n_cols - 1):
                targets += [("cols", page.state.columns[c].cid) for c in range(left, right + 1)]
            elif left == 0 and right == n_cols - 1:
                targets += [("rows", str(int(r))) for r in page.model.rows[top:bottom + 1]]
            else:
                targets += [("cells", f"{int(r)}|{page.state.columns[c].cid}")
                            for r in page.model.rows[top:bottom + 1] for c in range(left, right + 1)]
        for group, key in targets:
            if which is None:
                page.colors[group].pop(key, None)
                continue
            value = list(page.colors[group].get(key) or [None, None])
            value[which] = color
            page.colors[group][key] = value
        page.view.viewport().update()
        self._save_all()

    def copy_selection(self):
        page = self.current_page()
        if page is None or page.state is None:
            return
        ranges = page.ranges()
        if not ranges:
            return
        rows = page.selected_view_rows()
        header = page.view.horizontalHeader()
        columns = sorted({c for _, _, left, right in ranges for c in range(left, right + 1)}, key=header.visualIndex)
        lines = ["\t".join(page.state.columns[c].name for c in columns)]
        texts = [page.state.columns[c].texts(page.model.rows[rows]) for c in columns]
        lines += ["\t".join(str(column[i]) for column in texts) for i in range(len(rows))]
        QApplication.clipboard().setText("\n".join(lines))

    # ----------------------------------------------------------- salidas ----

    @gui.deferred
    def commit(self):
        slots = [(self.Outputs.data_1, self.Outputs.selected_data_1), (self.Outputs.data_2, self.Outputs.selected_data_2),
                 (self.Outputs.data_3, self.Outputs.selected_data_3), (self.Outputs.data_4, self.Outputs.selected_data_4),
                 (self.Outputs.data_5, self.Outputs.selected_data_5)]
        if not self.pages and not self._sent:
            return  # sin entradas todavía: no hay nada que enviar (todas las salidas ya son None)
        notes = []
        for index, (data_out, selected_out) in enumerate(slots):
            page = self.pages[index] if index < len(self.pages) else None
            if page is None or page.state is None:
                data, selected = None, None
            else:
                data, selected = page.output(), page.selected_output(self.select_full_rows)
                notes.extend(f"{page.identifier}: {note}" for note in page.state.notes)
            # Solo lo que ha cambiado: el lienzo no descarta los envíos repetidos y recalcularía todo.
            for output, value in ((data_out, data), (selected_out, selected)):
                if self._sent.get(output.name, False) is not value:
                    self._sent[output.name] = value
                    output.send(value)
        self.Information.notes("\n".join(notes), shown=bool(notes))

    def send_report(self):
        items = []
        for index, page in enumerate(self.pages):
            if page.state is None:
                continue
            output = f"Data {index + 1}" if index < MAX_OUTPUTS else "sin salida"
            items.append((page.identifier, f"{page.state.n_rows} filas, {len(page.state.columns)} columnas, "
                                           f"{len(page.state.log)} cambios ({output})"))
        self.report_items("Tablas", items)

    def onDeleteWidget(self):
        self._save_timer.stop()
        self._filter_timer.stop()
        self._edit_timer.stop()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataTablePro).run(insert_data=[(0, Table("iris")), (1, Table("titanic"))])
