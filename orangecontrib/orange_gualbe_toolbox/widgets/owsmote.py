"""Widget SMOTE: balancea las clases generando instancias sintéticas.

La lógica (elección del algoritmo, plan de muestreo y construcción de la
salida) va en funciones sueltas para poder probarla sin interfaz; el widget solo
valida, dibuja y lanza el cálculo en segundo plano.
"""
import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QPointF, QRectF, QSize, Qt
from AnyQt.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPalette, QPen,
)
from AnyQt.QtWidgets import QFrame, QScrollArea, QSizePolicy, QWidget

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Msg, OWWidget, Output

STRATEGIES = (
    "Igualar todas a la clase mayoritaria",
    "Solo la clase minoritaria",
    "Todas menos la clase minoritaria",
    "Hasta un porcentaje de la mayoritaria",
)
EQUAL, MINORITY, NOT_MINORITY, PERCENT = range(4)


def formatted(number):
    """Entero con separador de millares."""
    return f"{int(number):,}".replace(",", ".")


def plural(number, singular, plural_form):
    return f"{formatted(number)} {singular if number == 1 else plural_form}"


def choose_algorithm(domain):
    """Devuelve (algoritmo, índices de atributos categóricos, explicación)."""
    categorical = [i for i, var in enumerate(domain.attributes)
                   if var.is_discrete]
    n_attrs = len(domain.attributes)
    if not categorical:
        return "SMOTE", categorical, "Todos los atributos son numéricos."
    if len(categorical) == n_attrs:
        return "SMOTEN", categorical, "Todos los atributos son categóricos."
    return "SMOTENC", categorical, (
        "Atributos mixtos: "
        f"{plural(n_attrs - len(categorical), 'numérico', 'numéricos')} y "
        f"{plural(len(categorical), 'categórico', 'categóricos')}.")


def class_column(data):
    """Columna de la clase como vector denso de float (con NaN si falta)."""
    y = data.Y.toarray() if sp.issparse(data.Y) else data.Y
    return np.asarray(y, dtype=float).ravel()


def class_counts(y, n_values):
    """Instancias por cada valor declarado de la clase (ignora los NaN)."""
    known = y[np.isfinite(y)].astype(int)
    return np.bincount(known, minlength=n_values)


def sampling_targets(counts, strategy, percent):
    """Instancias finales que debe tener cada clase según la estrategia.

    Las estrategias equivalen a las de imbalanced-learn («not majority»,
    «minority» y «not minority»), con dos diferencias deliberadas: un empate en
    la minoritaria se resuelve siempre por el orden de los valores de la clase
    (imbalanced-learn depende del orden de las filas), y el porcentaje vale
    también con más de dos clases. «all» no se ofrece porque al sobremuestrear
    equivale a «not majority».
    """
    counts = np.asarray(counts)
    targets = counts.copy()
    present = np.flatnonzero(counts)
    majority = counts.max()
    minority = present[np.argmin(counts[present])]
    if strategy == MINORITY:
        targets[minority] = majority
    elif strategy == NOT_MINORITY:
        targets[present[present != minority]] = majority
    elif strategy == PERCENT:
        goal = (int(majority) * int(percent) + 99) // 100
        targets[present] = np.maximum(counts[present], goal)
    else:
        targets[present] = majority
    return targets


def missing_rows(data):
    """(filas con atributos perdidos o infinitos, filas sin clase)."""
    bad_attrs = np.zeros(len(data), dtype=bool)
    if sp.issparse(data.X):
        coords = data.X.tocoo()
        bad_attrs[coords.row[~np.isfinite(coords.data)]] = True
    elif data.X.size:
        bad_attrs = ~np.isfinite(data.X).all(axis=1)
    bad_class = ~np.isfinite(class_column(data))
    return int(bad_attrs.sum()), int(bad_class.sum())


def _synthetic_metas(data, n_rows):
    """Metas de las filas sintéticas: siempre desconocidas."""
    variables = data.domain.metas
    if not variables:
        return None
    if sp.issparse(data.metas):
        return sp.csr_matrix(np.full((n_rows, len(variables)), np.nan))
    metas = np.empty((n_rows, len(variables)), dtype=data.metas.dtype)
    for column, var in enumerate(variables):
        metas[:, column] = var.Unknown
    return metas


def build_output(data, X_new, y_new):
    """Añade las filas sintéticas al final de la tabla original.

    Las filas originales conservan valores, metas, pesos e identificadores. En
    las sintéticas las metas quedan desconocidas (una fila inventada no tiene
    nombre, identificador ni fecha) y el peso, si la tabla tiene pesos, es 1.
    """
    if sp.issparse(data.X) and not sp.issparse(X_new):
        X_new = sp.csr_matrix(X_new)
    weights = np.ones(len(y_new)) if data.has_weights() else None
    synthetic = Table.from_numpy(
        data.domain, X_new, y_new.astype(float),
        metas=_synthetic_metas(data, len(y_new)), W=weights)
    return Table.concatenate((data, synthetic))


def run_smote(data, targets, k_neighbors, seed, state=None):
    """Genera las instancias que faltan y devuelve la tabla balanceada.

    Se ejecuta en un hilo aparte. imbalanced-learn se importa aquí para que un
    fallo de instalación sea un mensaje del widget y no lo haga desaparecer del
    lienzo, y para no retrasar el arranque de Orange.
    """
    if state is not None:
        state.set_status("Generando instancias sintéticas…")
    from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC

    y = class_column(data).astype(int)
    counts = class_counts(y.astype(float), len(targets))
    wanted = {int(value): int(target) for value, target in enumerate(targets)
              if target > counts[value]}
    algorithm, categorical, _ = choose_algorithm(data.domain)
    options = dict(sampling_strategy=wanted, k_neighbors=k_neighbors,
                   random_state=seed)
    X = data.X
    if algorithm == "SMOTE":
        sampler = SMOTE(**options)
    else:
        # Solo SMOTE aprovecha la dispersión: los otros codifican las categorías.
        X = X.toarray() if sp.issparse(X) else X
        if algorithm == "SMOTENC":
            sampler = SMOTENC(categorical_features=categorical, **options)
        else:
            sampler = SMOTEN(**options)
    X_all, y_all = sampler.fit_resample(X, y)
    # imbalanced-learn devuelve primero las filas originales, en su orden.
    return build_output(data, X_all[len(data):], y_all[len(data):])


def imbalance_ratio(counts):
    """Razón entre la clase más numerosa y la menos numerosa con instancias."""
    present = counts[counts > 0]
    return present.max() / present.min() if len(present) else 1.0


class BalanceChart(QWidget):
    """Barras horizontales por clase, antes y después, con la misma escala."""

    MIN_ROW = 28
    MAX_ROW = 58
    MARGIN = 16
    GAP = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.names = []
        self.colors = []
        self.before = None
        self.after = None
        self.pending = False
        self.message = "Conecta una tabla con una variable objetivo categórica."
        self._rows_top = 0
        self._row_height = 1
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_message(self, text):
        self.before = self.after = None
        self.message = text
        self.setMinimumHeight(0)
        self.setToolTip("")
        self.update()

    def set_counts(self, names, colors, before, after, pending=False):
        self.names = list(names)
        self.colors = [QColor(*(int(c) for c in color)) for color in colors]
        self.before = np.asarray(before)
        self.after = None if after is None else np.asarray(after)
        self.pending = pending
        line = QFontMetricsF(self.font()).height()
        self.setMinimumHeight(int(
            2 * self.MARGIN + 5.6 * line + len(self.names) * self.MIN_ROW))
        self.update()

    def sizeHint(self):
        return self.minimumSize().expandedTo(QSize(620, 380))

    def headline(self):
        before_total = int(self.before.sum())
        text = f"{formatted(before_total)} filas"
        if self.after is None:
            return text + f"  ·  desbalanceo {self._ratio(self.before)}"
        added = int(self.after.sum()) - before_total
        return (f"{text} → {formatted(before_total + added)} filas  ·  "
                f"{formatted(added)} sintéticas  ·  desbalanceo "
                f"{self._ratio(self.before)} → {self._ratio(self.after)}")

    @staticmethod
    def _ratio(counts):
        return f"{imbalance_ratio(counts):.1f}:1".replace(".", ",")

    def _after_label(self, row):
        after = int(self.after[row])
        added = after - int(self.before[row])
        return formatted(after) + (f" (+{formatted(added)})" if added else "")

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = QRectF(self.rect()).adjusted(
            self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        ink = self.palette().color(QPalette.WindowText)
        muted = QColor(ink)
        muted.setAlphaF(0.62)
        if self.before is None:
            painter.setPen(muted)
            painter.drawText(area, int(Qt.AlignCenter | Qt.TextWordWrap),
                             self.message)
            return

        metrics = QFontMetricsF(self.font())
        line = metrics.height()
        headline_font = QFont(self.font())
        headline_font.setBold(True)
        painter.setFont(headline_font)
        painter.setPen(ink)
        bold_metrics = QFontMetricsF(headline_font)
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), line * 1.3),
            int(Qt.AlignLeft | Qt.AlignVCenter),
            bold_metrics.elidedText(self.headline(), Qt.ElideRight, area.width()))
        painter.setFont(self.font())

        columns_top = area.top() + line * 1.9
        rows_top = columns_top + line * 1.5
        count = len(self.names)
        row_height = (area.bottom() - line * 1.6 - rows_top) / max(count, 1)
        row_height = max(self.MIN_ROW, min(self.MAX_ROW, row_height))
        self._rows_top, self._row_height = rows_top, row_height
        legend_top = rows_top + row_height * count + line * 0.5
        bar_height = min(row_height - 10, 26)

        names_width = min(
            max(metrics.horizontalAdvance(name) for name in self.names) + 18,
            area.width() * 0.28)
        column_width = (area.width() - names_width - 2 * self.GAP) / 2
        labels = [formatted(n) for n in self.before]
        if self.after is not None:
            labels += [self._after_label(row) for row in range(count)]
        value_width = max(metrics.horizontalAdvance(text) for text in labels) + 8
        bar_space = max(column_width - value_width, 10)
        scale_max = max(self.before.max(),
                        self.after.max() if self.after is not None else 0, 1)
        before_x = area.left() + names_width + self.GAP
        after_x = before_x + column_width + self.GAP

        painter.setPen(muted)
        title = "Después"
        if self.pending:
            title += " (vista previa: pulsa Aplicar)"
        for x, text in ((before_x, "Antes"), (after_x, title)):
            painter.drawText(
                QRectF(x, columns_top, column_width, line * 1.3),
                int(Qt.AlignLeft | Qt.AlignVCenter), text)

        axis = QColor(ink)
        axis.setAlphaF(0.25)
        painter.setPen(QPen(axis, 1))
        bottom = rows_top + row_height * count
        for x in (before_x, after_x):
            painter.drawLine(QPointF(x, rows_top), QPointF(x, bottom))

        for row, name in enumerate(self.names):
            middle = rows_top + row_height * (row + 0.5)
            top = middle - bar_height / 2
            color = self.colors[row]
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(area.left() + 5, middle), 5, 5)
            painter.setPen(ink)
            painter.drawText(
                QRectF(area.left() + 16, middle - line, names_width - 16, 2 * line),
                int(Qt.AlignLeft | Qt.AlignVCenter),
                metrics.elidedText(name, Qt.ElideRight, names_width - 16))

            original = bar_space * self.before[row] / scale_max
            self._solid_bar(painter, before_x, top, original, bar_height, color, True)
            self._value(painter, before_x + original, middle, line, labels[row], ink)

            if self.after is None:
                continue
            added = self.after[row] - self.before[row]
            total = bar_space * self.after[row] / scale_max
            self._solid_bar(painter, after_x, top, original, bar_height, color, not added)
            if added:
                self._hatched_bar(painter, after_x + original + 2, top,
                                  max(total - original - 2, 3), bar_height, color)
            self._value(painter, after_x + total, middle, line,
                        self._after_label(row), ink)

        if self.after is None:
            painter.setPen(muted)
            painter.drawText(
                QRectF(after_x + 8, rows_top, column_width - 8, row_height * count),
                int(Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap),
                "Sin resultado: revisa el mensaje del widget.")
        self._legend(painter, area.left(), legend_top, line, ink)

    @staticmethod
    def _bar_path(x, y, width, height, round_end):
        path = QPainterPath()
        radius = min(4, width, height / 2) if round_end else 0
        path.moveTo(x, y)
        path.lineTo(x + width - radius, y)
        path.quadTo(x + width, y, x + width, y + radius)
        path.lineTo(x + width, y + height - radius)
        path.quadTo(x + width, y + height, x + width - radius, y + height)
        path.lineTo(x, y + height)
        path.closeSubpath()
        return path

    def _solid_bar(self, painter, x, y, width, height, color, round_end):
        if width <= 0:
            return
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(self._bar_path(x, y, max(width, 1), height, round_end))

    def _hatched_bar(self, painter, x, y, width, height, color):
        path = self._bar_path(x, y, width, height, True)
        pale = QColor(color)
        pale.setAlphaF(0.28)
        painter.setPen(Qt.NoPen)
        painter.setBrush(pale)
        painter.drawPath(path)
        painter.setBrush(QBrush(color, Qt.BDiagPattern))
        painter.drawPath(path)

    @staticmethod
    def _value(painter, x, middle, line, text, ink):
        painter.setPen(ink)
        painter.drawText(QRectF(x + 6, middle - line, 400, 2 * line),
                         int(Qt.AlignLeft | Qt.AlignVCenter), text)

    def _legend(self, painter, x, y, line, ink):
        grey = QColor(ink)
        grey.setAlphaF(0.62)
        metrics = QFontMetricsF(self.font())
        swatch_y = y + (line * 1.3 - 12) / 2
        self._solid_bar(painter, x, swatch_y, 18, 12, grey, True)
        painter.setPen(ink)
        text = "Originales"
        painter.drawText(QRectF(x + 24, y, 200, line * 1.3),
                         int(Qt.AlignLeft | Qt.AlignVCenter), text)
        x += 24 + metrics.horizontalAdvance(text) + 20
        self._hatched_bar(painter, x, swatch_y, 18, 12, grey)
        painter.setPen(ink)
        painter.drawText(QRectF(x + 24, y, 200, line * 1.3),
                         int(Qt.AlignLeft | Qt.AlignVCenter), "Sintéticas")

    def mouseMoveEvent(self, event):
        if self.before is None or not len(self.names):
            return
        row = int((event.pos().y() - self._rows_top) // self._row_height)
        if not 0 <= row < len(self.names):
            self.setToolTip("")
            return
        before = int(self.before[row])
        text = f"<b>{self.names[row]}</b><br/>Antes: {formatted(before)}"
        if self.after is not None:
            added = int(self.after[row]) - before
            text += (f"<br/>Después: {formatted(self.after[row])}"
                     f"<br/>Sintéticas: {formatted(added)}")
        self.setToolTip(text)


class OWSmote(OWWidget, ConcurrentWidgetMixin):
    name = "SMOTE"
    description = ("Balancea las clases generando instancias sintéticas de las "
                   "clases minoritarias (SMOTE, SMOTENC o SMOTEN).")
    icon = "icons/owsmote.svg"
    priority = 10
    keywords = ["smote", "balanceo", "desbalanceo", "sobremuestreo",
                "oversampling", "imbalanced", "clases"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        empty = Msg("La tabla de entrada no tiene filas.")
        no_class = Msg("Los datos no tienen variable objetivo. "
                       "Indica cuál es con el widget Select Columns.")
        many_classes = Msg("Hay {} variables objetivo y SMOTE necesita exactamente "
                           "una. Deja solo una con el widget Select Columns.")
        continuous_class = Msg(
            "La variable objetivo «{}» es numérica y SMOTE solo balancea clases "
            "categóricas. Conviértela con Discretize o Edit Domain, o elige "
            "otra con Select Columns.")
        no_attributes = Msg("La tabla no tiene atributos con los que generar "
                            "instancias. Añádelos con el widget Select Columns.")
        missing_attributes = Msg(
            "Hay {} con valores perdidos o infinitos en los atributos y SMOTE no "
            "los admite. Rellénalos antes con el widget Impute o quita esas filas "
            "con Select Rows.")
        missing_class = Msg(
            "Hay {} sin valor en la variable objetivo y SMOTE no las admite. "
            "Quítalas con Select Rows (condición «is defined» sobre la clase).")
        single_class = Msg("Solo hay instancias de la clase «{}»: SMOTE necesita "
                           "al menos dos clases para balancear.")
        too_few = Msg(
            "La clase «{}» tiene {} y con k = {} vecinos SMOTE necesita al menos "
            "{}. Baja k a {} o menos, añade instancias de esa clase o elige "
            "otra estrategia.")
        single_instance = Msg(
            "La clase «{}» solo tiene una instancia y SMOTE necesita al menos dos "
            "para generar otras. Añade instancias de esa clase, quítala con "
            "Select Rows o elige una estrategia que no la sobremuestree.")
        no_library = Msg("No se puede cargar imbalanced-learn ({}). Reinstala el "
                         "add-on desde Options → Add-ons.")
        failed = Msg("No se pudo aplicar SMOTE: {}")

    class Warning(OWWidget.Warning):
        weights_ignored = Msg("La tabla tiene pesos por instancia: SMOTE no los "
                              "usa y las filas sintéticas reciben peso 1.")

    class Information(OWWidget.Information):
        balanced = Msg("Las clases ya están balanceadas: no se generan instancias "
                       "y la salida es la tabla de entrada.")
        nothing_to_add = Msg("Con la estrategia elegida ninguna clase necesita "
                             "instancias nuevas: la salida es la tabla de entrada. "
                             "Prueba otra estrategia si quieres balancear más.")

    # Ojo: ningún Setting puede llamarse como un método de QWidget/OWWidget.
    strategy_index = Setting(EQUAL)
    target_percent = Setting(80)
    k_neighbors = Setting(5)
    random_seed = Setting(42)
    auto_apply = Setting(True)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self._counts = None
        self._targets = None
        self._applied = None

        box = gui.vBox(self.controlArea, "Datos")
        self.data_label = gui.widgetLabel(box, "Sin datos.")
        self.data_label.setWordWrap(True)

        box = gui.vBox(self.controlArea, "Algoritmo (se elige solo)")
        self.algorithm_label = gui.widgetLabel(box, "—")
        self.algorithm_label.setWordWrap(True)

        box = gui.vBox(self.controlArea, "Estrategia de muestreo")
        gui.radioButtons(box, self, "strategy_index", btnLabels=STRATEGIES,
                         callback=self._on_settings_changed)
        self.percent_spin = gui.spin(
            gui.indentedBox(box), self, "target_percent", minv=1, maxv=100, step=5,
            label="Porcentaje:", controlWidth=70, alignment=Qt.AlignRight,
            callback=self._on_settings_changed)
        self.percent_spin.setSuffix(" %")
        self.percent_spin.setKeyboardTracking(False)

        box = gui.vBox(self.controlArea, "Parámetros")
        k_spin = gui.spin(
            box, self, "k_neighbors", minv=1, maxv=100,
            label="Vecinos más cercanos (k):", controlWidth=70,
            alignment=Qt.AlignRight, callback=self._on_settings_changed)
        k_spin.setKeyboardTracking(False)
        seed_spin = gui.spin(
            box, self, "random_seed", minv=0, maxv=999999,
            label="Semilla aleatoria:", controlWidth=70,
            alignment=Qt.AlignRight, callback=self._on_settings_changed)
        seed_spin.setKeyboardTracking(False)

        gui.rubber(self.controlArea)
        gui.auto_apply(self.buttonsArea, self, "auto_apply")

        self.chart = BalanceChart()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.chart)
        self.mainArea.layout().addWidget(scroll)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)
        self._sync_percent_control()

    def _sync_percent_control(self):
        self.percent_spin.box.setEnabled(self.strategy_index == PERCENT)

    def _on_settings_changed(self):
        self._sync_percent_control()
        self._analyse()
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data):
        self.cancel()
        self.data = data
        self._applied = None
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(len(data), format_summary_details(data))
        self._analyse()
        self.commit.now()

    def _clear_messages(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

    def _describe_data(self):
        data = self.data
        attributes = data.domain.attributes
        numeric = sum(var.is_continuous for var in attributes)
        categorical = len(attributes) - numeric
        kinds = []
        if numeric:
            kinds.append(plural(numeric, "numérico", "numéricos"))
        if categorical:
            kinds.append(plural(categorical, "categórico", "categóricos"))
        text = (f"{plural(len(data), 'fila', 'filas')}, "
                f"{plural(len(attributes), 'atributo', 'atributos')}")
        if kinds:
            text += " (" + ", ".join(kinds) + ")"
        class_var = data.domain.class_var
        if class_var is not None and class_var.is_discrete:
            text += "\nObjetivo: " + class_var.name
        self.data_label.setText(text + ".")

    def _analyse(self):
        """Valida la entrada, fija los mensajes y calcula el plan de muestreo."""
        self._clear_messages()
        self._counts = self._targets = None
        self.algorithm_label.setText("—")
        data = self.data
        if data is None:
            self.data_label.setText("Sin datos.")
            self.chart.show_message(
                "Conecta una tabla con una variable objetivo categórica.")
            return
        self._describe_data()
        domain = data.domain
        class_var = domain.class_var

        if not len(data):
            self.Error.empty()
        elif class_var is None and len(domain.class_vars) > 1:
            self.Error.many_classes(len(domain.class_vars))
        elif class_var is None:
            self.Error.no_class()
        elif not class_var.is_discrete:
            self.Error.continuous_class(class_var.name)
        else:
            self._counts = class_counts(class_column(data), len(class_var.values))
            if self._validate_rows():
                self._plan_sampling()
        self._draw_chart()

    def _validate_rows(self):
        """Comprobaciones que dejan ver la distribución inicial en el gráfico."""
        data = self.data
        if not data.domain.attributes:
            self.Error.no_attributes()
            return False
        bad_attrs, bad_class = missing_rows(data)
        if bad_attrs:
            self.Error.missing_attributes(plural(bad_attrs, "fila", "filas"))
        if bad_class:
            self.Error.missing_class(plural(bad_class, "fila", "filas"))
        if bad_attrs or bad_class:
            return False
        present = np.flatnonzero(self._counts)
        if len(present) < 2:
            self.Error.single_class(data.domain.class_var.values[present[0]])
            return False
        return True

    def _plan_sampling(self):
        data = self.data
        counts = self._counts
        class_var = data.domain.class_var
        algorithm, _, why = choose_algorithm(data.domain)
        self.algorithm_label.setText(f"<b>{algorithm}</b><br/>{why}")

        targets = sampling_targets(counts, self.strategy_index, self.target_percent)
        growing = np.flatnonzero(targets > counts)
        if len(growing):
            smallest = growing[np.argmin(counts[growing])]
            size = int(counts[smallest])
            name = class_var.values[smallest]
            if size < 2:
                self.Error.single_instance(name)
                return
            if size <= self.k_neighbors:
                self.Error.too_few(name, plural(size, "instancia", "instancias"),
                                   self.k_neighbors, self.k_neighbors + 1, size - 1)
                return
            if data.has_weights():
                self.Warning.weights_ignored()
        elif len(set(counts[counts > 0])) == 1:
            self.Information.balanced()
        else:
            self.Information.nothing_to_add()
        self._targets = targets

    def _params(self):
        if self._targets is None:
            return None
        return tuple(int(t) for t in self._targets), self.k_neighbors, self.random_seed

    def _draw_chart(self):
        if self._counts is None or not self._counts.sum():
            self.chart.show_message("El gráfico necesita una tabla con filas y "
                                    "una variable objetivo categórica.")
            return
        class_var = self.data.domain.class_var
        pending = (not self.auto_apply and self._targets is not None
                   and self._params() != self._applied)
        self.chart.set_counts(class_var.values, class_var.colors, self._counts,
                              self._targets, pending)

    def _send(self, table):
        if table is None:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            self.info.set_output_summary(len(table), format_summary_details(table))
        self.Outputs.data.send(table)

    @gui.deferred
    def commit(self):
        self.cancel()
        self.Error.failed.clear()
        self.Error.no_library.clear()
        self._applied = self._params()
        if self._targets is None:
            self._send(None)
        elif (self._targets == self._counts).all():
            self._send(self.data)
        else:
            self.start(run_smote, self.data, self._targets,
                       self.k_neighbors, self.random_seed)
        self._draw_chart()

    def on_done(self, result):
        self._send(result)

    def on_exception(self, ex):
        if isinstance(ex, ImportError):
            self.Error.no_library(ex)
        else:
            self.Error.failed(ex)
        self._targets = None
        self._draw_chart()
        self._send(None)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSmote).run(Table("titanic"))
