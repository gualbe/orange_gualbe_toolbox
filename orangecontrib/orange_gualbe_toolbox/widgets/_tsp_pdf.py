"""Maquetación en PDF del informe de Test & Score Pro.

Se maqueta por bloques midiendo alturas y se dibuja con QPainter sobre QPdfWriter, porque
QTextDocument no respeta saltos de página ni mantiene un título con lo que le sigue. Funciona en
un hilo aparte y sin servidor gráfico. A4 apaisado: el informe se compone de tablas de 9 a 13
columnas y de gráficos anchos, que en vertical quedarían estrechos.

`render_pdf` devuelve el registro de lo colocado en cada página (en puntos), con el que los tests
comprueban la maquetación.
"""
from dataclasses import dataclass, field
from typing import Callable, List

from AnyQt.QtCore import QMarginsF, QPointF, QRectF, Qt
from AnyQt.QtGui import (
    QColor, QFont, QFontMetricsF, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen,
)

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots
from orangecontrib.orange_gualbe_toolbox.widgets._tsp_report import (
    Figures, FigureSpec, Heading, Note, PageBreak, Paragraph, Result, Summary, TableBlock, Title,
)

RESOLUTION = 300
PAGE_WIDTH, PAGE_HEIGHT = 842.0, 595.0
MARGIN_LEFT = MARGIN_RIGHT = 48.0
MARGIN_TOP, MARGIN_BOTTOM = 42.0, 50.0
BODY_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
BODY_TOP, BODY_BOTTOM = MARGIN_TOP, PAGE_HEIGHT - MARGIN_BOTTOM
BODY_HEIGHT = BODY_BOTTOM - BODY_TOP
FIGURE_DPI = 200

INK, MUTED, ACCENT, RULE = "#1b1b19", "#52514e", "#2a78d6", "#d9d8d2"
RESULT_BACK, NOTE_BACK, NOTE_BAR = "#e8f1fb", "#fff6e5", "#eda100"
HEAD_BACK, ZEBRA, BEST_BACK = "#f1f0ea", "#faf9f6", "#e8f1fb"

SHORT_TABLE_ROWS = 8
SHORT_TABLE_SHARE = 0.45  # de la altura útil: lo que no llega a esto nunca se parte
MIN_ROWS_ON_PAGE = 3
CELL_PAD_X, CELL_PAD_Y = 5.0, 3.5


@dataclass
class Placed:
    page: int
    kind: str
    label: str
    top: float
    bottom: float
    left: float
    right: float
    fragment: int = 0
    rows: int = 0
    total_rows: int = 0
    repeats_header: bool = False
    keep_next: bool = False


@dataclass
class PdfLayout:
    pages: int = 0
    items: List[Placed] = field(default_factory=list)
    footer: str = ""
    sequence: List[str] = field(default_factory=list)


class Styles:
    def __init__(self, device):
        self.device = device
        self.scale = RESOLUTION / 72.0
        self.h1 = self._font(21, True)
        self.h2 = self._font(14, True)
        self.h3 = self._font(11, True)
        self.h4 = self._font(9.5, True)
        self.body = self._font(9.5)
        self.small = self._font(8.2)
        self.cell = self._font(8.4)
        self.cell_bold = self._font(8.4, True)
        self.head = self._font(8, True)
        self.footer = self._font(7.8)

    @staticmethod
    def _font(size, bold=False):
        font = QFont("Helvetica Neue")
        font.setStyleHint(QFont.SansSerif)
        font.setPointSizeF(size)
        font.setBold(bold)
        return font

    def metrics(self, font):
        return QFontMetricsF(font, self.device)

    def text_height(self, font, text, width):
        rect = self.metrics(font).boundingRect(
            QRectF(0, 0, width * self.scale, 1e7), int(Qt.TextWordWrap | Qt.AlignLeft), text)
        return rect.height() / self.scale

    def line_height(self, font):
        return self.metrics(font).height() / self.scale

    def text_width(self, font, text):
        return self.metrics(font).horizontalAdvance(text) / self.scale

    def longest_word(self, font, text):
        return max((self.text_width(font, word) for word in str(text).split()), default=0.0)


class Canvas:
    """Dibuja en puntos sobre un dispositivo a 300 ppp."""

    def __init__(self, painter, scale):
        self.painter = painter
        self.scale = scale

    def _rect(self, x, y, w, h):
        s = self.scale
        return QRectF(x * s, y * s, w * s, h * s)

    def text(self, font, color, x, y, w, h, text, align=None, wrap=True):
        self.painter.setFont(font)
        self.painter.setPen(QColor(color))
        flags = Qt.AlignLeft | Qt.AlignTop if align is None else align
        if wrap:
            flags |= Qt.TextWordWrap
        self.painter.drawText(self._rect(x, y, w, h), int(flags), text)

    def fill(self, x, y, w, h, color):
        self.painter.fillRect(self._rect(x, y, w, h), QColor(color))

    def line(self, x0, y, x1, color, width=0.6):
        pen = QPen(QColor(color))
        pen.setWidthF(width * self.scale)
        self.painter.setPen(pen)
        self.painter.drawLine(QPointF(x0 * self.scale, y * self.scale), QPointF(x1 * self.scale, y * self.scale))

    def image(self, image, x, y, w, h):
        self.painter.drawImage(self._rect(x, y, w, h), image)


class Flow:
    kind = "block"
    keep_next = False
    keep_previous = False
    before = 0.0
    after = 0.0
    label = ""
    height = 0.0

    def draw(self, canvas, x, y):
        raise NotImplementedError


class TextFlow(Flow):
    def __init__(self, styles, font, text, color=INK, width=BODY_WIDTH, kind="paragraph", before=0.0, after=5.0,
                 keep_next=False, keep_previous=False, indent=0.0, bullet=False):
        self.font, self.text, self.color = font, text, color
        self.kind, self.before, self.after = kind, before, after
        self.keep_next, self.keep_previous = keep_next, keep_previous
        self.indent, self.bullet = indent + (10.0 if bullet else 0.0), bullet
        self.width = width
        self.label = text
        self.height = styles.text_height(font, text, width - self.indent)

    def draw(self, canvas, x, y):
        if self.bullet:
            canvas.text(self.font, MUTED, x, y, 10, self.height, "•")
        canvas.text(self.font, self.color, x + self.indent, y, self.width - self.indent, self.height + 1, self.text)


class HeadingFlow(TextFlow):
    def __init__(self, styles, level, text):
        font = {2: styles.h2, 3: styles.h3, 4: styles.h4}[level]
        color = ACCENT if level == 2 else INK
        before, after = {2: (16.0, 8.0), 3: (12.0, 4.0), 4: (6.0, 3.0)}[level]
        super().__init__(styles, font, text, color, kind="heading", before=before, after=after, keep_next=True)
        self.level = level

    def draw(self, canvas, x, y):
        super().draw(canvas, x, y)
        if self.level == 2:
            canvas.line(x, y + self.height + 2.5, x + BODY_WIDTH, ACCENT, 1.0)


class BoxFlow(Flow):
    """Recuadro con barra lateral: resultado principal o aviso."""

    def __init__(self, styles, font, lines, back, bar, kind, pad_x=12.0, pad_y=9.0, gap=3.0, after=7.0, emphasise_first=False):
        self.styles, self.font, self.lines, self.back, self.bar = styles, font, lines, back, bar
        self.kind, self.pad_x, self.pad_y, self.gap, self.after = kind, pad_x, pad_y, gap, after
        self.emphasise_first = emphasise_first
        self.label = lines[0] if lines else ""
        inner = BODY_WIDTH - 2 * pad_x - 3
        self.heights = [styles.text_height(self._font(i), line, inner) for i, line in enumerate(lines)]
        self.inner = inner
        self.height = sum(self.heights) + gap * max(len(lines) - 1, 0) + 2 * pad_y

    def _font(self, index):
        if index == 0 and self.emphasise_first:
            return self.styles.h3
        return self.font

    def draw(self, canvas, x, y):
        canvas.fill(x, y, BODY_WIDTH, self.height, self.back)
        canvas.fill(x, y, 3, self.height, self.bar)
        cursor = y + self.pad_y
        for index, (line, height) in enumerate(zip(self.lines, self.heights)):
            canvas.text(self._font(index), INK, x + self.pad_x + 3, cursor, self.inner, height + 1, line)
            cursor += height + self.gap


class TableFlow(Flow):
    kind = "table"

    def __init__(self, styles, block, width=BODY_WIDTH):
        self.styles, self.block, self.width = styles, block, width
        self.label = block.label or (block.headers[0] if block.headers else "")
        self.headers = block.headers
        self.rows = block.rows
        self.numeric = set(block.numeric)
        self.before, self.after = 2.0, 7.0
        if block.key_value and len(block.rows) > 4:
            self._fold_key_value()
        self.widths = self._column_widths()
        self.header_height = self._row_height(self.headers, styles.head)
        self.row_heights = [self._row_height(row, styles.cell_bold if i == self.best else styles.cell)
                            for i, row in enumerate(self.rows)]
        self.height = self.header_height + sum(self.row_heights)

    @property
    def best(self):
        return self.block.best

    def _fold_key_value(self):
        """Tablas concepto-valor a dos bandas: la mitad de alto y la misma información."""
        rows = self.rows
        middle = (len(rows) + 1) // 2
        left, right = rows[:middle], rows[middle:] + [["", ""]] * (middle - len(rows[middle:]))
        self.headers = self.headers + self.headers
        self.rows = [a + b for a, b in zip(left, right)]

    def _column_widths(self):
        styles, count = self.styles, len(self.headers)
        natural = [styles.text_width(styles.head, self.headers[c]) + 2 * CELL_PAD_X for c in range(count)]
        minimum = [styles.longest_word(styles.head, self.headers[c]) + 2 * CELL_PAD_X + 2.0 for c in range(count)]
        for row in self.rows:
            for column, cell in enumerate(row):
                natural[column] = max(natural[column], styles.text_width(styles.cell, cell) + 2 * CELL_PAD_X)
                # El margen evita que el redondeo parta un número o una palabra.
                floor = (styles.text_width(styles.cell_bold, cell) if column in self.numeric
                         else styles.longest_word(styles.cell_bold, cell)) + 2 * CELL_PAD_X + 2.0
                minimum[column] = max(minimum[column], floor)
        total = sum(natural)
        if total <= self.width:
            # Una tabla pequeña no se estira hasta el margen: quedaría con huecos enormes.
            target = self.width if self.block.key_value or total > 0.62 * self.width else total * 1.3
            extra = (target - total) / count
            return [value + extra for value in natural]
        spare = [n - m for n, m in zip(natural, minimum)]
        available = self.width - sum(minimum)
        if available <= 0:  # no cabe ni lo indivisible: se reparte en proporción
            return [value * self.width / sum(minimum) for value in minimum]
        factor = available / sum(spare)
        return [m + s * factor for m, s in zip(minimum, spare)]

    def _row_height(self, row, font):
        heights = [self.styles.line_height(font) if self._is_number(c, row) else
                   self.styles.text_height(font, str(cell), self.widths[c] - 2 * CELL_PAD_X)
                   for c, cell in enumerate(row)]
        return max(heights) + 2 * CELL_PAD_Y

    def _is_number(self, column, row):
        return column in self.numeric and row is not self.headers

    @property
    def splittable(self):
        return len(self.rows) > SHORT_TABLE_ROWS and self.height > SHORT_TABLE_SHARE * BODY_HEIGHT

    def minimum_height(self):
        if not self.splittable:
            return self.height
        return self.header_height + sum(self.row_heights[:MIN_ROWS_ON_PAGE])

    def _draw_row(self, canvas, x, y, height, row, font, align_numeric=True):
        cursor = x
        for column, cell in enumerate(row):
            width = self.widths[column]
            numeric = column in self.numeric if not self.block.key_value else False
            align = (Qt.AlignRight | Qt.AlignTop) if numeric else (Qt.AlignLeft | Qt.AlignTop)
            canvas.text(font, INK, cursor + CELL_PAD_X, y + CELL_PAD_Y, width - 2 * CELL_PAD_X, height, str(cell), align,
                        wrap=not self._is_number(column, row))
            cursor += width

    def draw_rows(self, canvas, x, y, start, stop):
        total = sum(self.widths)
        canvas.fill(x, y, total, self.header_height, HEAD_BACK)
        self._draw_row(canvas, x, y, self.header_height, self.headers, self.styles.head)
        canvas.line(x, y + self.header_height, x + total, MUTED, 0.7)
        cursor = y + self.header_height
        for index in range(start, stop):
            height = self.row_heights[index]
            if index == self.best:
                canvas.fill(x, cursor, total, height, BEST_BACK)
            elif (index - start) % 2:
                canvas.fill(x, cursor, total, height, ZEBRA)
            font = self.styles.cell_bold if index == self.best else self.styles.cell
            self._draw_row(canvas, x, cursor, height, self.rows[index], font)
            cursor += height
            canvas.line(x, cursor, x + total, RULE, 0.4)
        return cursor

    def draw(self, canvas, x, y):
        self.draw_rows(canvas, x, y, 0, len(self.rows))


def _spec_image(spec, width_pt, height_pt):
    figure = spec.make(width_pt / 72.0, height_pt / 72.0)
    png = plots.figure_png(figure, FIGURE_DPI)
    image = QImage()
    image.loadFromData(png)
    return image, figure.get_size_inches()


def _fit(size_inches, width, height):
    ratio = min(width / (size_inches[0] * 72.0), height / (size_inches[1] * 72.0), 1.0)
    return size_inches[0] * 72.0 * ratio, size_inches[1] * 72.0 * ratio


class FiguresFlow(Flow):
    """Una fila de gráficos: cada uno con su título encima, nunca separados."""

    kind = "figure"

    def __init__(self, styles, specs, cell_width, cell_height):
        self.styles = styles
        self.title_height = styles.line_height(styles.h4) + 4
        self.cells = []
        for spec in specs:
            image, size = _spec_image(spec, cell_width, cell_height)
            self.cells.append((spec.title, image, _fit(size, cell_width, cell_height)))
        self.cell_width = cell_width
        self.gap = BODY_WIDTH - cell_width * len(specs) if len(specs) > 1 else 0
        self.height = self.title_height + cell_height
        self.after = 10.0
        self.label = " / ".join(title for title, _, _ in self.cells)

    def draw(self, canvas, x, y):
        for index, (title, image, (w, h)) in enumerate(self.cells):
            left = x + index * (self.cell_width + self.gap)
            canvas.text(self.styles.h4, INK, left, y, self.cell_width, self.title_height, title)
            canvas.image(image, left + (self.cell_width - w) / 2, y + self.title_height, w, h)


class SummaryFlow(Flow):
    """Lo principal de un vistazo: tabla a la izquierda y gráfico a la derecha."""

    kind = "summary"
    GAP = 18.0

    def __init__(self, styles, block, available):
        self.styles = styles
        self.facts = block.facts
        self.facts_height = styles.text_height(styles.small, block.facts, BODY_WIDTH)
        self.label = "Resumen"
        self.left_width = BODY_WIDTH * 0.47
        self.right_width = BODY_WIDTH - self.left_width - self.GAP
        room = available - self.facts_height - 10
        caption_height = styles.line_height(styles.small) * 2 + 8
        rows = list(block.table.rows)
        self.table = TableFlow(styles, block.table, self.left_width)
        while len(rows) > 1 and (self.table.height > room or
                                 (len(rows) < block.total_models and self.table.height + caption_height > room)):
            rows.pop()
            shorter = TableBlock(block.table.headers, rows, block.table.numeric, block.table.best, label=block.table.label)
            self.table = TableFlow(styles, shorter, self.left_width)
        self.caption = ""
        if len(rows) < block.total_models:
            self.caption = (f"Se muestran los {len(rows)} mejores de {block.total_models} modelos; "
                            "la tabla completa está en Puntuaciones.")
        self.caption_height = caption_height if self.caption else 0.0
        chart_height = max(room - styles.line_height(styles.h4) - 4, 120.0)
        self.title = block.figure.title
        self.image, size = _spec_image(block.figure, self.right_width, chart_height)
        self.image_size = _fit(size, self.right_width, chart_height)
        self.chart_height = chart_height
        title_height = styles.line_height(styles.h4) + 4
        self.height = self.facts_height + 10 + max(self.table.height + self.caption_height, title_height + chart_height)

    def draw(self, canvas, x, y):
        styles = self.styles
        canvas.text(styles.small, MUTED, x, y, BODY_WIDTH, self.facts_height + 1, self.facts)
        top = y + self.facts_height + 10
        self.table.draw(canvas, x, top)
        if self.caption:
            canvas.text(styles.small, MUTED, x, top + self.table.height + 5, self.left_width, self.caption_height, self.caption)
        right = x + self.left_width + self.GAP
        title_height = styles.line_height(styles.h4) + 4
        canvas.text(styles.h4, INK, right, top, self.right_width, title_height, self.title)
        w, h = self.image_size
        canvas.image(self.image, right + (self.right_width - w) / 2, top + title_height, w, h)


class BreakFlow(Flow):
    kind = "break"


def _flows(document, styles):
    flows = []
    used = 0.0
    for block in document.blocks:
        if isinstance(block, Title):
            flows.append(TextFlow(styles, styles.h1, block.text, kind="title", after=2.0))
            flows.append(TextFlow(styles, styles.small, block.subtitle, MUTED, kind="subtitle", after=10.0))
            used += flows[-2].height + 2 + flows[-1].height + 10
        elif isinstance(block, Result):
            flows.append(BoxFlow(styles, styles.body, block.lines, RESULT_BACK, ACCENT, "result", pad_x=14, pad_y=11,
                                 gap=4, after=12, emphasise_first=True))
            used += flows[-1].height + 12
        elif isinstance(block, Summary):
            flows.append(SummaryFlow(styles, block, BODY_HEIGHT - used))
        elif isinstance(block, PageBreak):
            flows.append(BreakFlow())
        elif isinstance(block, Heading):
            flows.append(HeadingFlow(styles, block.level, block.text))
        elif isinstance(block, Paragraph):
            flows.append(TextFlow(styles, styles.small if block.small else styles.body, block.text,
                                  MUTED if block.small else INK, keep_next=block.keep_next,
                                  keep_previous=block.keep_previous, bullet=block.bullet,
                                  after=3.0 if block.small else 6.0))
        elif isinstance(block, Note):
            flows.append(BoxFlow(styles, styles.small, [block.text], NOTE_BACK, NOTE_BAR, "note", pad_x=9, pad_y=6, after=5))
            # El primer aviso de un test no se separa de la tabla o del pie al que se refiere.
            flows[-1].keep_previous = len(flows) > 1 and (flows[-2].keep_previous or isinstance(flows[-2], TableFlow))
        elif isinstance(block, TableBlock):
            flows.append(TableFlow(styles, block))
        elif isinstance(block, Figures):
            flows.extend(_figure_rows(styles, block))
    return flows


FULL_FIGURE_HEIGHT = 300.0


def _figure_rows(styles, block):
    half = (BODY_WIDTH - 16.0) / 2
    if block.full:
        return [FiguresFlow(styles, [spec], BODY_WIDTH, FULL_FIGURE_HEIGHT) for spec in block.specs]
    if not block.wide:
        return [FiguresFlow(styles, block.specs[i:i + 2], half, 205.0) for i in range(0, len(block.specs), 2)]
    # Con muchos modelos: la matriz a todo el ancho y los demás, de dos en dos a toda altura.
    matrices = [spec for spec in block.specs if spec.title.startswith("Matriz")]
    others = [spec for spec in block.specs if not spec.title.startswith("Matriz")]
    rows = [FiguresFlow(styles, others[i:i + 2], half, 430.0) for i in range(0, len(others), 2)]
    rows[1:1] = [FiguresFlow(styles, [spec], BODY_WIDTH, 430.0) for spec in matrices]
    return rows


def _chain_end(flows, start):
    end = start
    while flows[end].keep_next and end + 1 < len(flows):
        end += 1
    return end


def paginate(flows, footer):
    """Reparte los bloques en páginas. Devuelve (dibujos por página, registro)."""
    layout = PdfLayout(footer=footer)
    pages = [[]]
    page = 0
    y = BODY_TOP

    def new_page():
        nonlocal page, y
        pages.append([])
        page += 1
        y = BODY_TOP

    def register(kind, label, top, bottom, width=BODY_WIDTH, **extra):
        layout.items.append(Placed(page + 1, kind, label, top, bottom, MARGIN_LEFT, MARGIN_LEFT + width, **extra))

    def place_flow(flow):
        nonlocal y
        top = y + flow.before if y > BODY_TOP else y
        pages[page].append((flow.draw, MARGIN_LEFT, top))
        register(flow.kind, flow.label, top, top + flow.height, keep_next=flow.keep_next)
        y = top + flow.height + flow.after

    def place_table(flow, tail):
        nonlocal y
        if not flow.splittable:
            top = y + flow.before if y > BODY_TOP else y
            pages[page].append((flow.draw, MARGIN_LEFT, top))
            register("table", flow.label, top, top + flow.height, sum(flow.widths), fragment=1,
                     rows=len(flow.rows), total_rows=len(flow.rows), repeats_header=False)
            y = top + flow.height + flow.after
            return
        start, fragment = 0, 0
        total = len(flow.rows)
        while start < total:
            fragment += 1
            top = y + flow.before if y > BODY_TOP and start == 0 else y
            room = BODY_BOTTOM - top - flow.header_height
            used, stop = 0.0, start
            while stop < total and used + flow.row_heights[stop] <= room:
                used += flow.row_heights[stop]
                stop += 1
            if stop == total and used + tail > room and total - start > MIN_ROWS_ON_PAGE:
                # Lo último de la tabla va con su pie.
                stop = total - MIN_ROWS_ON_PAGE
            elif stop < total and total - stop < MIN_ROWS_ON_PAGE:
                stop = max(total - MIN_ROWS_ON_PAGE, start + 1)
            if stop - start < min(MIN_ROWS_ON_PAGE, total - start) and top > BODY_TOP:
                new_page()
                continue
            stop = max(stop, start + 1)
            height = flow.header_height + sum(flow.row_heights[start:stop])
            pages[page].append((lambda c, x, yy, a=start, b=stop: flow.draw_rows(c, x, yy, a, b), MARGIN_LEFT, top))
            register("table", flow.label, top, top + height, sum(flow.widths), fragment=fragment, rows=stop - start,
                     total_rows=total, repeats_header=fragment > 1)
            start = stop
            y = top + height + flow.after
            if start < total:
                new_page()

    index = 0
    while index < len(flows):
        flow = flows[index]
        if isinstance(flow, BreakFlow):
            if y > BODY_TOP:
                new_page()
            index += 1
            continue
        last = _chain_end(flows, index)
        stop = last
        while stop + 1 < len(flows) and flows[stop + 1].keep_previous:
            stop += 1
        needed = sum(f.before + f.height + f.after for f in flows[index:last])
        anchor = flows[last]
        tail = sum(f.before + f.height + f.after for f in flows[last + 1:stop + 1])
        if isinstance(anchor, TableFlow) and anchor.splittable:
            needed += anchor.before + anchor.minimum_height()
        else:
            needed += anchor.before + anchor.height + tail
        if y > BODY_TOP and y + needed > BODY_BOTTOM:
            new_page()
        for position in range(index, stop + 1):
            current = flows[position]
            if isinstance(current, TableFlow):
                place_table(current, tail if position == last else 0.0)
            else:
                place_flow(current)
        index = stop + 1
    layout.pages = len(pages)
    layout.sequence = [flow.kind for flow in flows]
    return pages, layout


def _draw_footer(canvas, styles, footer, page, total):
    y = PAGE_HEIGHT - MARGIN_BOTTOM + 16
    canvas.line(MARGIN_LEFT, y - 5, PAGE_WIDTH - MARGIN_RIGHT, RULE, 0.5)
    canvas.text(styles.footer, MUTED, MARGIN_LEFT, y, BODY_WIDTH * 0.8, 14, footer)
    canvas.text(styles.footer, MUTED, MARGIN_LEFT, y, BODY_WIDTH, 14, f"Página {page} de {total}", Qt.AlignRight | Qt.AlignTop)


def render_pdf(document, path):
    """Escribe el PDF y devuelve su registro de maquetación."""
    writer = QPdfWriter(path)
    writer.setResolution(RESOLUTION)
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Landscape, QMarginsF(0, 0, 0, 0)))
    writer.setTitle(document.title)
    writer.setCreator("Test & Score Pro")
    styles = Styles(writer)
    pages, layout = paginate(_flows(document, styles), document.footer)
    painter = QPainter(writer)
    if not painter.isActive():
        raise OSError(f"No se puede escribir en {path}")
    canvas = Canvas(painter, styles.scale)
    try:
        for number, drawings in enumerate(pages, 1):
            if number > 1:
                writer.newPage()
            for draw, x, y in drawings:
                draw(canvas, x, y)
            _draw_footer(canvas, styles, document.footer, number, layout.pages)
    finally:
        painter.end()
    return layout
