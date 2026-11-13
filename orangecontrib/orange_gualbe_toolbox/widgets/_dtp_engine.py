# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Motor de Data Table Pro (sin Qt): el estado editable de una tabla y sus operaciones.

Cada tabla de entrada se convierte una sola vez en columnas (vectores de numpy) y todo lo
demás trabaja sobre ellas: la vista lee celdas sueltas sin crear objetos y las operaciones
masivas (buscar, reemplazar, prefijos, conversiones, ordenar) son vectoriales sobre los
valores distintos de cada columna.

Cada operación que cambia datos queda en un registro serializable (`log`), que permite
guardar los cambios con el workflow y repetirlos cuando vuelve a llegar la misma tabla, y
en una pila de deshacer con instantáneas de las columnas afectadas.
"""
import copy
import hashlib
import math
import re
from datetime import timezone as dt_timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from Orange.data import (ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table,
                         TimeVariable)

NUMERIC, CATEGORICAL, DATETIME, STRING = "numeric", "categorical", "datetime", "string"
KINDS = (CATEGORICAL, NUMERIC, DATETIME, STRING)
KIND_NAMES = {CATEGORICAL: "Categórica", NUMERIC: "Numérica", DATETIME: "Fecha y hora", STRING: "Texto"}
ATTRIBUTE, TARGET, META = "attribute", "target", "meta"
ROLES = (ATTRIBUTE, TARGET, META)
ROLE_NAMES = {ATTRIBUTE: "Atributo", TARGET: "Objetivo", META: "Meta"}
UNDO_LIMIT = 50
# Una tabla dispersa se muestra y se emite densa solo si cabe con holgura en memoria.
MAX_DENSE_CELLS = 50_000_000
MAX_CATEGORIES = 10_000


class OperationError(ValueError):
    """Operación rechazada; el texto es el mensaje para el usuario."""


def _kind_of(var):
    if var.is_time:
        return DATETIME
    if var.is_continuous:
        return NUMERIC
    if var.is_discrete:
        return CATEGORICAL
    return STRING


def _is_unknown_text(text):
    return text is None or str(text).strip() in ("", "?")


_TZ_SUFFIX = re.compile(r"(Z|[+-]\d\d:?\d\d)$")
_ISO_PLAIN = re.compile(r"^\d{4}-\d\d-\d\d([ T]\d\d:\d\d(:\d\d(\.\d+)?)?)?$")


def _offset_seconds(tz):
    """Desfase fijo de una zona horaria en segundos (0 para UTC o sin zona)."""
    if tz is None or tz == dt_timezone.utc:
        return 0.0
    try:
        offset = tz.utcoffset(None)
    except Exception:  # pylint: disable=broad-except
        return None
    return None if offset is None else offset.total_seconds()


def _format_numbers(values, var):
    """Lo mismo que ContinuousVariable.str_val, en un bucle sin llamadas a métodos.

    Con 300.000 valores distintos, llamar a str_val uno a uno tarda diez veces más.
    """
    fmt = getattr(var, "format_str", None) or getattr(var, "_format_str", "%g")
    decimals = getattr(var, "_number_of_decimals", 3)
    limit = getattr(var, "_max_round_diff", None)
    if fmt == "%g" or limit is None:
        return np.array(["?" if v != v else fmt % v for v in values.tolist()], dtype=object)
    wide = f"{{:.{decimals + 2}f}}"
    return np.array(["?" if v != v else (wide.format(v) if abs(round(v, decimals) - v) > limit else fmt % v)
                     for v in values.tolist()], dtype=object)


def _exact(text, value):
    try:
        return float(text) == value
    except (TypeError, ValueError):
        return False


def full_numbers(values):
    """Textos con toda la precisión (el más corto que reproduce el número), vectorizado."""
    texts = np.asarray(values, dtype=float).astype(str).astype(object)
    integral = np.array([t.endswith(".0") for t in texts], dtype=bool)
    texts[integral] = np.array([t[:-2] for t in texts[integral]], dtype=object)
    return texts


def _parse_number(text):
    cleaned = str(text).strip().replace("−", "-")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")  # coma decimal
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


@dataclass
class Column:
    cid: str            # identificador estable: el nombre original
    name: str
    kind: str
    role: str
    values: np.ndarray  # float para numéricas, categóricas (código) y fechas; object (str) para texto
    categories: List[str] = field(default_factory=list)
    var: object = None  # variable original mientras no cambie su definición
    edited: Optional[np.ndarray] = None
    have_date: int = 1
    have_time: int = 0
    tz: object = None   # zona horaria de una columna de fecha (None = UTC)
    _out_var: object = None

    def definition_changed(self):
        self.var = None
        self._out_var = None

    def snapshot(self):
        clone = copy.copy(self)
        clone.values = self.values.copy()
        clone.categories = list(self.categories)
        clone.edited = None if self.edited is None else self.edited.copy()
        return clone

    # ----------------------------------------------------------- variables ----

    def format_var(self):
        """Variable con la que se formatean los valores (la original o una nueva)."""
        if self.var is not None:
            return self.var
        if self._out_var is None:
            self._out_var = self._new_var()
        return self._out_var

    def output_var(self):
        if self.var is not None and self.var.name == self.name:
            return self.var
        if self.var is not None:
            self._out_var = self.var.copy(name=self.name)
            self.var = None
        if self._out_var is None or self._out_var.name != self.name:
            self._out_var = self._new_var()
        return self._out_var

    def _new_var(self):
        if self.kind == NUMERIC:
            return ContinuousVariable(self.name)
        if self.kind == DATETIME:
            var = TimeVariable(self.name, have_date=self.have_date, have_time=self.have_time)
            if self.tz is not None:
                var.timezone = self.tz
            return var
        if self.kind == CATEGORICAL:
            return DiscreteVariable(self.name, values=tuple(self.categories))
        return StringVariable(self.name)

    # -------------------------------------------------------------- texto ----

    def unknown_mask(self, rows=None):
        values = self.values if rows is None else self.values[rows]
        if self.kind == STRING:
            return np.array([v is None or v == "" for v in values], dtype=bool)
        return np.isnan(values.astype(float))

    def text(self, row):
        value = self.values[row]
        if self.kind == STRING:
            return "" if value is None else str(value)
        if np.isnan(value):
            return "?"
        if self.kind == CATEGORICAL:
            return self.categories[int(value)]
        return self.format_var().repr_val(value) if self.kind == DATETIME else self.format_var().str_val(value)

    def texts(self, rows=None):
        """Textos de las filas `rows` (todas si es None), formateando cada valor distinto una vez."""
        values = self.values if rows is None else self.values[rows]
        if self.kind == STRING:
            return np.array(["" if v is None else str(v) for v in values], dtype=object)
        if self.kind == CATEGORICAL:
            labels = np.array(list(self.categories) + ["?"], dtype=object)
            codes = np.where(np.isnan(values), len(self.categories), np.nan_to_num(values)).astype(int)
            return labels[codes]
        if not len(values):
            return np.array([], dtype=object)
        if self.kind == DATETIME:
            fast = _format_times(values, self.have_date, self.have_time, self.tz)
            if fast is not None:
                return fast
        unique, inverse = np.unique(values, return_inverse=True)
        var = self.format_var()
        if self.kind == NUMERIC:
            formatted = _format_numbers(unique, var)
        else:
            formatted = np.array(["?" if np.isnan(v) else var.repr_val(v) for v in unique], dtype=object)
        return formatted[inverse.ravel()]

    def edit_text(self, row):
        """Texto para editar una celda: el que se ve si es exacto; si redondea, el valor completo."""
        if self.kind == NUMERIC:
            value = self.values[row]
            return "" if np.isnan(value) else self.edit_texts(np.array([row]))[0]
        text = self.text(row)
        return "" if text == "?" and self.kind != STRING else text

    def edit_texts(self, rows=None):
        """Como `texts`, pero sin perder precisión: en las numéricas, el texto visible si reproduce
        exactamente el valor («5.0») y, si lo redondea («3.14159»), el valor completo."""
        shown = self.texts(rows)
        if self.kind != NUMERIC:
            return shown
        values = self.values if rows is None else self.values[rows]
        known = ~np.isnan(values)
        if not known.any():
            return shown
        texts = shown.copy()
        unique, inverse = np.unique(values[known], return_inverse=True)
        shown_unique = np.empty(len(unique), dtype=object)
        shown_unique[inverse.ravel()] = shown[known]
        exact = np.array([_exact(t, v) for t, v in zip(shown_unique, unique)], dtype=bool)
        if not exact.all():
            fixed = shown_unique.copy()
            fixed[~exact] = full_numbers(unique[~exact])
            texts[known] = fixed[inverse.ravel()]
        return texts

    def parse(self, text, extend=True):
        """Valor interno de un texto. Lanza OperationError si no es válido para el tipo."""
        if self.kind == STRING:
            return "" if text is None else str(text)
        if _is_unknown_text(text):
            return np.nan
        text = str(text).strip()
        if self.kind == NUMERIC:
            try:
                return _parse_number(text)
            except ValueError:
                raise OperationError(f"«{text}» no es un número válido para la columna numérica «{self.name}».") from None
        if self.kind == DATETIME:
            value, has_date, has_time = self.parse_time(text)
            if value is None:
                raise OperationError(f"«{text}» no es una fecha u hora válida para «{self.name}» "
                                     "(usa por ejemplo 2026-11-13 o 2026-11-13 09:30).")
            self.widen_time(has_date, has_time)
            return value
        if text in self.categories:
            return float(self.categories.index(text))
        if not extend:
            raise OperationError(f"«{text}» no es un valor de «{self.name}».")
        if len(self.categories) >= MAX_CATEGORIES:
            raise OperationError(f"«{self.name}» ya tiene {MAX_CATEGORIES} valores distintos.")
        self.categories.append(text)
        self.definition_changed()
        return float(len(self.categories) - 1)

    def parse_time(self, text):
        """Fecha u hora en segundos. Sin zona escrita, se toma la de la columna."""
        parser = TimeVariable("t", have_date=0, have_time=0)
        try:
            value = float(parser.parse(text))
        except Exception:  # pylint: disable=broad-except
            return None, 0, 0
        if not _TZ_SUFFIX.search(text.strip()):
            offset = _offset_seconds(self.tz)
            value -= offset or 0.0
        return value, int(parser.have_date), int(parser.have_time)

    def widen_time(self, has_date, has_time):
        """Si se escribe una hora en una columna solo de fechas (o al revés), se muestra."""
        if (has_time and not self.have_time) or (has_date and not self.have_date):
            self.have_time = int(self.have_time or has_time)
            self.have_date = int(self.have_date or has_date)
            if self.var is not None:
                self.tz = self.tz if self.tz is not None else getattr(self.var, "_timezone", None)
            self.definition_changed()

    def mark_edited(self, rows):
        if self.edited is None:
            self.edited = np.zeros(len(self.values), dtype=bool)
        self.edited[rows] = True

    # -------------------------------------------------------- estadística ----

    def stats(self):
        n = len(self.values)
        unknown = int(self.unknown_mask().sum())
        lines = [f"{KIND_NAMES[self.kind]} · {ROLE_NAMES[self.role]}",
                 f"Filas: {n} · desconocidos: {unknown}" + (f" ({unknown / n:.1%})" if n else "")]
        if self.kind in (NUMERIC, DATETIME) and n - unknown:
            values = self.values[~np.isnan(self.values)]
            var = self.format_var()
            show = var.repr_val if self.kind == DATETIME else var.str_val
            lines.append(f"Mínimo: {show(values.min())} · máximo: {show(values.max())}")
            if self.kind == NUMERIC:
                lines.append(f"Media: {np.mean(values):.4g} · mediana: {np.median(values):.4g} · "
                             f"desv. típica: {np.std(values):.4g}")
        elif self.kind == CATEGORICAL and n - unknown:
            counts = np.bincount(self.values[~np.isnan(self.values)].astype(int), minlength=len(self.categories))
            top = np.argsort(-counts, kind="stable")[:5]
            lines.append(f"Valores distintos: {int((counts > 0).sum())} de {len(self.categories)}")
            lines.append("Más frecuentes: " + ", ".join(f"{self.categories[i]} ({counts[i]})" for i in top if counts[i]))
        elif self.kind == STRING and n - unknown:
            lines.append(f"Textos distintos: {len(set(self.values.tolist()))}")
        return lines


def _signature(table):
    domain = table.domain
    return [len(table)] + [[var.name, type(var).__name__] for var in domain.variables + domain.metas]


FINGERPRINT_ROWS = 2000


def fingerprint(table):
    """Huella barata del contenido: una muestra de filas repartidas por toda la tabla.

    No usa los ids, que Orange cambia cada vez que vuelve a leer un archivo.
    """
    n = len(table)
    if not n:
        return "vacía"
    rows = np.unique(np.linspace(0, n - 1, min(n, FINGERPRINT_ROWS)).round().astype(int))
    digest = hashlib.sha1()
    for block in (table.X, table.Y, table.metas):
        part = block[rows] if not sp.issparse(block) else block[rows].toarray()
        part = np.asarray(part)
        if part.dtype == object:
            digest.update("\x1f".join(map(str, part.ravel().tolist())).encode("utf-8", "replace"))
        else:
            digest.update(np.ascontiguousarray(part, dtype=float).tobytes())
    return digest.hexdigest()[:16]


@dataclass
class Snapshot:
    columns: Dict[str, Column]
    names: Dict[str, str]


class TableState:
    """Estado editable de una tabla: columnas, orden de filas, registro y deshacer."""

    def __init__(self, table, name=""):
        self.source = table
        self.name = name or getattr(table, "name", "") or "Datos"
        self.signature = _signature(table)
        self.fingerprint = fingerprint(table)
        self.notes = []
        self.columns: List[Column] = []
        self._load(table)
        self.n_rows = len(table)
        self.order = np.arange(self.n_rows)
        self.sort_keys = []          # [(cid, descending)], de la más antigua a la más reciente
        self.log = []                # operaciones de datos (serializables), en orden
        self._undo, self._redo = [], []
        self.version = 0             # cambia con cada modificación de datos u orden
        self._texts = {}             # cid → (versión, textos de todas las filas)

    # ------------------------------------------------------------ carga ----

    def _load(self, table):
        domain = table.domain
        X, Y, metas = table.X, table.Y, table.metas
        if sp.issparse(X) or sp.issparse(Y) or sp.issparse(metas):
            cells = len(table) * len(domain.variables + domain.metas)
            if cells > MAX_DENSE_CELLS:
                raise OperationError(f"La tabla dispersa tiene {cells:,} celdas: es demasiado grande para "
                                     "editarla como tabla densa.".replace(",", "."))
            X = X.toarray() if sp.issparse(X) else X
            Y = Y.toarray() if sp.issparse(Y) else Y
            metas = metas.toarray() if sp.issparse(metas) else metas
            self.notes.append("La tabla de entrada es dispersa: se muestra y se emite como tabla densa.")
        n = len(table)
        X = np.asarray(X, dtype=float).reshape(n, len(domain.attributes))
        Y = np.asarray(Y, dtype=float).reshape(n, len(domain.class_vars))
        metas = np.asarray(metas, dtype=object).reshape(n, len(domain.metas))
        parts = [(domain.attributes, X, ATTRIBUTE), (domain.class_vars, Y, TARGET), (domain.metas, metas, META)]
        for variables, block, role in parts:
            for index, var in enumerate(variables):
                kind = _kind_of(var)
                raw = block[:, index]
                if kind == STRING:
                    values = np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                                       for v in raw], dtype=object)
                else:
                    values = np.array(raw, dtype=float)
                column = Column(var.name, var.name, kind, role, values,
                                list(var.values) if kind == CATEGORICAL else [], var)
                if kind == DATETIME:
                    column.have_date, column.have_time = int(var.have_date), int(var.have_time)
                    zone = getattr(var, "_timezone", None)
                    column.tz = zone if zone not in (None, "different timezones") else None
                self.columns.append(column)

    def column_texts(self, cid, full=False):
        """Textos de todas las filas de una columna, guardados hasta el siguiente cambio.

        Con `full`, las numéricas van con toda su precisión (para buscar y reemplazar).
        """
        key = (cid, full)
        cached = self._texts.get(key)
        if cached is None or cached[0] != self.version:
            column = self.column(cid)
            cached = (self.version, column.edit_texts() if full else column.texts())
            self._texts[key] = cached
        return cached[1]

    def _texts_of(self, column, rows):
        if rows is None or len(rows) == self.n_rows:
            return self.column_texts(column.cid, full=True)
        return column.edit_texts(rows)

    def column(self, cid):
        for column in self.columns:
            if column.cid == cid:
                return column
        raise OperationError(f"No existe la columna «{cid}».")

    def index_of(self, cid):
        return next(i for i, column in enumerate(self.columns) if column.cid == cid)

    def compatible_with(self, table):
        return _signature(table) == self.signature

    # ------------------------------------------------------------- orden ----

    def sort(self, cid, descending=False, record=True):
        """Ordena de forma estable a partir del orden actual (la última ordenación manda)."""
        column = self.column(cid)
        values = column.values[self.order]
        if column.kind == STRING:
            unknown = np.array([v is None or v == "" for v in values], dtype=bool)
            keys = np.array(["" if v is None else str(v).lower() for v in values], dtype=object)
            ranks = np.unique(keys, return_inverse=True)[1].ravel()
        else:
            unknown = np.isnan(values)
            ranks = np.unique(np.where(unknown, 0.0, values), return_inverse=True)[1].ravel()
        known = np.flatnonzero(~unknown)
        keys = -ranks[known] if descending else ranks[known]
        # Estable en los dos sentidos: los iguales conservan su orden; los desconocidos, al final.
        position = np.r_[known[np.argsort(keys, kind="stable")], np.flatnonzero(unknown)]
        self.order = self.order[position]
        if record:
            self.sort_keys.append([cid, bool(descending)])
        self.version += 1

    def restore_order(self):
        self.order = np.arange(self.n_rows)
        self.sort_keys = []
        self.version += 1

    # ------------------------------------------------------- deshacer ----

    def _begin(self, cids):
        return Snapshot({cid: self.column(cid).snapshot() for cid in cids},
                        {c.cid: c.name for c in self.columns})

    def _commit(self, before, entry, description):
        after = Snapshot({cid: self.column(cid).snapshot() for cid in before.columns},
                         {c.cid: c.name for c in self.columns})
        entry = dict(entry, id=self._next_id())
        self.log.append(entry)
        self._undo.append((description, before, after, entry))
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self.version += 1

    def _next_id(self):
        return max([e.get("id", 0) for e in self.log] + [0]) + 1

    def _restore(self, snapshot):
        for cid, saved in snapshot.columns.items():
            self.columns[self.index_of(cid)] = saved.snapshot()
        for column in self.columns:
            column.name = snapshot.names.get(column.cid, column.name)
        self.version += 1

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo_text(self):
        return self._undo[-1][0] if self._undo else ""

    def redo_text(self):
        return self._redo[-1][0] if self._redo else ""

    def undo(self):
        if not self._undo:
            return ""
        item = self._undo.pop()
        description, before, _, entry = item
        self._restore(before)
        self.log = [e for e in self.log if e.get("id") != entry["id"]]
        self._redo.append(item)
        return description

    def redo(self):
        if not self._redo:
            return ""
        item = self._redo.pop()
        description, _, after, entry = item
        self._restore(after)
        self.log.append(entry)
        self._undo.append(item)
        return description

    # ------------------------------------------------------- operaciones ----

    def edit(self, row, cid, text):
        """Cambia una celda (`row` es la fila original). Lanza OperationError si no es válida."""
        column = self.column(cid)
        if text is not None and str(text) in (column.edit_text(row), column.text(row)):
            return False  # se ha abierto y confirmado el editor sin cambiar nada (sin redondear)
        before = self._begin([cid])
        try:
            value = column.parse(text)
        except OperationError:
            self._restore(before)
            self.version -= 1
            raise
        old = column.values[row]
        if column.kind == STRING:
            same = old == value
        else:
            same = (np.isnan(old) and np.isnan(value)) or old == value
        if same:
            self._restore(before)
            self.version -= 1
            return False
        column.values[row] = value
        column.mark_edited([row])
        self._commit(before, dict(op="edit", row=int(row), col=cid, text="" if text is None else str(text)),
                     f"Editar «{column.name}», fila {row + 1}")
        return True

    def _scope(self, rows, cids):
        rows = np.arange(self.n_rows) if rows is None else np.unique(_expand_rows(rows))
        columns = self.columns if cids is None else [self.column(c) for c in cids]
        return rows, columns

    @staticmethod
    def _compile(pattern, regex, case):
        if not pattern:
            raise OperationError("Escribe el texto que quieres buscar.")
        flags = 0 if case else re.IGNORECASE
        try:
            return re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as error:
            raise OperationError(f"La expresión regular no es válida: {error}.") from None

    def find(self, pattern, regex=False, case=False, rows=None, cids=None):
        """Coincidencias como lista de (fila original, cid), en el orden de la vista."""
        compiled = self._compile(pattern, regex, case)
        rows, columns = self._scope(rows, cids)
        if not len(rows):
            return []
        hits = []
        position = np.empty(self.n_rows, dtype=int)
        position[self.order] = np.arange(self.n_rows)
        for column in columns:
            texts = self._texts_of(column, rows)
            known = ~column.unknown_mask(rows)
            matched = np.array([bool(known[i]) and compiled.search(t) is not None for i, t in enumerate(texts)],
                               dtype=bool)
            for row in rows[matched]:
                hits.append((int(row), column.cid))
        order_index = {c.cid: i for i, c in enumerate(self.columns)}
        hits.sort(key=lambda hit: (position[hit[0]], order_index[hit[1]]))
        return hits

    def replace(self, pattern, replacement, regex=False, case=False, rows=None, cids=None, record=True):
        """Reemplaza en el ámbito. Devuelve (reemplazos, celdas cambiadas, rechazadas por columna)."""
        compiled = self._compile(pattern, regex, case)
        if regex:
            replacement_fn = replacement  # admite grupos: \1, \g<1>, \g<nombre>
        else:
            literal = replacement
            replacement_fn = lambda _match: literal  # noqa: E731  (sin interpretar barras)
        rows, columns = self._scope(rows, cids)
        before = self._begin([c.cid for c in columns])
        total, changed, rejected = 0, 0, {}
        for column in columns:
            if column.kind == CATEGORICAL:
                count, cells, bad = self._replace_categorical(column, compiled, replacement_fn, rows, before)
                total += count
                changed += cells
                if bad:
                    rejected[column.name] = bad
                continue
            # En las numéricas se reemplaza sobre el valor completo, nunca sobre el texto redondeado.
            texts = self._texts_of(column, rows)
            known = ~column.unknown_mask(rows)
            positions, new_texts, counts = [], [], []
            for i in np.flatnonzero(known):
                text = texts[i]
                try:
                    new_text, count = compiled.subn(replacement_fn, text)
                except (re.error, IndexError) as error:
                    self._restore(before)
                    self.version -= 1
                    raise OperationError(f"El reemplazo no es válido: {error}.") from None
                if count and new_text != text:
                    positions.append(i)
                    new_texts.append(new_text)
                    counts.append(count)
            if not positions:
                continue
            positions = np.asarray(positions, dtype=int)
            counts = np.asarray(counts, dtype=int)
            values, valid = _parse_many(column, np.asarray(new_texts, dtype=object))
            bad = int((~valid).sum())
            if bad:
                rejected[column.name] = bad
            targets = rows[positions]
            if column.kind == STRING:
                differs = valid
            else:
                old = column.values[targets]
                same = (old == values) | (np.isnan(old) & np.isnan(values))
                differs = valid & ~same  # si el número no cambia, la celda no se toca
            if differs.any():
                column.values[targets[differs]] = values[differs] if column.kind != STRING else \
                    np.asarray(values, dtype=object)[differs]
                column.mark_edited(targets[differs])
                changed += int(differs.sum())
                total += int(counts[differs].sum())
        if not changed:
            self._restore(before)
            self.version -= 1
            return 0, 0, rejected
        if record:
            self._commit(before, dict(op="replace", pattern=pattern, replacement=replacement, regex=bool(regex),
                                      case=bool(case), rows=None if rows is None else _rows_or_none(rows, self.n_rows),
                                      cols=None if cids is None else list(cids)),
                         f"Reemplazar «{pattern}»")
        return total, changed, rejected

    def _replace_categorical(self, column, compiled, replacement, rows, before):
        """Reemplazo por valor distinto: cada etiqueta se transforma una vez.

        El valor nuevo ocupa el lugar del antiguo en la lista de valores, así que el orden de
        las categorías se conserva; los que dejan de usarse desaparecen.
        """
        codes = column.values[rows]
        known = ~np.isnan(codes)
        present = np.unique(codes[known]).astype(int)
        renamed, total = {}, 0
        for code in present:
            label = column.categories[code]
            try:
                new_label, count = compiled.subn(replacement, label)
            except (re.error, IndexError) as error:
                self._restore(before)
                self.version -= 1
                raise OperationError(f"El reemplazo no es válido: {error}.") from None
            if count and new_label != label:
                if not _is_unknown_text(new_label) and new_label.strip() not in column.categories \
                        and len(column.categories) + len(renamed) >= MAX_CATEGORIES:
                    self._restore(before)
                    self.version -= 1
                    raise OperationError(f"«{column.name}» superaría {MAX_CATEGORIES} valores distintos.")
                renamed[code] = (new_label, count)
        if not renamed:
            return 0, 0, 0
        old = list(column.categories)
        in_scope = np.zeros(len(column.values), dtype=bool)
        in_scope[rows] = True
        all_codes = column.values
        final = []

        def add(label):
            if label not in final:
                final.append(label)
            return final.index(label)

        new_values = all_codes.copy()
        cells = total = 0
        for code, label in enumerate(old):
            mask = all_codes == code
            if code not in renamed:
                new_values[mask] = add(label)
                continue
            outside = mask & ~in_scope
            if outside.any():
                new_values[outside] = add(label)  # fuera del ámbito sigue igual
            inside = mask & in_scope
            new_label = renamed[code][0]
            # «» o «?» es un valor desconocido, como al editar; si no, ocupa el sitio del antiguo.
            new_values[inside] = np.nan if _is_unknown_text(new_label) else add(new_label.strip())
            cells += int(inside.sum())
            total += int(inside.sum()) * renamed[code][1]
        column.values = new_values
        column.categories, column.values = _drop_unused(final, column.values)
        column.definition_changed()
        column.mark_edited(rows[np.isin(codes, list(renamed))])
        return total, cells, 0

    def affix(self, prefix="", suffix="", rows=None, cids=None):
        """Añade prefijo y sufijo a las celdas de texto y categóricas. Devuelve (celdas, columnas omitidas)."""
        if not prefix and not suffix:
            raise OperationError("Escribe un prefijo o un sufijo.")
        rows, columns = self._scope(rows, cids)
        skipped = [c.name for c in columns if c.kind in (NUMERIC, DATETIME)]
        targets = [c for c in columns if c.kind in (STRING, CATEGORICAL)]
        if not targets:
            return 0, skipped
        before = self._begin([c.cid for c in targets])
        try:
            changed = self._affix_columns(targets, rows, prefix, suffix)
        except OperationError:
            self._restore(before)  # p. ej. al superar el máximo de valores: la columna queda como estaba
            self.version -= 1
            raise
        if not changed:
            self._restore(before)
            self.version -= 1
            return 0, skipped
        self._commit(before, dict(op="affix", prefix=prefix, suffix=suffix,
                                  rows=None if rows is None else _rows_or_none(rows, self.n_rows),
                                  cols=None if cids is None else list(cids)),
                     "Añadir prefijo y sufijo")
        return changed, skipped

    @staticmethod
    def _affix_columns(targets, rows, prefix, suffix):
        changed = 0
        for column in targets:
            known = rows[~column.unknown_mask(rows)]
            if not len(known):
                continue
            if column.kind == STRING:
                column.values[known] = np.array([f"{prefix}{v}{suffix}" for v in column.values[known]], dtype=object)
            else:
                codes = column.values[known].astype(int)
                new_codes = np.empty(len(codes), dtype=float)
                for code in np.unique(codes):
                    label = f"{prefix}{column.categories[code]}{suffix}"
                    new_codes[codes == code] = column.parse(label)
                column.values[known] = new_codes
                column.categories, column.values = _drop_unused(column.categories, column.values)
                column.definition_changed()
            column.mark_edited(known)
            changed += len(known)
        return changed

    def rename(self, cid, new_name):
        column = self.column(cid)
        new_name = (new_name or "").strip()
        if not new_name:
            raise OperationError("El nombre de la columna no puede quedar vacío.")
        if new_name == column.name:
            return False
        if any(c.name == new_name for c in self.columns):
            raise OperationError(f"Ya hay una columna llamada «{new_name}».")
        before = self._begin([cid])
        column.name = new_name
        self._commit(before, dict(op="rename", col=cid, name=new_name), f"Renombrar «{before.names[cid]}»")
        return True

    def set_role(self, cids, role):
        """Cambia el rol de una o varias columnas. Es atómico: si una no puede, no cambia ninguna."""
        cids = [cids] if isinstance(cids, str) else list(cids)
        if role not in ROLES:
            raise OperationError(f"Rol desconocido: {role}.")
        columns = [self.column(cid) for cid in cids]
        texts = [c.name for c in columns if c.kind == STRING and role != META]
        if texts:
            raise OperationError(f"{_names(texts)} {'es' if len(texts) == 1 else 'son'} de texto: en Orange solo "
                                 "puede ser meta. Conviértela antes a categórica o numérica. No se ha cambiado "
                                 "ningún rol.")
        changing = [c for c in columns if c.role != role]
        if not changing:
            return False
        before = self._begin([c.cid for c in changing])
        for column in changing:
            column.role = role
        entry = dict(op="role", role=role)
        entry.update(dict(col=changing[0].cid) if len(changing) == 1 else dict(cols=[c.cid for c in changing]))
        self._commit(before, entry, f"{_names([c.name for c in changing])} como {ROLE_NAMES[role].lower()}")
        return True

    def convert(self, cids, kind):
        """Cambia el tipo de una o varias columnas, de forma atómica.

        Con una columna devuelve cuántos valores conocidos pasan a desconocidos; con varias, un
        diccionario {nombre: perdidos} solo con las que pierden alguno.
        """
        single = isinstance(cids, str)
        cids = [cids] if single else list(cids)
        if kind not in KINDS:
            raise OperationError(f"Tipo desconocido: {kind}.")
        changing = [self.column(cid) for cid in cids if self.column(cid).kind != kind]
        if not changing:
            return 0 if single else {}
        before = self._begin([c.cid for c in changing])
        lost = {}
        try:
            for column in changing:
                count = self._convert_column(column, kind)
                if count:
                    lost[column.name] = count
        except OperationError:
            self._restore(before)  # ninguna columna a medias
            self.version -= 1
            raise
        entry = dict(op="convert", kind=kind)
        entry.update(dict(col=changing[0].cid) if len(changing) == 1 else dict(cols=[c.cid for c in changing]))
        self._commit(before, entry, f"{_names([c.name for c in changing])} a {KIND_NAMES[kind].lower()}")
        if single:
            return sum(lost.values())
        return lost

    @staticmethod
    def _convert_column(column, kind):
        texts = column.texts()
        known_before = ~column.unknown_mask()
        old_kind = column.kind
        if kind == STRING:
            column.values = np.where(known_before, texts, "").astype(object)
            column.categories = []
            column.role = META
        elif kind == CATEGORICAL:
            usable = known_before & np.array([not _is_unknown_text(t) for t in texts], dtype=bool)
            if old_kind == NUMERIC:
                unique = np.unique(column.values[usable])
                labels = [column.format_var().str_val(v) for v in unique]
                label_of = dict(zip(unique.tolist(), labels))
                keys = [label_of[v] if k else None for v, k in zip(column.values.tolist(), usable)]
            else:
                keys = [str(t).strip() if k else None for t, k in zip(texts, usable)]
                labels = sorted({k for k in keys if k is not None}, key=_natural_key)
            labels = list(dict.fromkeys(labels))
            if len(labels) > MAX_CATEGORIES:
                raise OperationError(f"«{column.name}» tiene {len(labels)} valores distintos: demasiados para una "
                                     f"columna categórica (máximo {MAX_CATEGORIES}).")
            index = {label: i for i, label in enumerate(labels)}
            column.values = np.array([np.nan if k is None else index[k] for k in keys], dtype=float)
            column.categories = labels
        elif kind == NUMERIC:
            if old_kind == DATETIME:
                column.values = column.values.astype(float)
            else:
                column.values = _parse_unique(texts, known_before, _parse_number)
            column.categories = []
        else:  # DATETIME
            if old_kind == NUMERIC:
                column.values = column.values.astype(float)  # se toman como segundos desde 1970
                column.have_date, column.have_time = 1, 1
            else:
                values, has_date, has_time = _parse_times(texts, known_before, column.tz)
                column.values = values
                column.have_date = int(has_date) or (0 if has_time else 1)
                column.have_time = int(has_time)
            column.categories = []
        column.kind = kind
        column.definition_changed()
        return int((known_before & column.unknown_mask()).sum())

    # --------------------------------------------------------- registro ----

    def replay(self, entries, sort_keys=()):
        """Repite un registro guardado. Devuelve los avisos de lo que no se pudo aplicar."""
        problems = []
        for entry in entries:
            try:
                kind = entry.get("op")
                if kind == "edit":
                    if not 0 <= int(entry["row"]) < self.n_rows:
                        raise OperationError("fila fuera de la tabla")
                    self.edit(int(entry["row"]), entry["col"], entry["text"])
                elif kind == "replace":
                    self.replace(entry["pattern"], entry["replacement"], entry["regex"], entry["case"],
                                 entry.get("rows"), entry.get("cols"))
                elif kind == "affix":
                    self.affix(entry["prefix"], entry["suffix"], entry.get("rows"), entry.get("cols"))
                elif kind == "rename":
                    self.rename(entry["col"], entry["name"])
                elif kind == "role":
                    self.set_role(entry.get("cols") or entry["col"], entry["role"])
                elif kind == "convert":
                    self.convert(entry.get("cols") or entry["col"], entry["kind"])
            except (OperationError, KeyError, StopIteration, TypeError, ValueError) as error:
                problems.append(f"{entry.get('op', '?')}: {error}")
        for cid, descending in sort_keys:
            try:
                self.sort(cid, descending)
            except OperationError as error:
                problems.append(f"orden: {error}")
        # Los pasos repetidos quedan en la pila: se pueden deshacer igual que en la sesión original.
        self._redo.clear()
        return problems

    def is_modified(self):
        return bool(self.log)

    # ----------------------------------------------------------- salida ----

    def output(self, column_order=None, name=None):
        """Tabla con todos los cambios, en el orden de filas actual y de columnas `column_order`."""
        order = list(column_order) if column_order else [c.cid for c in self.columns]
        known = {c.cid for c in self.columns}
        order = [cid for cid in order if cid in known] + [c.cid for c in self.columns if c.cid not in order]
        by_role = {role: [self.column(cid) for cid in order if self.column(cid).role == role] for role in ROLES}
        domain = Domain([c.output_var() for c in by_role[ATTRIBUTE]],
                        [c.output_var() for c in by_role[TARGET]],
                        [c.output_var() for c in by_role[META]])
        rows = self.order
        n = len(rows)
        X = np.column_stack([c.values[rows] for c in by_role[ATTRIBUTE]]) if by_role[ATTRIBUTE] else np.empty((n, 0))
        Y = np.column_stack([c.values[rows] for c in by_role[TARGET]]) if by_role[TARGET] else np.empty((n, 0))
        if Y.shape[1] == 1:
            Y = Y[:, 0]
        if by_role[META]:
            metas = np.empty((n, len(by_role[META])), dtype=object)
            for j, c in enumerate(by_role[META]):
                metas[:, j] = c.values[rows] if c.kind == STRING else c.values[rows].astype(float)
        else:
            metas = np.empty((n, 0), dtype=object)
        source = self.source
        W = source.W[rows] if source.has_weights() else None
        table = Table.from_numpy(domain, X.astype(float), Y.astype(float) if Y.size or Y.ndim == 1 else Y,
                                 metas, W=W, ids=source.ids[rows])
        table.name = name or self.name
        table.attributes = dict(getattr(source, "attributes", {}) or {})
        return table


def _names(names):
    shown = ", ".join(f"«{n}»" for n in names[:4]) + ("…" if len(names) > 4 else "")
    return shown


def _rows_or_none(rows, n_rows):
    """Filas para el registro, comprimidas en tramos [inicio, fin] (None si son todas)."""
    rows = np.unique(np.asarray(rows, dtype=int))
    if len(rows) == n_rows:
        return None
    if not len(rows):
        return {"ranges": []}
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts, stops = rows[np.r_[0, breaks + 1]], rows[np.r_[breaks, len(rows) - 1]]
    return {"ranges": [[int(a), int(b)] for a, b in zip(starts, stops)]}


def _expand_rows(rows):
    """Admite una lista de filas, un vector o los tramos guardados en el registro."""
    if isinstance(rows, dict):
        ranges = rows.get("ranges", [])
        if not ranges:
            return np.arange(0, dtype=int)
        return np.concatenate([np.arange(a, b + 1) for a, b in ranges]).astype(int)
    return np.asarray(rows, dtype=int).ravel()


def _drop_unused(categories, values):
    used = np.unique(values[~np.isnan(values)]).astype(int)
    if len(used) == len(categories):
        return categories, values
    mapping = np.full(len(categories), np.nan)
    mapping[used] = np.arange(len(used))
    new_values = values.copy()
    known = ~np.isnan(values)
    new_values[known] = mapping[values[known].astype(int)]
    return [categories[i] for i in used], new_values


def _natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _parse_unique(texts, known, parser):
    """Convierte textos a float analizando cada texto distinto una sola vez."""
    values = np.full(len(texts), np.nan)
    if not known.any():
        return values
    unique, inverse = np.unique(texts[known].astype(str), return_inverse=True)
    parsed = np.full(len(unique), np.nan)
    for i, text in enumerate(unique):
        try:
            value = float(parser(text))
            parsed[i] = value if math.isfinite(value) else np.nan
        except Exception:  # pylint: disable=broad-except
            pass
    values[known] = parsed[inverse.ravel()]
    return values


def _parse_many(column, texts):
    """Analiza muchos textos de una vez según el tipo. Devuelve (valores, válidos)."""
    n = len(texts)
    if column.kind == STRING:
        return np.array(["" if t is None else str(t) for t in texts], dtype=object), np.ones(n, dtype=bool)
    unknown = np.array([_is_unknown_text(t) for t in texts], dtype=bool)
    values = np.full(n, np.nan)
    valid = unknown.copy()
    rest = ~unknown
    if not rest.any():
        return values, valid
    if column.kind == NUMERIC:
        parsed = _parse_unique(texts, rest, _parse_number)
    else:
        parsed, has_date, has_time = _parse_times(texts, rest, column.tz)
        column.widen_time(has_date, has_time)
    ok = rest & ~np.isnan(parsed)
    values[ok] = parsed[ok]
    valid |= ok
    return values, valid


def _format_times(values, have_date, have_time, tz):
    """Fechas como las escribe Orange, vectorizado. None si hay que hacerlo valor a valor."""
    if tz is not None and tz != dt_timezone.utc:
        return None
    if not have_date and not have_time:
        return None
    values = np.asarray(values, dtype=float)
    known = ~np.isnan(values)
    finite = values[known]
    if len(finite) and (np.any(finite != np.round(finite)) or finite.min() < -6e10 or finite.max() > 2.5e11):
        return None  # fracciones de segundo o fechas fuera de los años 1-9999
    texts = np.full(len(values), "?", dtype=object)
    if known.any():
        stamps = np.datetime_as_string(finite.astype("int64").astype("datetime64[s]"), unit="s")
        if have_date and not have_time:
            stamps = np.array([t[:10] for t in stamps], dtype=object)
        elif have_time and not have_date:
            stamps = np.array([t[11:] for t in stamps], dtype=object)
        else:
            stamps = np.array([t.replace("T", " ") for t in stamps], dtype=object)
        texts[known] = stamps
    return texts


def _parse_times(texts, mask, tz):
    """Textos de fecha a segundos: los ISO sencillos con pandas y el resto como Orange.

    Sin zona escrita se toma la de la columna. Devuelve (valores, ¿alguno con fecha?, ¿con hora?).
    """
    values = np.full(len(texts), np.nan)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return values, 0, 0
    subset = np.array([str(t).strip() for t in texts[indices]], dtype=object)
    plain = np.array([bool(_ISO_PLAIN.match(t)) for t in subset], dtype=bool)
    has_date = has_time = 0
    if plain.any():
        import pandas as pd
        parsed = pd.to_datetime(pd.Series(subset[plain]), format="ISO8601", errors="coerce")
        seconds = parsed.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        seconds[parsed.isna().to_numpy()] = np.nan
        offset = _offset_seconds(tz) or 0.0
        values[indices[plain]] = seconds - offset
        has_date = 1
        has_time = int(any(len(t) > 10 for t in subset[plain]))
        plain[plain] = ~np.isnan(seconds)  # los que pandas no entiende pasan al análisis de Orange
    others = np.flatnonzero(~plain)
    if len(others):
        unique, inverse = np.unique(subset[others].astype(str), return_inverse=True)
        parsed = np.full(len(unique), np.nan)
        offset = _offset_seconds(tz) or 0.0
        for i, text in enumerate(unique):
            parser = TimeVariable("t", have_date=0, have_time=0)
            try:
                value = float(parser.parse(text))
            except Exception:  # pylint: disable=broad-except
                continue
            parsed[i] = value - (0.0 if _TZ_SUFFIX.search(text) else offset)
            has_date |= int(parser.have_date)
            has_time |= int(parser.have_time)
        values[indices[others]] = parsed[inverse.ravel()]
    return values, has_date, has_time
