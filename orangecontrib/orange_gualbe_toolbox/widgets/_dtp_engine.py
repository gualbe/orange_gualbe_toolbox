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
import math
import re
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
            return TimeVariable(self.name, have_date=self.have_date, have_time=self.have_time)
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
        unique, inverse = np.unique(values, return_inverse=True)
        var = self.format_var()
        formatter = var.repr_val if self.kind == DATETIME else var.str_val
        formatted = np.array(["?" if np.isnan(v) else formatter(v) for v in unique], dtype=object)
        return formatted[inverse.ravel()]

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
            try:
                return float(TimeVariable("t").parse(text))
            except Exception:  # pylint: disable=broad-except
                raise OperationError(f"«{text}» no es una fecha u hora válida para «{self.name}» "
                                     "(usa por ejemplo 2026-11-13 o 2026-11-13 09:30).") from None
        if text in self.categories:
            return float(self.categories.index(text))
        if not extend:
            raise OperationError(f"«{text}» no es un valor de «{self.name}».")
        if len(self.categories) >= MAX_CATEGORIES:
            raise OperationError(f"«{self.name}» ya tiene {MAX_CATEGORIES} valores distintos.")
        self.categories.append(text)
        self.definition_changed()
        return float(len(self.categories) - 1)

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
                self.columns.append(column)

    def column_texts(self, cid):
        """Textos de todas las filas de una columna, guardados hasta el siguiente cambio."""
        cached = self._texts.get(cid)
        if cached is None or cached[0] != self.version:
            cached = (self.version, self.column(cid).texts())
            self._texts[cid] = cached
        return cached[1]

    def _texts_of(self, column, rows):
        if rows is None or len(rows) == self.n_rows:
            return self.column_texts(column.cid)
        return column.texts(rows)

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
        rows = np.arange(self.n_rows) if rows is None else np.asarray(sorted(set(int(r) for r in rows)), dtype=int)
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
            texts = column.texts(rows)
            known = ~column.unknown_mask(rows)
            new_values = []
            for i, text in enumerate(texts):
                if not known[i]:
                    continue
                try:
                    new_text, count = compiled.subn(replacement_fn, text)
                except (re.error, IndexError) as error:
                    self._restore(before)
                    self.version -= 1
                    raise OperationError(f"El reemplazo no es válido: {error}.") from None
                if count and new_text != text:
                    new_values.append((i, new_text, count))
            done = []
            for i, new_text, count in new_values:
                try:
                    value = column.parse(new_text)
                except OperationError:
                    rejected[column.name] = rejected.get(column.name, 0) + 1
                    continue
                column.values[rows[i]] = value
                done.append(rows[i])
                total += count
            if done:
                column.mark_edited(done)
                changed += len(done)
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
            new_values[inside] = add(renamed[code][0])  # el nuevo ocupa el sitio del antiguo
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
        if not changed:
            self._restore(before)
            self.version -= 1
            return 0, skipped
        self._commit(before, dict(op="affix", prefix=prefix, suffix=suffix,
                                  rows=None if rows is None else _rows_or_none(rows, self.n_rows),
                                  cols=None if cids is None else list(cids)),
                     "Añadir prefijo y sufijo")
        return changed, skipped

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

    def set_role(self, cid, role):
        column = self.column(cid)
        if role not in ROLES:
            raise OperationError(f"Rol desconocido: {role}.")
        if role == column.role:
            return False
        if column.kind == STRING and role != META:
            raise OperationError(f"«{column.name}» es de texto: en Orange solo puede ser meta. "
                                 "Conviértela antes a categórica o numérica.")
        before = self._begin([cid])
        column.role = role
        self._commit(before, dict(op="role", col=cid, role=role),
                     f"«{column.name}» como {ROLE_NAMES[role].lower()}")
        return True

    def convert(self, cid, kind):
        """Cambia el tipo. Devuelve cuántos valores conocidos pasan a desconocidos."""
        column = self.column(cid)
        if kind not in KINDS:
            raise OperationError(f"Tipo desconocido: {kind}.")
        if kind == column.kind:
            return 0
        before = self._begin([cid])
        texts = column.texts()
        known_before = ~column.unknown_mask()
        old_kind = column.kind
        if kind == STRING:
            column.values = np.where(known_before, texts, "").astype(object)
            column.categories = []
            if column.role != META:
                column.role = META
        elif kind == CATEGORICAL:
            if old_kind == NUMERIC:
                unique = np.unique(column.values[known_before])
                labels = [column.format_var().str_val(v) for v in unique]
            else:
                labels = sorted({str(t) for t in texts[known_before]}, key=_natural_key)
            labels = list(dict.fromkeys(labels))
            if len(labels) > MAX_CATEGORIES:
                self._restore(before)
                self.version -= 1
                raise OperationError(f"«{column.name}» tiene {len(labels)} valores distintos: demasiados para una "
                                     f"columna categórica (máximo {MAX_CATEGORIES}).")
            index = {label: i for i, label in enumerate(labels)}
            column.values = np.array([index[str(t)] if k else np.nan for t, k in zip(texts, known_before)],
                                     dtype=float)
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
            else:
                parser = TimeVariable("t")
                column.values = _parse_unique(texts, known_before, parser.parse)
                column.have_date = int(parser.have_date) or 1
                column.have_time = int(parser.have_time)
            column.categories = []
        column.kind = kind
        column.definition_changed()
        lost = int((known_before & column.unknown_mask()).sum())
        self._commit(before, dict(op="convert", col=cid, kind=kind),
                     f"«{column.name}» a {KIND_NAMES[kind].lower()}")
        return lost

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
                    self.set_role(entry["col"], entry["role"])
                elif kind == "convert":
                    self.convert(entry["col"], entry["kind"])
            except (OperationError, KeyError, StopIteration, TypeError, ValueError) as error:
                problems.append(f"{entry.get('op', '?')}: {error}")
        for cid, descending in sort_keys:
            try:
                self.sort(cid, descending)
            except OperationError as error:
                problems.append(f"orden: {error}")
        self._undo.clear()
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


def _rows_or_none(rows, n_rows):
    rows = [int(r) for r in rows]
    return None if len(rows) == n_rows else rows


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
