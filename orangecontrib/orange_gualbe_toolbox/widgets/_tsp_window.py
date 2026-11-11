# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.
# Portions adapted from Orange Data Mining (Orange/evaluation/testing.py),
# Copyright (C) Bioinformatics Laboratory, University of Ljubljana, GPL-3.0-or-later.

"""Validación por ventana deslizante para series temporales (sin Qt).

Las filas se ordenan por una variable de tiempo (o se toman en su orden) y en cada iteración
se entrena con un bloque y se evalúa con el bloque siguiente, sin mirar nunca al futuro. El
plan de ventanas se calcula aparte de la evaluación: es barato y alimenta la vista previa.

Convenciones: las posiciones son índices de la tabla ordenada y los intervalos son
semiabiertos, [inicio, fin). Un corte nunca separa filas con la misma fecha: se desplaza
hasta después del grupo de fechas iguales.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from Orange.data import Domain
from Orange.evaluation import Results
from Orange.evaluation.testing import Validation, _mp_worker

START_FIRST, START_ROW, START_DATE = range(3)
INITIAL_PERCENT, INITIAL_ROWS, INITIAL_DATE = range(3)
EXPANDING, ROLLING = range(2)
UNIT_ROWS, UNIT_TIME = range(2)
END_ALL, END_DATE = range(2)
LAST_DISCARD, LAST_KEEP = range(2)

# Segundos por unidad; meses y años se suman con el calendario.
TIME_UNITS = (("segundos", 1.0), ("minutos", 60.0), ("horas", 3600.0), ("días", 86400.0),
              ("semanas", 604800.0), ("meses", "months"), ("años", "years"))
DEFAULT_TIME_UNIT = 3
# Un hueco más de 5 veces mayor que la separación habitual se avisa como irregular.
IRREGULAR_FACTOR = 5.0
# Una ventana de test medida en tiempo está completa si no pasa de la última fila en más de
# media separación habitual: con datos mensuales, el test de diciembre acaba el 1 de enero.
COMPLETE_SLACK = 0.5
MAX_PLANNED = 100000


class IterationError(Exception):
    """Fallo de un modelo en una iteración concreta, con su número para el mensaje."""

    def __init__(self, iteration, label, error):
        super().__init__(f"en la iteración {iteration + 1} ({label}): {type(error).__name__}: {error}")
        self.iteration = iteration
        self.error = error


@dataclass
class WindowSpec:
    time_var: object = None
    start_mode: int = START_FIRST
    start_row: int = 1
    start_date: float = math.nan
    initial_mode: int = INITIAL_PERCENT
    initial_percent: float = 50.0
    initial_rows: int = 100
    initial_date: float = math.nan
    advance: int = EXPANDING
    unit: int = UNIT_ROWS
    time_unit: int = DEFAULT_TIME_UNIT
    test_size: int = 10
    gap: int = 0
    step: int = 10
    step_same: bool = True
    end_mode: int = END_ALL
    end_date: float = math.nan
    last: int = LAST_DISCARD
    min_train: int = 10
    max_iterations: int = 200

    @property
    def effective_step(self):
        return self.test_size if self.step_same else self.step

    def amount(self, value):
        """Texto de una cantidad en la unidad de la ventana."""
        if self.unit == UNIT_ROWS:
            return f"{value} fila" + ("" if value == 1 else "s")
        name = TIME_UNITS[self.time_unit][0]
        if value == 1:
            name = {"meses": "mes", "años": "año"}.get(name, name[:-1])
        return f"{value} {name}"

    def describe(self):
        mode = "creciente (inicio fijo)" if self.advance == EXPANDING else "deslizante (tamaño fijo)"
        order = (f"ordenada por «{self.time_var.name}»" if self.time_var is not None
                 else "en el orden de las filas")
        gap = f", hueco de {self.amount(self.gap)}" if self.gap else ""
        return (f"Ventana {mode}, {order}: test de {self.amount(self.test_size)}, paso de "
                f"{self.amount(self.effective_step)}{gap}")


@dataclass
class Iteration:
    index: int
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int
    complete: bool = True

    @property
    def n_train(self):
        return self.train_hi - self.train_lo

    @property
    def n_test(self):
        return self.test_hi - self.test_lo


@dataclass
class WindowPlan:
    spec: WindowSpec
    order: np.ndarray
    times: Optional[np.ndarray]
    start: int = 0
    end: int = 0
    iterations: List[Iteration] = field(default_factory=list)
    planned: int = 0
    missing_time: int = 0
    skipped_small: int = 0
    skipped_empty: int = 0
    spacing: float = math.nan
    notes: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def n_rows(self):
        return len(self.order)

    def value_at(self, position):
        """Valor en el eje temporal: la fecha, o el número de fila (desde 1)."""
        if self.times is None:
            return float(position + 1)
        return float(self.times[min(max(position, 0), len(self.times) - 1)])

    def label(self, position):
        if self.times is None:
            return f"fila {position + 1}"
        return self.spec.time_var.repr_val(self.value_at(position))

    def span(self, lo, hi):
        """Texto «desde – hasta» de las filas [lo, hi)."""
        if hi <= lo:
            return "—"
        return f"{self.label(lo)} – {self.label(hi - 1)}"

    @property
    def tested(self):
        mask = np.zeros(self.n_rows, dtype=bool)
        for it in self.iterations:
            mask[it.test_lo:it.test_hi] = True
        return mask

    def unused_counts(self):
        """Filas que nunca se evalúan: antes del inicio, después del fin y en medio."""
        tested = self.tested
        last = max((it.test_hi for it in self.iterations), default=self.start)
        first = min((it.test_lo for it in self.iterations), default=self.start)
        return dict(before=self.start + self.missing_time, after=self.n_rows - self.end,
                    initial=first - self.start, between=int((~tested[first:last]).sum()),
                    tail=self.end - last)

    def summary(self, n_models=0, n_targets=1):
        """Una línea para el panel: iteraciones, entrenamientos y filas sin evaluar."""
        if self.error:
            return self.error
        count = len(self.iterations)
        parts = [f"{count} iteraci{'ón' if count == 1 else 'ones'}"]
        if n_models:
            trainings = count * n_models * n_targets
            parts.append(f"{trainings:,} entrenamientos".replace(",", "."))
        unused = self.unused_counts()
        tested = int(self.tested.sum())
        parts.append(f"{tested} filas evaluadas")
        lost = unused["between"] + unused["tail"]
        if lost:
            parts.append(f"{lost} sin evaluar")
        return " · ".join(parts)


# ---------------------------------------------------------------- utilidades ----

def column(data, var):
    """Valores de una variable como vector de float, esté donde esté en el dominio."""
    try:
        values = data.get_column(var)
    except AttributeError:  # Orange < 3.34
        values = data.get_column_view(var)[0]
    return np.asarray(values, dtype=float)


def shift(value, amount, unit):
    """Suma `amount` unidades de tiempo a una marca en segundos (meses y años, con calendario)."""
    factor = TIME_UNITS[unit][1]
    if not isinstance(factor, str):
        return value + amount * factor
    if not np.isfinite(value) or amount == 0:
        return value
    import pandas as pd
    stamp = pd.Timestamp(value, unit="s")
    offset = pd.DateOffset(months=amount) if factor == "months" else pd.DateOffset(years=amount)
    try:
        return (stamp + offset).value / 1e9
    except (OverflowError, ValueError):
        return math.inf if amount > 0 else -math.inf


def unit_seconds(unit):
    factor = TIME_UNITS[unit][1]
    return {"months": 30.44 * 86400.0, "years": 365.25 * 86400.0}.get(factor, factor)


def format_duration(seconds):
    if not np.isfinite(seconds):
        return "—"
    for name, size in (("años", 365.25 * 86400), ("días", 86400), ("horas", 3600), ("minutos", 60)):
        if seconds >= size:
            value = seconds / size
            return f"{value:.3g}".replace(".", ",") + f" {name}"
    return f"{seconds:.3g}".replace(".", ",") + " segundos"


def time_order(data, time_var):
    """Orden de las filas, tiempos ordenados, filas sin tiempo y avisos sobre el tiempo."""
    n = len(data)
    if time_var is None:
        return np.arange(n), None, 0, []
    values = column(data, time_var)
    valid = np.flatnonzero(np.isfinite(values))
    notes = []
    missing = n - len(valid)
    if missing:
        notes.append(f"{missing} filas sin valor en «{time_var.name}» quedan fuera de la validación.")
    ordered = values[valid]
    if len(ordered) > 1 and np.any(np.diff(ordered) < 0):
        notes.append(f"Las filas no estaban ordenadas por «{time_var.name}»: se ordenan antes de "
                     "formar las ventanas.")
    position = np.argsort(ordered, kind="stable")
    order = valid[position]
    times = ordered[position]
    if len(times) > 1:
        repeated = int(np.sum(np.diff(times) == 0))
        if repeated:
            notes.append(f"{repeated} filas repiten la fecha de la anterior: las filas con la misma "
                         "fecha nunca se separan entre entrenamiento y test.")
    return order, times, missing, notes


def _spacing(times):
    if times is None or len(times) < 2:
        return math.nan
    steps = np.diff(np.unique(times))
    return float(np.median(steps)) if len(steps) else math.nan


# ---------------------------------------------------------------------- plan ----

def plan_windows(data, spec):
    """Calcula las ventanas sin entrenar nada. Nunca lanza excepciones: rellena `error`."""
    order, times, missing, notes = time_order(data, spec.time_var)
    plan = WindowPlan(spec, order, times, missing_time=missing, notes=notes)
    try:
        _plan(plan)
    except Exception as error:  # pylint: disable=broad-except
        plan.error = f"No se han podido formar las ventanas: {error}"
        plan.iterations = []
    return plan


def _plan(plan):
    spec, times = plan.spec, plan.times
    n = plan.n_rows
    needs_time = (spec.unit == UNIT_TIME or spec.start_mode == START_DATE
                  or spec.initial_mode == INITIAL_DATE or spec.end_mode == END_DATE)
    if needs_time and times is None:
        plan.error = ("Para medir en tiempo o usar fechas hace falta ordenar por una variable de "
                      "tiempo: elígela en «Orden temporal».")
        return
    if n < 3:
        plan.error = f"Hay {n} filas con tiempo: hacen falta al menos 3 para una ventana."
        return
    if min(spec.test_size, spec.effective_step) < 1 or spec.gap < 0:
        plan.error = "El test y el paso deben ser de al menos 1 y el hueco no puede ser negativo."
        return
    plan.spacing = _spacing(times)
    if times is not None and np.isfinite(plan.spacing) and plan.spacing > 0:
        largest = float(np.max(np.diff(times)))
        if largest > IRREGULAR_FACTOR * plan.spacing:
            plan.notes.append(f"El tiempo tiene huecos irregulares: el mayor es de "
                              f"{format_duration(largest)} y la separación habitual, de "
                              f"{format_duration(plan.spacing)}.")

    def snap(position):
        """Desplaza un corte para no separar filas con la misma fecha."""
        if times is None or position <= 0 or position >= n:
            return max(position, 0)  # sin tope: quien llama compara con el fin
        if times[position] == times[position - 1]:
            return int(np.searchsorted(times, times[position - 1], side="right"))
        return position

    def first_at(value):
        return int(np.searchsorted(times, value, side="left"))

    # Inicio
    if spec.start_mode == START_ROW:
        if spec.start_row > n:
            plan.error = f"La fila de inicio ({spec.start_row}) supera las {n} filas disponibles."
            return
        start = max(spec.start_row, 1) - 1
        if times is not None:
            start = int(np.searchsorted(times, times[start], side="left"))
    elif spec.start_mode == START_DATE:
        date = spec.start_date if np.isfinite(spec.start_date) else times[0]
        if date > times[-1]:
            plan.error = (f"La fecha de inicio ({spec.time_var.repr_val(date)}) es posterior al "
                          f"último dato ({spec.time_var.repr_val(times[-1])}).")
            return
        start = first_at(date)
    else:
        start = 0
    # Fin
    end = n
    if spec.end_mode == END_DATE:
        date = spec.end_date if np.isfinite(spec.end_date) else times[-1]
        end = int(np.searchsorted(times, date, side="right"))
        if end <= start:
            plan.error = (f"La fecha de fin ({spec.time_var.repr_val(date)}) no es posterior al "
                          "inicio de la ventana.")
            return
    plan.start, plan.end = start, end
    available = end - start
    # Entrenamiento inicial
    if spec.initial_mode == INITIAL_PERCENT:
        cut = start + int(round(available * min(max(spec.initial_percent, 0.0), 100.0) / 100.0))
    elif spec.initial_mode == INITIAL_ROWS:
        cut = start + max(int(spec.initial_rows), 0)
    else:
        date = spec.initial_date
        if not np.isfinite(date):
            date = times[start + available // 2]
        cut = int(np.searchsorted(times, date, side="right"))
    cut = snap(cut)
    if cut <= start:
        plan.error = "El entrenamiento inicial queda vacío: amplíalo o adelanta el inicio."
        return
    if cut >= end:
        plan.error = (f"El entrenamiento inicial ({cut - start} filas) ocupa todas las "
                      f"{available} filas disponibles y no deja nada para el test: redúcelo.")
        return
    if cut - start < spec.min_train:
        plan.error = (f"El entrenamiento inicial tiene {cut - start} filas y el mínimo es "
                      f"{spec.min_train}: amplíalo o baja el mínimo.")
        return
    step, test, gap = spec.effective_step, spec.test_size, spec.gap
    iterations = []
    previous = None
    if spec.unit == UNIT_ROWS:
        length = cut - start
        for i in range(MAX_PLANNED):
            train_hi = snap(cut + i * step)
            test_lo = snap(train_hi + gap)
            if test_lo >= end:
                break
            test_hi = snap(test_lo + test)
            complete = test_hi <= end
            if not complete:
                if spec.last == LAST_DISCARD:
                    break
                test_hi = end
            train_lo = start if spec.advance == EXPANDING else snap(max(train_hi - length, start))
            key = (train_lo, train_hi, test_lo, test_hi)
            if key == previous:
                plan.skipped_empty += 1
                continue
            previous = key
            if train_hi - train_lo < spec.min_train:
                plan.skipped_small += 1
                continue
            iterations.append(Iteration(len(iterations), train_lo, train_hi, test_lo, test_hi, complete))
            if not complete:
                break
    else:
        unit = spec.time_unit
        origin = float(times[cut])
        first_time = float(times[start])
        spacing = plan.spacing if np.isfinite(plan.spacing) else 0.0
        limit = float(times[end - 1]) + (1 + COMPLETE_SLACK) * spacing
        for i in range(MAX_PLANNED):
            # Todas las fronteras se cuentan desde el origen, nunca desde otra frontera: con meses
            # y años, partir de una fecha ya recortada a fin de mes (31 → 29 de febrero) arrastra
            # el recorte y deja días sin evaluar entre tests.
            boundary = shift(origin, i * step, unit)
            test_from = shift(origin, i * step + gap, unit)
            test_lo = max(first_at(test_from), start)
            if test_lo >= end or not np.isfinite(test_from):
                break
            test_to = shift(origin, i * step + gap + test, unit)
            test_hi = min(first_at(test_to), end)
            complete = test_to <= limit
            if not complete and spec.last == LAST_DISCARD:
                break
            train_hi = first_at(boundary)
            if spec.advance == EXPANDING:
                train_lo = start
            else:
                train_lo = max(first_at(shift(first_time, i * step, unit)), start)
            if test_hi <= test_lo:
                plan.skipped_empty += 1
                continue
            if train_hi - train_lo < spec.min_train:
                plan.skipped_small += 1
                continue
            iterations.append(Iteration(len(iterations), train_lo, train_hi, test_lo, test_hi, complete))
            if not complete:
                break
    plan.planned = len(iterations)
    if plan.skipped_small:
        plan.notes.append(f"{plan.skipped_small} iteraciones con menos de {spec.min_train} filas de "
                          "entrenamiento se omiten.")
    if plan.skipped_empty:
        plan.notes.append(f"{plan.skipped_empty} iteraciones sin filas de test nuevas (huecos en el "
                          "tiempo o fechas repetidas) se omiten.")
    if not iterations:
        plan.error = ("No cabe ninguna iteración completa: tras el entrenamiento inicial no queda "
                      f"sitio para el hueco y un test de {spec.amount(test)}. Reduce el entrenamiento "
                      "inicial o el test, o usa también el último test incompleto.")
        return
    if len(iterations) > spec.max_iterations:
        plan.error = (f"Saldrían {len(iterations)} iteraciones y el límite es {spec.max_iterations}: "
                      "aumenta el paso o el test, o sube el límite de iteraciones.")
        plan.iterations = iterations
        return
    plan.iterations = iterations
    if any(not it.complete for it in iterations):
        plan.notes.append("La última iteración tiene un test incompleto, más corto que los demás.")
    if len(iterations) < 5:
        plan.notes.append(f"Solo {len(iterations)} iteraciones: las comparaciones entre iteraciones "
                          "tendrán poca potencia.")
    if step < test:
        plan.notes.append("El paso es menor que el test: los tests se solapan y algunas filas se "
                          "evalúan varias veces. Diebold-Mariano usa la predicción más reciente de "
                          "cada fila.")


# --------------------------------------------------------------- objetivos ----

def target_tables(data, plan):
    """Una tabla ordenada por cada variable objetivo, con las demás objetivo como metas.

    Las otras variables objetivo nunca pueden servir para aprender: serían valores futuros.
    """
    ordered = data[plan.order]
    class_vars = ordered.domain.class_vars
    if len(class_vars) == 1:
        return [ordered]
    tables = []
    for position, var in enumerate(class_vars):
        others = tuple(v for i, v in enumerate(class_vars) if i != position)
        domain = Domain(ordered.domain.attributes, var, ordered.domain.metas + others)
        table = ordered.transform(domain)
        table.name = ordered.name
        tables.append(table)
    return tables


def target_indices(plan, table):
    """Índices de entrenamiento y test por iteración, sin las filas sin valor en el objetivo."""
    y = np.asarray(table.Y, dtype=float).ravel()
    valid = np.isfinite(y)
    out = []
    for it in plan.iterations:
        train = np.arange(it.train_lo, it.train_hi)
        test = np.arange(it.test_lo, it.test_hi)
        out.append((train[valid[train]], test[valid[test]]))
    return out


def target_notes(plan, table):
    """Avisos de un objetivo: valores perdidos, clases o valores degenerados en las ventanas."""
    var = table.domain.class_var
    y = np.asarray(table.Y, dtype=float).ravel()
    valid = np.isfinite(y)
    notes = []
    used = np.zeros(len(y), dtype=bool)
    for it in plan.iterations:
        used[it.train_lo:it.train_hi] = True
        used[it.test_lo:it.test_hi] = True
    missing = int((used & ~valid).sum())
    if missing:
        notes.append(f"«{var.name}»: {missing} filas sin valor quedan fuera del entrenamiento y del test.")
    empty = {it.index for it in plan.iterations
             if not valid[it.test_lo:it.test_hi].any() or not valid[it.train_lo:it.train_hi].any()}
    if empty:
        return notes, empty  # quien llama omite esas iteraciones y vuelve a revisar
    single, unseen, constant = [], [], []
    if var.is_discrete:
        n_classes = len(var.values)
        codes = np.where(valid, np.nan_to_num(y), -1).astype(int)
        cumulative = np.zeros((len(y) + 1, n_classes), dtype=np.int64)
        for value in range(n_classes):
            cumulative[1:, value] = np.cumsum(codes == value)
        for it in plan.iterations:
            train = cumulative[it.train_hi] - cumulative[it.train_lo]
            test = cumulative[it.test_hi] - cumulative[it.test_lo]
            if np.count_nonzero(train) == 1:
                single.append(it.index)
            if np.any((test > 0) & (train == 0)):
                unseen.append(it.index)
    else:
        compressed = np.flatnonzero(valid)
        values = y[compressed]
        changes = np.r_[0, np.cumsum(values[1:] != values[:-1])]
        for it in plan.iterations:
            lo = np.searchsorted(compressed, it.train_lo)
            hi = np.searchsorted(compressed, it.train_hi)
            if hi - lo >= 1 and changes[hi - 1] == changes[lo]:
                constant.append(it.index)
    if single:
        notes.append(f"«{var.name}»: el entrenamiento de {_iterations_text(single)} tiene una sola "
                     "clase; algunos modelos (como la regresión logística) fallarán.")
    if unseen:
        notes.append(f"«{var.name}»: el test de {_iterations_text(unseen)} tiene clases que no "
                     "aparecen en su entrenamiento.")
    if constant:
        notes.append(f"«{var.name}» es constante en el entrenamiento de {_iterations_text(constant)}.")
    return notes, set()


def omit_iterations(plan, empty):
    """Quita las iteraciones sin entrenamiento o sin test para algún objetivo y lo avisa.

    `empty` asocia el índice de cada iteración con los objetivos que se quedan vacíos. Las
    demás se renumeran: cada una es una partición de Results, que no admite huecos.
    """
    if not empty:
        return
    shown = []
    for index in sorted(empty)[:4]:
        it = plan.iterations[index]
        targets = ", ".join(f"«{name}»" for name in empty[index])
        shown.append(f"la {index + 1} (test {plan.span(it.test_lo, it.test_hi)}; {targets})")
    more = "…" if len(empty) > 4 else ""
    count = len(empty)
    plan.notes.append(
        f"Se omite{'n' if count > 1 else ''} {count} iteraci{'ones' if count > 1 else 'ón'} sin valores de "
        f"algún objetivo en su entrenamiento o su test: {'; '.join(shown)}{more}.")
    kept = [it for it in plan.iterations if it.index not in empty]
    for position, it in enumerate(kept):
        it.index = position
    plan.iterations = kept


def drop_missing_targets(data):
    """Quita las filas sin valor en algún objetivo. Devuelve la tabla y cuántas faltan por objetivo."""
    Y = np.asarray(data.Y, dtype=float).reshape(len(data), -1)
    missing = ~np.isfinite(Y)
    rows = missing.any(axis=1)
    if not rows.any():
        return data, {}
    counts = {var.name: int(missing[:, i].sum()) for i, var in enumerate(data.domain.class_vars)
              if missing[:, i].any()}
    return data[~rows], counts


def dropped_message(n_rows, counts):
    """Texto del aviso de filas descartadas por no tener valor en algún objetivo."""
    rows = f"{n_rows} fila{'s' if n_rows != 1 else ''}"
    if len(counts) == 1 and n_rows == next(iter(counts.values())):
        name = next(iter(counts))
        return f"Se han descartado {rows} sin valor en la variable objetivo «{name}»."
    detail = "; ".join(f"{name}: {count}" for name, count in counts.items())
    return f"Se han descartado {rows} sin valor en algún objetivo ({detail})."


def _iterations_text(indices):
    shown = ", ".join(str(i + 1) for i in indices[:6]) + ("…" if len(indices) > 6 else "")
    return f"la iteración {shown}" if len(indices) == 1 else f"{len(indices)} iteraciones ({shown})"


def check_targets(data):
    """Error si las variables objetivo no se pueden evaluar juntas."""
    class_vars = data.domain.class_vars
    if len(class_vars) > 1 and len({var.is_discrete for var in class_vars}) > 1:
        return ("Hay objetivos numéricos y categóricos mezclados: la evaluación multi-horizonte "
                "compara la misma métrica en todos. Sepáralos con Select Columns.")
    Y = np.asarray(data.Y, dtype=float).reshape(len(data), -1)
    for position, var in enumerate(class_vars):
        values = Y[:, position]
        if not np.isfinite(values).any():
            return f"La variable objetivo «{var.name}» no tiene ningún valor."
        if var.is_discrete and len(np.unique(values[np.isfinite(values)])) < 2:
            return f"La variable objetivo «{var.name}» solo tiene un valor."
    return ""


# ---------------------------------------------------------------- evaluación ----

class WindowValidation(Validation):
    """Validación con las ventanas ya calculadas; cada iteración es una partición.

    Es la misma mecánica que Orange.evaluation.Validation (el preprocesador se ajusta solo
    con el entrenamiento de cada iteración), con dos añadidos: el error de un modelo indica
    en qué iteración ha fallado y, tras fallar, ese modelo ya no se entrena más.
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        # Validation.__new__ interpreta los argumentos posicionales como datos y modelos.
        return object.__new__(cls)

    def __init__(self, indices, labels, store_data=True):
        super().__init__(store_data=store_data, store_models=False)
        self.indices = indices
        self.labels = labels

    def get_indices(self, data):
        return self.indices

    def __call__(self, data, learners, preprocessor=None, *, callback=None, suppresses_exceptions=True):
        preprocessor = preprocessor or (lambda table: table)
        callback = callback or (lambda fraction: None)
        indices = self.get_indices(data)
        folds, row_indices, actual = self.prepare_arrays(data, indices)
        failed = {}
        parts = []
        total = max(len(indices) * len(learners), 1)
        done = 0
        for fold_i, (train_i, test_i) in enumerate(indices):
            pending = [i for i in range(len(learners)) if i not in failed]
            train_data = preprocessor(data[train_i]) if pending else None
            test_data = data[test_i]
            for learner_i, learner in enumerate(learners):
                if learner_i in failed:
                    parts.append(_failed_part(fold_i, learner_i, failed[learner_i], len(test_i)))
                else:
                    part = _mp_worker(fold_i, train_data, test_data, learner_i, learner, False)
                    if part.failed:
                        failed[learner_i] = IterationError(fold_i, self.labels[fold_i], part.failed)
                        part = part._replace(failed=failed[learner_i])
                    parts.append(part)
                done += 1
                callback(done / total)
        results = Results(data=data if self.store_data else None, domain=data.domain,
                          nrows=len(row_indices), learners=learners, row_indices=row_indices,
                          folds=folds, actual=actual, score_by_folds=self.score_by_folds,
                          train_time=np.zeros((len(learners),)), test_time=np.zeros((len(learners),)))
        self._collect_part_results(results, parts)
        return results


def _failed_part(fold_i, learner_i, error, n_values):
    from Orange.evaluation.testing import _MpResults
    return _MpResults(fold_i, learner_i, None, error, n_values, None, None, 0.0, 0.0)


def run_window(plan, data, learners, preprocessor=None, callback=None):
    """Evalúa `learners` en cada objetivo. Devuelve, por objetivo, Results o una excepción."""
    callback = callback or (lambda fraction: None)
    tables = target_tables(data, plan)
    labels = [plan.span(it.test_lo, it.test_hi) for it in plan.iterations]
    out = []
    for position, table in enumerate(tables):
        def partial(fraction, position=position):
            callback((position + fraction) / len(tables))

        indices = target_indices(plan, table)
        try:
            out.append(WindowValidation(indices, labels)(table, learners, preprocessor, callback=partial))
        except Exception as error:  # pylint: disable=broad-except
            if type(error).__name__ == "Cancelled":
                raise
            out.append(error)
    return out


def training_rows(plan):
    return sum(it.n_train for it in plan.iterations)


# ---------------------------------------------------- pérdidas para Diebold-Mariano ----

SQUARED, ABSOLUTE, PERCENTAGE, ZERO_ONE, LOG = range(5)
LOSS_OF_METRIC = {"MAE": ABSOLUTE, "MAPE": PERCENTAGE, "LogLoss": LOG}
LOSS_EVIDENCE = {
    SQUARED: "error cuadrático por fila", ABSOLUTE: "error absoluto por fila",
    PERCENTAGE: "error porcentual absoluto por fila", ZERO_ONE: "aciertos por fila (pérdida 0/1)",
    LOG: "pérdida logarítmica por fila",
}
LOSS_UNITS = {
    SQUARED: "error cuadrático medio", ABSOLUTE: "error absoluto medio",
    PERCENTAGE: "error porcentual absoluto medio", ZERO_ONE: "tasa de error",
    LOG: "pérdida logarítmica media",
}
# Métricas sobre las que aporta evidencia cada pérdida: con el mismo test, MSE, RMSE, R² y
# CVRMSE ordenan igual a los modelos.
LOSS_METRICS = {
    SQUARED: frozenset(("MSE", "RMSE", "R2", "CVRMSE")), ABSOLUTE: frozenset(("MAE",)),
    PERCENTAGE: frozenset(("MAPE",)), ZERO_ONE: frozenset(("CA",)), LOG: frozenset(("LogLoss",)),
}


def loss_kind(metric, classification):
    if classification:
        return LOG if metric == "LogLoss" else ZERO_ONE
    return LOSS_OF_METRIC.get(metric, SQUARED) if metric != "LogLoss" else SQUARED


def latest_positions(results):
    """Posiciones de Results con una sola predicción por fila (la de la iteración más
    reciente), en orden temporal."""
    seen = set()
    keep = []
    for fold in reversed(results.folds):
        for position in range(fold.start, fold.stop)[::-1]:
            row = int(results.row_indices[position])
            if row not in seen:
                seen.add(row)
                keep.append(position)
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(results.row_indices[keep], kind="stable")]


def row_losses(results, kind, tolerance):
    """Pérdida por fila de cada modelo (filas en orden temporal) y su suelo de redondeo."""
    positions = latest_positions(results)
    actual = np.asarray(results.actual, dtype=float)[positions]
    predicted = np.asarray(results.predicted, dtype=float)[:, positions]
    scale = float(np.nanmax(np.abs(actual))) if actual.size else 0.0
    delta = tolerance * scale
    if kind == ZERO_ONE:
        return (predicted != actual).astype(float), 0.0
    if kind == LOG:
        probabilities = results.probabilities[:, positions]
        chosen = np.take_along_axis(probabilities, actual.astype(int)[None, :, None], axis=2)[:, :, 0]
        return -np.log(np.clip(chosen, np.finfo(float).eps, 1)), tolerance
    errors = np.abs(predicted - actual)
    if kind == ABSOLUTE:
        return errors, delta
    if kind == PERCENTAGE:
        with np.errstate(divide="ignore", invalid="ignore"):
            losses = np.where(actual != 0, errors / np.abs(actual), np.nan)
        return losses, tolerance
    return errors ** 2, 2 * delta * errors.max(axis=0) + delta ** 2
