# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.
"""Motor de Nomogram Pro (sin Qt).

Un nomograma muestra un modelo como una suma de un efecto por atributo. Todo se
expresa en columnas *raw*, una por clase (o una sola en regresión):

* modo exacto, para los modelos aditivos por construcción
  - modelos lineales (``skl_model.coef_``): raw = función de decisión o predicción,
  - Naive Bayes: raw = log-posterior sin normalizar de cada clase;
  para una clase objetivo ``c`` la puntuación es ``raw_c - media(raw_k, k != c)``
  (``contrast``), que con dos clases es exactamente el log-odds;
* modo aproximado, para cualquier otro modelo: dependencia parcial (o un ajuste
  aditivo directo por mínimos cuadrados) de la probabilidad, del log-odds de cada
  clase o del valor predicho; la puntuación de ``c`` es su propia columna.

La puntuación de una instancia es ``base + suma_a c_a(x_a)``, con cada componente
centrada en la muestra de referencia, así que ``base`` es la puntuación media. Un
*link* convierte la puntuación en lo que predice el modelo (probabilidad o valor).
La fidelidad de la suma aditiva frente al modelo real se mide siempre en la muestra
de referencia; en el modo exacto es una verificación numérica.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from Orange.data import Domain, Table, TimeVariable
from Orange.preprocess.transformation import Transformation
from Orange.preprocess.discretize import Discretizer

try:
    from Orange.preprocess.impute import ReplaceUnknownsRandom
except ImportError:  # pragma: no cover
    ReplaceUnknownsRandom = ()

EXACT, APPROXIMATE = "exact", "approximate"
#: métodos de aproximación
METHOD_AUTO, PARTIAL_DEPENDENCE, ADDITIVE_FIT = 0, 1, 2
#: escala aditiva de la aproximación en clasificación
SPACE_AUTO, SPACE_PROB, SPACE_LOGIT = 0, 1, 2

LOGIT_EPS = 1e-3          # recorte de las probabilidades en log-odds
EXACT_TOL = 1e-7          # tolerancia relativa de la verificación exacta
FINE_GRID = 101           # puntos de un efecto continuo exacto no lineal
MAX_ROWS_PER_CALL = 100_000
MIN_PD_ROWS = 50
# escala de probabilidad aproximada con varias clases: error medio, percentil 95
# y máximo admitidos en los datos de referencia
MULTI_MAE, MULTI_P95, MULTI_MAX = 0.05, 0.10, 0.25
MIN_REFERENCE_ROWS = 20   # filas mínimas de unos datos de referencia fiables
PD_GRID_POINTS = 2000     # puntos de rejilla de todos los numéricos juntos
MAX_RARE_ROWS = 0.2       # filas añadidas a la muestra por categorías raras
MAX_FIT_COLUMNS = 1200    # columnas de la base del ajuste aditivo


class NomogramError(Exception):
    """Modelo o datos que el nomograma no puede tratar (mensaje para el usuario)."""


class Cancelled(Exception):
    pass


def logit(p, eps=LOGIT_EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1 + np.tanh(0.5 * z))


# ---------------------------------------------------------------------------
# Efectos de los atributos
# ---------------------------------------------------------------------------

@dataclass
class FeatureEffect:
    var: object                 # variable original de Orange
    index: int                  # columna en el dominio de atributos
    kind: str                   # discrete | linear | bins | curve
    grid: np.ndarray            # valores en los que se conoce ``raw``
    raw: np.ndarray             # (len(grid), K)
    raw_nan: Optional[np.ndarray] = None    # (K,) efecto de un valor desconocido
    lo: float = 0.0             # rango que se dibuja en los numéricos
    hi: float = 0.0
    slope: Optional[np.ndarray] = None      # lineal: raw = offset + slope * v
    offset: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None  # tramos: puntos de corte
    labels: Optional[List[str]] = None       # tramos y categóricas: nombres
    mean: Optional[np.ndarray] = None        # (K,) media en la muestra
    cov: Optional[np.ndarray] = None         # (K, K) covarianza en la muestra
    counts: Optional[np.ndarray] = None      # distribución de los datos
    edges: Optional[np.ndarray] = None       # bordes del histograma (numéricos)
    n_unknown: int = 0
    typical: float = float("nan")            # mediana o valor más frecuente
    unsupported: Optional[np.ndarray] = None  # categorías sin datos (ajuste)

    @property
    def name(self):
        return self.var.name

    @property
    def is_discrete(self):
        return self.kind in ("discrete", "bins")

    def evaluate(self, values):
        """Efecto raw (n, K) de los valores dados (NaN = desconocido)."""
        values = np.asarray(values, dtype=float).ravel()
        K = self.raw.shape[1]
        out = np.empty((len(values), K))
        nan = np.isnan(values)
        known = values[~nan]
        if self.kind == "discrete":
            idx = np.clip(known.astype(int), 0, len(self.grid) - 1)
            res = self.raw[idx]
            if self.unsupported is not None and self.unsupported[idx].any():
                # sin datos en la muestra: no hay efecto estimado, se toma la media
                res = res.copy()
                res[self.unsupported[idx]] = self.mean if self.mean is not None \
                    else 0
        elif self.kind == "linear":
            res = self.offset[None, :] + known[:, None] * self.slope[None, :]
        elif self.kind == "bins":
            res = self.raw[self.bin_of(known)]
        else:
            res = np.column_stack([
                np.interp(known, self.grid, self.raw[:, k])
                for k in range(K)]) if len(known) else np.empty((0, K))
        out[~nan] = res
        if nan.any():
            out[nan] = self.raw_nan if self.raw_nan is not None \
                else (self.mean if self.mean is not None else 0)
        return out

    def bin_of(self, values):
        return np.searchsorted(self.thresholds, values, side="right")

    def position_value(self, value):
        """Índice en ``grid`` de un valor categórico o por tramos (None si es NaN)."""
        if value is None or np.isnan(value):
            return None
        if self.kind == "bins":
            return int(self.bin_of(np.array([value]))[0])
        return int(value)


@dataclass
class Variant:
    """Una descomposición aditiva (la exacta o una de las aproximadas)."""
    space: str                  # score | prob | logit | value
    derive: str                 # contrast | direct
    effects: List[FeatureEffect]
    base_raw: np.ndarray        # (K,) puntuación raw media en la muestra
    recon: Optional[np.ndarray]  # (n, K) puntuación aditiva de la muestra
    holdout: Optional[np.ndarray] = None   # filas con las que se mide la fidelidad
    max_error: Optional[float] = None      # verificación exacta
    method: str = "exact"                  # exact | pd | fit
    note: str = ""


@dataclass
class Analysis:
    model: object
    kind: str                   # classification | regression
    mode: str                   # exact | approximate
    domain: Domain              # dominio de atributos del modelo
    class_var: object
    variants: dict              # (method, space) -> Variant
    method: int = METHOD_AUTO
    model_name: str = ""
    reason: str = ""            # por qué el modelo se aproxima
    sample: Optional[np.ndarray] = None     # (n, p) filas de referencia
    real: Optional[np.ndarray] = None       # (n, K) probabilidades o valores
    real_score: Optional[np.ndarray] = None  # exacto: (n, K) raw del modelo
    present: Optional[np.ndarray] = None    # clases que conoce el modelo
    n_reference: int = 0
    n_pd: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    residuals: Optional[np.ndarray] = None  # regresión: y - predicción
    residuals_on_training: bool = False
    has_nan_effect: bool = True

    @property
    def n_classes(self):
        return len(self.class_var.values) if self.kind == "classification" else 1

    def variant_for(self, target, space=SPACE_AUTO):
        """La variante que se muestra: la más fiel entre las permitidas."""
        cands = list(self.variants.values())
        if self.mode == APPROXIMATE and self.kind == "classification":
            if space == SPACE_PROB:
                cands = [v for v in cands if v.space == "prob"]
            elif space == SPACE_LOGIT:
                cands = [v for v in cands if v.space == "logit"]
        if len(cands) == 1:
            return cands[0]
        best, best_err = cands[0], np.inf
        for var in cands:
            err = TargetView(self, var, target).fidelity.get("mae", np.inf)
            # a igualdad (menos de un 5 % de diferencia), la dependencia parcial,
            # que se lee como «qué pasa al cambiar este valor»
            if var.method == "fit":
                err *= 1.05
            if err < best_err * (1 - 1e-9):
                best, best_err = var, err
        return best

    def target_view(self, target, space=SPACE_AUTO):
        return TargetView(self, self.variant_for(target, space), target)


# ---------------------------------------------------------------------------
# Links: puntuación -> predicción
# ---------------------------------------------------------------------------

@dataclass
class Link:
    kind: str  # identity | logistic | clip | monotone | decision | score | none
    exact: bool = True
    a: float = 1.0
    b: float = 0.0
    xs: Optional[np.ndarray] = None
    ys: Optional[np.ndarray] = None
    threshold: float = 0.0

    @property
    def has_probability(self):
        return self.kind in ("logistic", "clip", "monotone")

    def __call__(self, score):
        s = np.asarray(score, dtype=float)
        if self.kind == "logistic":
            return sigmoid(self.a * s + self.b)
        if self.kind == "clip":
            return np.clip(s, 0, 1)
        if self.kind == "monotone":
            return np.interp(s, self.xs, self.ys)
        return s

    def inverse(self, y):
        """Puntuación que da la predicción ``y`` (NaN si no se alcanza)."""
        y = np.asarray(y, dtype=float)
        if self.kind == "logistic":
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log(y / (1 - y))
            return (z - self.b) / self.a
        if self.kind == "clip":
            return y
        if self.kind == "monotone":
            ys, idx = np.unique(self.ys, return_index=True)
            if len(ys) < 2:
                return np.full_like(y, np.nan)
            res = np.interp(y, ys, self.xs[idx], left=np.nan, right=np.nan)
            return res
        return y


def isotonic(x, y):
    """Regresión isotónica creciente (pool adjacent violators)."""
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    vals, weights, sizes = [], [], []
    for v in y:
        vals.append(v)
        weights.append(1.0)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = weights[-2] + weights[-1]
            v = (vals[-2] * weights[-2] + vals[-1] * weights[-1]) / w
            s = sizes[-2] + sizes[-1]
            vals[-2:] = [v]
            weights[-2:] = [w]
            sizes[-2:] = [s]
    fitted = np.repeat(vals, sizes)
    # un punto por cada x distinta
    ux, start = np.unique(x, return_index=True)
    uy = np.array([fitted[i] for i in start])
    return ux, uy


def fit_link(kind, space, scores, probs, exact):
    """Link de una clase a partir de las puntuaciones y probabilidades reales."""
    if kind == "regression":
        return Link("identity", exact=True)
    if space == "prob":
        return Link("clip", exact=True)
    if space == "logit":
        return Link("logistic", exact=True)
    ok = np.isfinite(scores) & np.isfinite(probs)
    scores, probs = scores[ok], probs[ok]
    if len(scores) == 0:
        return Link("none", exact=False)
    if np.all((probs < 1e-9) | (probs > 1 - 1e-9)):
        return Link("decision", exact=exact)
    inner = (probs > 1e-6) & (probs < 1 - 1e-6)
    if exact and inner.sum() >= 2 and np.ptp(scores[inner]) > 0:
        z = np.log(probs[inner] / (1 - probs[inner]))
        A = np.column_stack([scores[inner], np.ones(inner.sum())])
        (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = np.abs(A @ np.array([a, b]) - z)
        if a > 0 and resid.max() <= 1e-6 * max(1.0, np.abs(z).max()):
            return Link("logistic", exact=True, a=float(a), b=float(b))
    xs, ys = isotonic(scores, probs)
    if len(xs) < 2:
        return Link("none", exact=False)
    return Link("monotone", exact=False, xs=xs, ys=ys)


# ---------------------------------------------------------------------------
# Vista de una clase objetivo
# ---------------------------------------------------------------------------

class TargetView:
    """Puntuación aditiva de una clase objetivo (o del valor en regresión)."""

    def __init__(self, analysis, variant, target=0):
        self.analysis = analysis
        self.variant = variant
        K = variant.base_raw.shape[0]
        self.target = target if analysis.kind == "classification" else 0
        if variant.derive == "contrast" and K > 1:
            present = analysis.present if analysis.present is not None \
                else np.ones(K, bool)
            others = present.copy()
            others[self.target] = False
            d = np.zeros(K)
            d[self.target] = 1
            if others.any():
                d[others] = -1 / others.sum()
            self.weights = d
            self.mask = present
        else:
            d = np.zeros(K)
            d[self.target] = 1
            self.weights = d
            self.mask = d != 0
        self.base = self._dot(variant.base_raw)
        self.link = self._make_link()
        self.fidelity = self._fidelity()

    def _dot(self, raw):
        """Puntuación a partir de las columnas raw de las clases que cuentan.

        Un NaN (categoría sin datos en el ajuste aditivo) se conserva: no tiene
        efecto estimado y no se dibuja.
        """
        raw = np.asarray(raw, dtype=float)
        sub = raw[..., self.mask]
        return sub @ self.weights[self.mask]

    @property
    def effects(self):
        return self.variant.effects

    @property
    def exact(self):
        return self.analysis.mode == EXACT

    def component(self, effect, values):
        """Componente centrada de ``effect`` en ``values``."""
        return self._dot(effect.evaluate(values) - effect.mean)

    def grid_component(self, effect):
        return self._dot(effect.raw - effect.mean)

    def nan_component(self, effect):
        if effect.raw_nan is None:
            return None
        return float(self._dot(effect.raw_nan - effect.mean))

    def importance(self, effect):
        """Desviación típica de la componente en la muestra de referencia."""
        if effect.cov is None:
            return 0.0
        w = self.weights[self.mask]
        c = effect.cov[np.ix_(self.mask, self.mask)]
        return float(math.sqrt(max(0.0, w @ c @ w)))

    def score(self, values):
        """Puntuación aditiva de una instancia (un valor por atributo)."""
        return self.base + sum(
            float(self.component(e, [values[e.index]])[0])
            for e in self.effects)

    def _make_link(self):
        a = self.analysis
        var = self.variant
        if a.kind == "regression":
            return Link("identity")
        scores = self._dot(var.recon) if var.recon is not None else None
        probs = a.real[:, self.target] if a.real is not None else None
        if scores is None or probs is None:
            return Link("none", exact=False)
        if var.space in ("prob", "logit"):
            return fit_link(a.kind, var.space, scores, probs, False)
        # puntuaciones de contraste exactas: el link solo es exacto con dos clases
        link = fit_link(a.kind, var.space, scores, probs,
                        exact=self.mask.sum() <= 2)
        if link.kind == "monotone" and self.mask.sum() > 2:
            # con varias clases la probabilidad depende de todas: la escala
            # monótona solo se muestra si es buena en casi todas las filas de
            # referencia, no solo de media
            err = np.abs(link(scores) - probs)
            err = err[np.isfinite(err)]
            if not len(err) or err.mean() > MULTI_MAE \
                    or np.percentile(err, 95) > MULTI_P95 or err.max() > MULTI_MAX:
                return Link("score", exact=True)
        return link

    def _fidelity(self):
        a = self.analysis
        var = self.variant
        if var.recon is None or a.real is None:
            return {}
        rows = var.holdout if var.holdout is not None \
            else np.arange(len(var.recon))
        if len(rows) == 0:
            return {}
        fid = {"rows": int(len(rows))}
        scores = self._dot(var.recon[rows])
        if a.kind == "regression":
            real = a.real[rows, 0]
            pred = scores
        else:
            real = a.real[rows, self.target]
            pred = self.link(scores) if self.link.has_probability else None
        if a.mode == EXACT and var.max_error is not None:
            fid["max_error"] = var.max_error
        if pred is not None:
            err = pred - real
            fid["mae"] = float(np.mean(np.abs(err)))
            fid["max_abs"] = float(np.max(np.abs(err)))
            fid["p95"] = float(np.percentile(np.abs(err), 95))
            size = max(1.0, float(np.max(np.abs(real))))
            var = float(np.mean((real - real.mean()) ** 2))
            if var <= (1e-12 * size) ** 2:
                # predicción constante: R² solo se define si se reproduce
                fid["r2"] = 1.0 if fid["max_abs"] <= 1e-9 * size \
                    else float("nan")
            else:
                fid["r2"] = float(1 - np.mean(err ** 2) / var)
            if a.kind == "regression":
                fid["scale"] = float(np.std(real)) or 1.0
        if a.kind == "classification":
            fid["agreement"] = self._agreement(rows)
        return fid

    def _agreement(self, rows):
        """Proporción de filas en las que la suma aditiva predice la misma clase."""
        a = self.analysis
        var = self.variant
        real_cls = np.argmax(a.real[rows], axis=1)
        if a.real.shape[1] == 2 and self.link.has_probability:
            p = self.link(self._dot(var.recon[rows]))
            add = p >= 0.5
            real = a.real[rows, self.target] >= 0.5
            return float(np.mean(add == real))
        if var.derive == "contrast":
            raw = var.recon[rows]
            raw = np.where(a.present[None, :] if a.present is not None
                           else True, raw, -np.inf)
            add_cls = np.argmax(np.nan_to_num(raw, nan=-np.inf), axis=1)
        elif var.space == "prob":
            add_cls = np.argmax(var.recon[rows], axis=1)
        else:
            add_cls = np.argmax(var.recon[rows], axis=1)
        return float(np.mean(real_cls == add_cls))


# ---------------------------------------------------------------------------
# Utilidades sobre los datos
# ---------------------------------------------------------------------------

def _original_domain(model):
    domain = getattr(model, "original_domain", None)
    return model.domain if domain is None else domain


def attribute_domain(model):
    return Domain(_original_domain(model).attributes)


def model_name(model):
    name = getattr(model, "name", None) or type(model).__name__
    return str(name)


def model_kind(model):
    domain = _original_domain(model)
    cv = domain.class_var
    if cv is None or len(domain.class_vars) != 1:
        raise NomogramError(
            "El modelo debe tener una sola variable objetivo "
            "(clasificación o regresión).")
    if not domain.attributes:
        raise NomogramError("El modelo no tiene atributos.")
    if cv.is_discrete:
        if len(cv.values) < 2:
            raise NomogramError("La variable objetivo tiene un solo valor.")
        return "classification", cv
    return "regression", cv


def _dense(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def reference_table(model, data):
    """Filas de referencia en el dominio de atributos del modelo y notas.

    ``data`` (la entrada opcional) tiene prioridad sobre los datos de
    entrenamiento que guarda el modelo. Devuelve ``(tabla, origen, notas)``;
    la tabla puede ser None.
    """
    notes = []
    adom = attribute_domain(model)
    if data is not None and len(data):
        names = {v.name for v in data.domain.variables + data.domain.metas}
        missing = [a.name for a in adom.attributes if a.name not in names]
        if not missing:
            try:
                return data.transform(adom), "data", notes
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"No se pudieron usar los datos de entrada: {exc}")
        else:
            notes.append(
                "Los datos de entrada no tienen "
                + _plural(len(missing), "el atributo", "los atributos")
                + " " + _names(missing)
                + " del modelo: se usan los datos de entrenamiento.")
    od = getattr(model, "original_data", None)
    if od is not None and len(od):
        try:
            return od.transform(adom), "training", notes
        except Exception:  # pragma: no cover - defensive
            pass
    return None, None, notes


def training_columns(model, adom, limit=50_000):
    """Columnas densas de los datos de entrenamiento (muestra si son muchas)."""
    od = getattr(model, "original_data", None)
    if od is None or not len(od):
        return None
    try:
        t = od.transform(adom)
    except Exception:  # pragma: no cover - defensive
        return None
    X, _ = sample_rows(t, limit, seed=2)
    return X


def _usable(col, var):
    known = col[~np.isnan(col)]
    if not len(known):
        return False
    return var.is_discrete or float(np.ptp(known)) > 0


def range_columns(analysis, model, columns, source):
    """Columna de cada atributo para rangos, rejillas y distribución.

    Es la de los datos de referencia salvo si no tiene valores conocidos o, en
    un numérico, es constante: entonces la de entrenamiento si la hay.
    """
    adom = analysis.domain
    cols = [columns[:, a] for a in range(len(adom.attributes))]
    train = None
    used, empty, const = [], [], []
    for a, var in enumerate(adom.attributes):
        if _usable(cols[a], var):
            continue
        if source == "data":
            if train is None:
                train = training_columns(model, adom)
                if train is None:
                    train = False
            if train is not False and _usable(train[:, a], var):
                cols[a] = train[:, a]
                used.append(var.name)
                continue
        if np.isnan(cols[a]).all():
            empty.append(var.name)
        else:
            const.append(var.name)
    if used:
        analysis.notes.append(
            "Sin valores conocidos o constantes en Data: "
            + _names(used) + "; su escala usa el rango de los datos de "
            "entrenamiento.")
    if empty:
        analysis.warnings.append(
            _plural(len(empty), "El atributo ", "Los atributos ")
            + _names(empty) + _plural(len(empty), " no tiene", " no tienen")
            + " valores conocidos en los datos de referencia: su escala se reduce "
            "a un punto y no muestra su efecto.")
    if const:
        analysis.warnings.append(
            _plural(len(const), "El atributo ", "Los atributos ")
            + _names(const) + _plural(len(const), " es constante", " son constantes")
            + " en los datos de referencia: su escala se reduce a un punto.")
    return cols


def _plural(n, one, many):
    return one if n == 1 else many


def _names(names, limit=5):
    shown = ", ".join(f"«{n}»" for n in names[:limit])
    if len(names) > limit:
        shown += f" y {len(names) - limit} más"
    return shown


def sample_rows(table, size, seed=0, rare=False):
    """Muestra al azar de ``size`` filas (densa) y sus índices.

    Con ``rare``, añade una fila por cada categoría presente en la tabla que no
    haya salido en la muestra (hasta un 20 % más de filas), para que toda
    categoría con datos tenga un efecto estimado.
    """
    X = _dense(table.X)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    if n <= size:
        return X, np.arange(n)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(n, size, replace=False))
    if rare:
        chosen = set(idx.tolist())
        extra = []
        limit = int(MAX_RARE_ROWS * size)
        for a, var in enumerate(table.domain.attributes):
            if not var.is_discrete or len(extra) >= limit:
                continue
            col = X[:, a]
            seen = set(np.unique(col[idx][~np.isnan(col[idx])]).astype(int))
            for v in np.unique(col[~np.isnan(col)]).astype(int):
                if v in seen or len(extra) >= limit:
                    continue
                rows = np.flatnonzero(col == v)
                pick = int(rows[rng.randint(len(rows))])
                if pick not in chosen:
                    chosen.add(pick)
                    extra.append(pick)
        if extra:
            idx = np.sort(np.concatenate([idx, extra]))
    return X[idx], idx


def _make_table(adom, X):
    return Table.from_numpy(adom, np.asarray(X, dtype=float))


class Predictor:
    """Llama al modelo real por bloques, con progreso y cancelación."""

    def __init__(self, model, adom, kind, state=None):
        self.model = model
        self.adom = adom
        self.kind = kind
        self.state = state
        self.rows = 0

    def __call__(self, X):
        out = []
        for start in range(0, len(X), MAX_ROWS_PER_CALL):
            if self.state is not None and self.state.is_interruption_requested():
                raise Cancelled
            chunk = X[start:start + MAX_ROWS_PER_CALL]
            t = _make_table(self.adom, chunk)
            if self.kind == "classification":
                p = np.asarray(self.model(t, self.model.Probs), dtype=float)
                if p.ndim == 1:  # pragma: no cover - defensive
                    p = p[:, None]
            else:
                p = np.asarray(self.model(t), dtype=float).reshape(-1, 1)
            out.append(p)
            self.rows += len(chunk)
        if not out:
            k = 1 if self.kind == "regression" else None
            return np.empty((0, k or 0))
        return np.vstack(out)


def predict_instance(model, adom, values):
    """Predicción real del modelo para una instancia (lista de valores)."""
    kind, cv = model_kind(model)
    pred = Predictor(model, adom, kind)
    res = pred(np.asarray(values, dtype=float)[None, :])[0]
    return res if kind == "classification" else float(res[0])


def ensemble_spread(model, adom, values, target=None, q=(5, 95)):
    """Dispersión de los árboles de un bosque aleatorio para una instancia.

    Devuelve ``(bajo, alto, n_árboles)`` de la probabilidad objetivo o del
    valor, o None si el modelo no es un bosque por bagging.
    """
    skl = getattr(model, "skl_model", None)
    trees = getattr(skl, "estimators_", None)
    if skl is None or trees is None:
        return None
    name = type(skl).__name__
    if not ("Forest" in name or "ExtraTrees" in name or "Bagging" in name):
        return None
    try:
        t = _make_table(adom, np.asarray(values, dtype=float)[None, :])
        X = t.transform(model.domain).X
        X = _dense(X)
        if np.isnan(X).any():
            return None
        trees = list(np.ravel(trees))
        if target is None:
            vals = np.array([tr.predict(X)[0] for tr in trees])
        else:
            classes = list(np.asarray(skl.classes_).astype(int))
            if target not in classes:
                return None
            col = classes.index(target)
            vals = np.array([tr.predict_proba(X)[0][col] for tr in trees])
    except Exception:
        return None
    lo, hi = np.percentile(vals, q)
    return float(lo), float(hi), len(trees)


# ---------------------------------------------------------------------------
# Distribución de los valores de referencia (histograma)
# ---------------------------------------------------------------------------

def _distribution(effect, column):
    col = column[~np.isnan(column)]
    effect.n_unknown = int(np.isnan(column).sum())
    if len(col):
        if effect.var.is_discrete:
            effect.typical = float(np.bincount(col.astype(int)).argmax())
        else:
            effect.typical = float(np.median(col))
    if effect.kind == "discrete":
        effect.counts = np.bincount(col.astype(int),
                                    minlength=len(effect.grid))[:len(effect.grid)]
    elif effect.kind == "bins":
        effect.counts = np.bincount(effect.bin_of(col),
                                    minlength=len(effect.grid))[:len(effect.grid)]
    elif len(col):
        lo, hi = effect.lo, effect.hi
        if hi > lo:
            effect.counts, effect.edges = np.histogram(
                col, bins=min(30, max(5, len(np.unique(col)))), range=(lo, hi))
        else:
            effect.counts, effect.edges = np.array([len(col)]), \
                np.array([lo, hi])


def _centre(effect, sample_raw):
    """Media y covarianza del efecto raw en la muestra."""
    if sample_raw is None or not len(sample_raw):
        # sin muestra, se centra en la media de la rejilla
        effect.mean = np.nanmean(effect.raw, axis=0)
        effect.cov = None
        return
    sr = np.nan_to_num(sample_raw)
    effect.mean = sr.mean(axis=0)
    if len(sr) > 1:
        effect.cov = np.atleast_2d(np.cov(sr, rowvar=False, bias=True))
    else:
        effect.cov = np.zeros((sr.shape[1], sr.shape[1]))
    nanm = np.isnan(sample_raw).all(axis=0)
    effect.mean[nanm] = np.nan


def _continuous_range(column, var):
    col = column[~np.isnan(column)]
    if not len(col):
        return None
    return float(col.min()), float(col.max())


def continuous_grid(column, size):
    col = np.unique(column[~np.isnan(column)])
    if len(col) <= size:
        return col.astype(float)
    q = np.quantile(column[~np.isnan(column)], np.linspace(0, 1, size))
    return np.unique(q)


# ---------------------------------------------------------------------------
# Adaptadores exactos
# ---------------------------------------------------------------------------

def _single_variable(cv):
    """True si el compute value ``cv`` depende solo de ``cv.variable``."""
    if not isinstance(cv, Transformation):
        return False
    if ReplaceUnknownsRandom and isinstance(cv, ReplaceUnknownsRandom):
        return False
    return type(cv).__call__ is Transformation.__call__


def feature_origins(model, adom):
    """Atributo original de cada atributo de ``model.domain``.

    Devuelve ``(origins, discretizers)``: ``origins[j]`` es el índice en
    ``adom`` del atributo del que sale el ``j`` (None si no sale de ninguno), u
    ``origins`` es None si alguno depende de varios atributos.
    ``discretizers[a]`` es ``(Discretizer, variable producida)`` cuando todo lo
    que sale del atributo ``a`` se calcula a partir de su tramo.
    """
    lookup = {}
    for i, a in enumerate(adom.attributes):
        lookup.setdefault(a, i)
    origins, last = [], {}
    for feat in model.domain.attributes:
        var, origin, chain = feat, None, []
        for _ in range(100):
            if var in lookup:
                origin = lookup[var]
                break
            cv = var.compute_value
            if cv is None:
                break
            if not _single_variable(cv):
                return None, {}
            chain.append((cv, var))
            var = cv.variable
        origins.append(origin)
        if origin is not None:
            last.setdefault(origin, []).append(chain[-1] if chain else None)
    discretizers = {}
    for a, ends in last.items():
        if all(e is not None and isinstance(e[0], Discretizer) for e in ends):
            pts = [tuple(np.asarray(e[0].points, float)) for e in ends]
            if len(set(pts)) == 1:
                discretizers[a] = ends[0]
    return origins, discretizers


class _ExactAdapter:
    """Descomposición raw de un modelo aditivo."""
    derive = "contrast"
    space = "score"

    def __init__(self, model, adom, kind, K):
        self.model = model
        self.adom = adom
        self.kind = kind
        self.K = K

    def transform(self, X):
        t = _make_table(self.adom, X).transform(self.model.domain)
        return _dense(t.X)

    def present(self):
        return np.ones(self.K, bool)


class LinearAdapter(_ExactAdapter):
    """Modelos lineales en los atributos de ``model.domain`` (``coef_``)."""

    def __init__(self, model, adom, kind, K, origins):
        super().__init__(model, adom, kind, K)
        self.skl = model.skl_model
        self.origins = origins
        q = len(model.domain.attributes)
        self.const = self._decision(np.zeros((1, q)))[0]
        W = []
        for j0 in range(0, q, 500):
            probe = np.eye(q)[j0:j0 + 500]
            W.append(self._decision(probe) - self.const)
        self.W = np.vstack(W) if W else np.zeros((0, len(self.const)))
        self.groups = {}
        for j, o in enumerate(origins):
            if o is not None:
                self.groups.setdefault(o, []).append(j)
        if kind == "regression":
            self.space = "value"
            self.derive = "direct"

    def _decision(self, Xt):
        skl = self.skl
        if self.kind == "regression":
            return np.asarray(skl.predict(Xt), dtype=float).reshape(len(Xt), 1)
        d = np.asarray(skl.decision_function(Xt), dtype=float)
        classes = np.asarray(skl.classes_).astype(int)
        out = np.full((len(Xt), self.K), np.nan)
        if d.ndim == 1 or d.shape[1] == 1:
            d = d.ravel()
            out[:, classes[1]] = d / 2
            out[:, classes[0]] = -d / 2
        else:
            out[:, classes] = d
        return out

    def present(self):
        if self.kind == "regression":
            return np.ones(1, bool)
        pres = np.zeros(self.K, bool)
        pres[np.asarray(self.skl.classes_).astype(int)] = True
        return pres

    def raw_of(self, a, Xt):
        """Efecto raw (n, K) del atributo ``a`` a partir de filas transformadas."""
        cols = self.groups.get(a, [])
        if not cols:
            return np.zeros((len(Xt), self.W.shape[1]))
        return Xt[:, cols] @ self.W[cols]

    def real_score(self, Xt):
        return self._decision(Xt)


class NaiveBayesAdapter(_ExactAdapter):
    def __init__(self, model, adom, kind, K, origins):
        super().__init__(model, adom, kind, K)
        self.origins = origins
        with np.errstate(divide="ignore"):
            self.const = np.log(np.asarray(model.class_prob, dtype=float))
        self.lcp = model.log_cont_prob
        self.cols = {o: j for j, o in enumerate(origins) if o is not None}

    def raw_of(self, a, Xt):
        j = self.cols.get(a)
        if j is None or self.lcp is None:
            return np.zeros((len(Xt), self.K))
        lp = np.asarray(self.lcp[j], dtype=float)       # (K, n_values)
        col = Xt[:, j]
        out = np.zeros((len(Xt), self.K))
        ok = ~np.isnan(col)
        out[ok] = lp[:, col[ok].astype(int)].T
        return out

    def real_score(self, Xt):
        total = np.tile(self.const, (len(Xt), 1))
        for a in self.cols:
            total += self.raw_of(a, Xt)
        return total


def exact_adapter(model, adom, kind, K):
    """Adaptador de un modelo aditivo o ``(None, motivo)``."""
    from Orange.classification.naive_bayes import NaiveBayesModel
    origins, discretizers = feature_origins(model, adom)
    if isinstance(model, NaiveBayesModel):
        if origins is None:
            return None, ("El preprocesado del modelo combina varios "
                          "atributos."), {}
        return NaiveBayesAdapter(model, adom, kind, K, origins), "", \
            discretizers
    skl = getattr(model, "skl_model", None)
    if skl is None or not hasattr(skl, "coef_"):
        return None, "El modelo no es aditivo en los atributos.", {}
    if kind == "classification" and not hasattr(skl, "decision_function"):
        return None, "El modelo no es aditivo en los atributos.", {}
    if origins is None:
        return None, ("El preprocesado del modelo combina varios atributos "
                      "(por ejemplo, PCA o atributos construidos)."), {}
    if kind == "classification":
        classes = np.asarray(skl.classes_)
        n_dec = len(classes)
        if type(skl).__name__ in ("SVC", "NuSVC") and n_dec > 2:
            return None, ("Una SVM con varias clases combina clasificadores "
                          "por pares: no es aditiva."), {}
    try:
        return LinearAdapter(model, adom, kind, K, origins), "", discretizers
    except Exception as exc:
        return None, f"No se pudieron leer los coeficientes ({exc}).", {}


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def _progress(state, value, text=None):
    if state is None:
        return
    if state.is_interruption_requested():
        raise Cancelled
    state.set_progress_value(value)
    if text:
        state.set_status(text)


def analyze(model, data=None, *, method=METHOD_AUTO,
            sample_size=1000, grid_size=20, time_budget=5.0,
            allow_exact=True, seed=0, state=None):
    """Descomposición aditiva de ``model`` (ver el docstring del módulo)."""
    t0 = time.perf_counter()
    kind, cv = model_kind(model)
    adom = attribute_domain(model)
    K = len(cv.values) if kind == "classification" else 1
    ref, source, notes = reference_table(model, data)
    analysis = Analysis(model=model, kind=kind, mode=EXACT, domain=adom,
                        class_var=cv, variants={}, method=method,
                        model_name=model_name(model), notes=notes)
    if ref is not None and len(ref) < MIN_REFERENCE_ROWS:
        od = getattr(model, "original_data", None)
        if source == "data" and od is not None \
                and len(od) >= MIN_REFERENCE_ROWS:
            analysis.notes.append(
                f"Data solo tiene {len(ref)} {_plural(len(ref), 'fila', 'filas')} "
                f"(hacen falta al menos {MIN_REFERENCE_ROWS}): la referencia son "
                "los datos de entrenamiento del modelo.")
            ref, source = od.transform(adom), "training"
        else:
            analysis.warnings.append(
                f"Los datos de referencia solo tienen {len(ref)} "
                f"{_plural(len(ref), 'fila', 'filas')}: los rangos, la distribución "
                "y la fidelidad no son fiables (hacen falta al menos "
                f"{MIN_REFERENCE_ROWS}).")
    sample = None
    rcols = None
    if ref is not None:
        analysis.n_reference = len(ref)
        sample, _ = sample_rows(ref, sample_size, seed, rare=True)
        rcols = range_columns(analysis, model, _dense(ref.X), source)
    analysis.sample = sample
    if source == "training" and data is None:
        analysis.notes.append(
            "Sin datos de entrada: los datos de referencia son los de "
            "entrenamiento del modelo.")
    adapter, reason = None, ""
    discretizers = {}
    if allow_exact:
        adapter, reason, discretizers = exact_adapter(model, adom, kind, K)
    predictor = Predictor(model, adom, kind, state)
    if adapter is not None:
        try:
            ok = _exact(analysis, adapter, discretizers, rcols, sample,
                        predictor, state)
        except Cancelled:
            raise
        except NomogramError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            ok = False
            reason = f"La descomposición exacta falló ({exc})."
        if ok is not True:
            reason = ok if isinstance(ok, str) else reason
            adapter = None
    if adapter is None:
        if sample is None:
            raise NomogramError(
                "Este modelo solo admite un nomograma aproximado, que necesita "
                "datos de referencia: conecta una tabla a la entrada Data.")
        analysis.mode = APPROXIMATE
        analysis.reason = reason
        analysis.variants = {}
        test, train = holdout_split(len(sample), seed)
        if method in (METHOD_AUTO, PARTIAL_DEPENDENCE):
            _partial_dependence(analysis, rcols, sample, grid_size, predictor,
                                time_budget, state, test, train)
        if method in (METHOD_AUTO, ADDITIVE_FIT):
            _additive_fit(analysis, rcols, sample, grid_size, predictor,
                          seed, state, test, train,
                          start=60 if method == METHOD_AUTO else 5)
    if kind == "regression" and ref is not None:
        _residuals(analysis, model, ref, data, source)
    analysis.seconds = time.perf_counter() - t0
    return analysis


def holdout_split(n, seed=0):
    """Filas apartadas (30 %) para medir la fidelidad y filas de trabajo."""
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(round(0.3 * n)) if n >= 10 else 0
    if not n_test:
        return np.arange(n), np.arange(n)
    return np.sort(perm[:n_test]), np.sort(perm[n_test:])


def _residuals(analysis, model, ref, data, source):
    cv = analysis.class_var
    table = data if source == "data" else getattr(model, "original_data", None)
    if table is None or table.domain.class_var is None \
            or table.domain.class_var.name != cv.name \
            or not table.domain.class_var.is_continuous:
        return
    y = np.asarray(table.Y, dtype=float).ravel()
    sample, idx = sample_rows(ref, 5000, 1)
    yy = y[idx]
    ok = ~np.isnan(yy)
    if ok.sum() < 10:
        return
    pred = Predictor(model, analysis.domain, "regression")(sample[ok])[:, 0]
    analysis.residuals = yy[ok] - pred
    analysis.residuals_on_training = source == "training"


def _exact(analysis, adapter, discretizers, rcols, sample, predictor,
           state):
    """Descomposición exacta; True, o el motivo para pasar a la aproximada."""
    adom = analysis.domain
    p = len(adom.attributes)
    K = adapter.K if analysis.kind == "classification" else 1
    analysis.present = adapter.present()
    effects = []
    # 1. rejillas de cada atributo (filas con solo ese atributo conocido)
    blocks, specs = [], []
    for a, var in enumerate(adom.attributes):
        col = rcols[a] if rcols is not None else None
        if var.is_discrete:
            grid = np.arange(len(var.values), dtype=float)
            spec = dict(kind="discrete", grid=grid,
                        labels=list(var.values))
        else:
            disc = discretizers.get(a)
            rng = _continuous_range(col, var) if col is not None else None
            if disc is not None:
                cvd, produced = disc
                thr = np.asarray(cvd.points, dtype=float)
                labels = list(produced.values) if produced.is_discrete \
                    else [str(i) for i in range(len(thr) + 1)]
                if len(labels) != len(thr) + 1:
                    labels = [str(i) for i in range(len(thr) + 1)]
                reps = _bin_representatives(thr, col, rng)
                lo, hi = rng if rng else (
                    (thr[0], thr[-1]) if len(thr) else (0.0, 0.0))
                spec = dict(kind="bins", grid=reps, thresholds=thr,
                            labels=labels, lo=lo, hi=hi)
            else:
                if rng is None:
                    if rcols is None:
                        raise NomogramError(
                            f"Sin datos de referencia no se conoce el rango "
                            f"de «{var.name}»: conecta una tabla a Data.")
                    rng = (0.0, 0.0)
                lo, hi = rng
                grid = np.linspace(lo, hi, FINE_GRID) if hi > lo \
                    else np.array([lo])
                spec = dict(kind="curve", grid=grid, lo=lo, hi=hi)
        rows = np.full((len(spec["grid"]) + 1, p), np.nan)
        rows[:-1, a] = spec["grid"]          # última fila: valor desconocido
        blocks.append(rows)
        specs.append(spec)
    allrows = np.vstack(blocks)
    _progress(state, 10, "Descomposición exacta")
    Xt_all = adapter.transform(allrows)
    start = 0
    for a, (var, spec, rows) in enumerate(zip(adom.attributes, specs, blocks)):
        Xt = Xt_all[start:start + len(rows)]
        start += len(rows)
        raw = adapter.raw_of(a, Xt)
        raw_grid, raw_nan = raw[:-1], raw[-1]
        eff = FeatureEffect(var=var, index=a, kind=spec["kind"],
                            grid=spec["grid"], raw=raw_grid, raw_nan=raw_nan,
                            lo=spec.get("lo", 0.0), hi=spec.get("hi", 0.0),
                            thresholds=spec.get("thresholds"),
                            labels=spec.get("labels"))
        if eff.kind == "curve":
            _maybe_linear(eff)
        effects.append(eff)
    # 2. contribuciones exactas de la muestra de referencia
    if sample is not None:
        Xs = adapter.transform(sample)
        real_score = adapter.real_score(Xs)
        recon = np.tile(adapter.const, (len(sample), 1)).astype(float)
        for eff in effects:
            sr = adapter.raw_of(eff.index, Xs)
            recon = recon + sr
            _centre(eff, sr)
        diff = np.abs(np.nan_to_num(recon - real_score))
        scale = max(1.0, float(np.nanmax(np.abs(real_score)))
                    if np.isfinite(real_score).any() else 1.0)
        max_err = float(diff.max()) / scale if diff.size else 0.0
        if max_err > EXACT_TOL:
            return ("La suma de los efectos no reproduce el modelo "
                    f"(error relativo {max_err:.1e}).")
        _progress(state, 50, "Comprobación con el modelo real")
        real = predictor(sample)
        if analysis.kind == "classification" and isinstance(
                adapter, NaiveBayesAdapter):
            # se comprueba con las probabilidades reales de Naive Bayes
            r = recon - np.nanmax(recon, axis=1, keepdims=True)
            sm = np.exp(r)
            sm /= sm.sum(axis=1, keepdims=True)
            if np.abs(sm - real).max() > 1e-7:
                return "Las probabilidades del modelo no siguen Naive Bayes."
        if analysis.kind == "regression":
            if np.abs(real[:, 0] - recon[:, 0]).max() / scale > EXACT_TOL:
                return ("La predicción del modelo no coincide con su parte "
                        "lineal.")
        base_raw = np.nanmean(recon, axis=0)
        analysis.real = real
        analysis.real_score = real_score
        for eff in effects:
            _distribution(eff, rcols[eff.index])
    else:
        # sin datos: se centra en la rejilla y se calibra con sus filas
        base_raw = adapter.const + sum(
            np.nanmean(e.raw, axis=0) for e in effects)
        for eff in effects:
            _centre(eff, None)
        # verificación en las filas de la rejilla: un atributo conocido y el resto no
        total_nan = adapter.const + sum(e.raw_nan for e in effects)
        recon = []
        for eff, rows in zip(effects, blocks):
            raws = np.vstack([eff.raw if eff.kind != "linear"
                              else eff.evaluate(rows[:-1, eff.index]),
                              eff.raw_nan[None, :]])
            recon.append(total_nan - eff.raw_nan + raws)
        recon = np.vstack(recon)
        rs = adapter.real_score(Xt_all)
        diff = np.abs(np.nan_to_num(recon - rs))
        scale = max(1.0, float(np.nanmax(np.abs(rs)))
                    if np.isfinite(rs).any() else 1.0)
        max_err = float(diff.max()) / scale if diff.size else 0.0
        if max_err > EXACT_TOL:
            return ("La suma de los efectos no reproduce el modelo "
                    f"(error relativo {max_err:.1e}).")
        real = predictor(allrows)
        analysis.real = real
        analysis.real_score = rs
        analysis.notes.append(
            "Sin datos de referencia: los efectos se centran en la media de "
            "sus valores posibles y no se puede calcular la importancia en "
            "los datos.")
    variant = Variant(space=adapter.space, derive=adapter.derive,
                      effects=effects, base_raw=np.asarray(base_raw, float),
                      recon=recon, max_error=max_err)
    analysis.variants = {("exact", variant.space): variant}
    analysis.n_pd = len(sample) if sample is not None else 0
    return True


def _maybe_linear(eff):
    g, raw = eff.grid, eff.raw
    if len(g) < 2:
        eff.kind = "linear"
        eff.slope = np.zeros(raw.shape[1])
        eff.offset = raw[0].copy()
        return
    slope = (raw[-1] - raw[0]) / (g[-1] - g[0])
    pred = raw[0][None, :] + (g - g[0])[:, None] * slope[None, :]
    err = np.nan_to_num(np.abs(pred - raw))
    scale = max(1e-300, float(np.nanmax(np.abs(raw))) if raw.size else 1.0)
    if err.max() <= 1e-9 * max(scale, 1.0):
        eff.kind = "linear"
        eff.slope = slope
        eff.offset = raw[0] - g[0] * slope
        eff.grid = np.array([g[0], g[-1]])
        eff.raw = eff.raw[[0, -1]]


def _bin_representatives(thr, col, rng):
    edges = np.concatenate([[-np.inf], thr, [np.inf]])
    reps = []
    span = (thr[-1] - thr[0]) if len(thr) > 1 else 1.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        vals = None
        if col is not None:
            c = col[~np.isnan(col)]
            vals = c[(c >= lo) & (c < hi)] if i < len(edges) - 2 \
                else c[c >= lo]
        if vals is not None and len(vals):
            reps.append(float(np.median(vals)))
        elif np.isfinite(lo) and np.isfinite(hi):
            reps.append((lo + hi) / 2)
        elif np.isfinite(hi):
            reps.append(hi - 0.1 * (span or 1.0))
        elif np.isfinite(lo):
            reps.append(lo + 0.1 * (span or 1.0))
        else:
            reps.append(0.0)
    return np.asarray(reps, dtype=float)


# ---------------------------------------------------------------------------
# Descomposiciones aproximadas
# ---------------------------------------------------------------------------

def _approx_grids(adom, rcols, grid_size):
    specs = []
    for a, var in enumerate(adom.attributes):
        col = rcols[a]
        if var.is_discrete:
            specs.append(dict(kind="discrete",
                              grid=np.arange(len(var.values), dtype=float),
                              labels=list(var.values)))
        else:
            rng = _continuous_range(col, var)
            if rng is None:
                specs.append(dict(kind="curve", grid=np.array([0.0]),
                                  lo=0.0, hi=0.0, empty=True))
                continue
            grid = continuous_grid(col, grid_size)
            if not len(grid):
                grid = np.array([rng[0]])
            lo, hi = min(rng[0], grid[0]), max(rng[1], grid[-1])
            grid[0], grid[-1] = lo, hi
            grid = np.unique(grid)
            specs.append(dict(kind="curve", grid=grid, lo=lo, hi=hi))
    return specs


def _handles_nan(predictor, sample):
    row = sample[:1].copy()
    row[:] = np.nan
    try:
        res = predictor(row)
        return bool(np.isfinite(res).all())
    except Cancelled:
        raise
    except Exception:
        return False


def _spaces(kind, P):
    if kind == "regression":
        return {"value": P}
    return {"prob": P, "logit": logit(P)}


def _finish_variants(analysis, tag, specs, raws, raw_nans, rcols, sample,
                     real, holdout=None, unsupported=None):
    """Construye las variantes a partir de los efectos raw en las rejillas."""
    for space, real_space in _spaces(analysis.kind, real).items():
        effects = []
        n = len(sample)
        recon = np.zeros((n, real_space.shape[1]))
        for a, var in enumerate(analysis.domain.attributes):
            spec = specs[a]
            eff = FeatureEffect(
                var=var, index=a, kind=spec["kind"], grid=spec["grid"],
                raw=raws[space][a],
                raw_nan=None if raw_nans[space][a] is None
                else raw_nans[space][a],
                lo=spec.get("lo", 0.0), hi=spec.get("hi", 0.0),
                labels=spec.get("labels"))
            if unsupported is not None and unsupported[a] is not None:
                eff.unsupported = unsupported[a]
            sr = eff.evaluate(sample[:, a])
            _centre(eff, sr)
            if eff.unsupported is not None:
                eff.raw = eff.raw.copy()
                eff.raw[eff.unsupported] = np.nan   # no se dibujan como efectos
            recon += sr - eff.mean
            _distribution(eff, rcols[a])
            effects.append(eff)
        base = real_space.mean(axis=0)
        recon += base
        derive = "direct"
        analysis.variants[(tag, space)] = Variant(
            space=space, derive=derive, effects=effects,
            base_raw=np.asarray(base, dtype=float), recon=recon,
            holdout=holdout, method=tag)


def _pd_grid_size(adom, grid_size):
    """Puntos por numérico: menos cuando hay muchos, para no perder filas."""
    n_cont = sum(1 for a in adom.attributes if not a.is_discrete)
    if not n_cont:
        return grid_size
    return int(max(8, min(grid_size, PD_GRID_POINTS // n_cont)))


def _partial_dependence(analysis, rcols, sample, grid_size, predictor,
                        time_budget, state, test, train):
    """Dependencia parcial con las filas de trabajo de fondo.

    La fidelidad se mide en las filas apartadas (``test``), que no se usan
    como fondo, igual que en el ajuste aditivo.
    """
    adom = analysis.domain
    kind = analysis.kind
    specs = _approx_grids(adom, rcols, _pd_grid_size(adom, grid_size))
    _progress(state, 2, "Predicciones del modelo")
    t0 = time.perf_counter()
    real = predictor(sample)
    per_row = (time.perf_counter() - t0) / max(1, len(sample))
    analysis.real = real
    nan_ok = _handles_nan(predictor, sample)
    analysis.has_nan_effect = nan_ok
    rows_per_sample = sum(len(s["grid"]) + nan_ok for s in specs)
    n = len(train)
    budget_rows = time_budget / max(per_row, 1e-7)
    n_pd = int(min(n, max(min(MIN_PD_ROWS, n),
                          budget_rows // max(1, rows_per_sample))))
    analysis.n_pd = n_pd
    pd_note = ""
    if n_pd < n:
        pd_note = (
            f"La dependencia parcial usa {n_pd} filas de fondo de las {n} "
            "disponibles para no tardar demasiado con este modelo.")
    base_rows = sample[train[:n_pd]]
    spaces = list(_spaces(kind, real[:1]).keys())
    raws = {s: [None] * len(specs) for s in spaces}
    raw_nans = {s: [None] * len(specs) for s in spaces}
    # se predicen varios atributos por llamada: menos sobrecoste por llamada
    pending, pending_rows = [], 0

    def flush():
        nonlocal pending, pending_rows
        if not pending:
            return
        X = np.vstack([blk for _, _, blk in pending])
        P = predictor(X)
        start = 0
        for a, m, blk in pending:
            Pa = P[start:start + len(blk)].reshape(m, n_pd, -1)
            start += len(blk)
            for space in spaces:
                vals = Pa if space in ("prob", "value") else logit(Pa)
                pdv = vals.mean(axis=1)
                if nan_ok:
                    raws[space][a], raw_nans[space][a] = pdv[:-1], pdv[-1]
                else:
                    raws[space][a] = pdv
        pending, pending_rows = [], 0

    total = len(specs)
    for a, spec in enumerate(specs):
        _progress(state, 5 + 55 * a / max(1, total),
                  f"Dependencia parcial: {adom.attributes[a].name}")
        grid = list(spec["grid"]) + ([np.nan] if nan_ok else [])
        m = len(grid)
        X = np.tile(base_rows, (m, 1))
        X[:, a] = np.repeat(np.asarray(grid, dtype=float), n_pd)
        if pending_rows + len(X) > MAX_ROWS_PER_CALL:
            flush()
        pending.append((a, m, X))
        pending_rows += len(X)
    flush()
    _finish_variants(analysis, "pd", specs, raws, raw_nans, rcols, sample,
                     real, holdout=test)
    for (tag, _), var in analysis.variants.items():
        if tag == "pd":
            var.note = pd_note


def _hat_basis(grid, values):
    """Pesos de interpolación lineal a trozos (n, len(grid))."""
    n, m = len(values), len(grid)
    B = np.zeros((n, m))
    if m == 1:
        B[:, 0] = 1
        return B
    v = np.clip(values, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, v, side="right") - 1, 0, m - 2)
    w = (v - grid[j]) / (grid[j + 1] - grid[j])
    rows = np.arange(n)
    B[rows, j] = 1 - w
    B[rows, j + 1] += w
    return B


def _fit_basis(specs, sample):
    n = len(sample)
    blocks, slices = [], []
    start = 0
    for a, spec in enumerate(specs):
        col = sample[:, a]
        nan = np.isnan(col)
        if spec["kind"] == "discrete":
            B = np.zeros((n, len(spec["grid"])))
            ok = ~nan
            B[np.where(ok)[0], col[ok].astype(int)] = 1
        else:
            B = _hat_basis(spec["grid"], np.where(nan, spec["grid"][0], col))
            B[nan] = 0
        has_nan = bool(nan.any())
        if has_nan:
            B = np.column_stack([B, nan.astype(float)])
        blocks.append(B)
        slices.append((start, start + B.shape[1], has_nan))
        start += B.shape[1]
    Bfull = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return Bfull, slices


def _choose_lambda(B, Y, penalty, seed):
    """La penalización más fuerte a menos de un 5 % del mejor error de validación.

    La validación se hace dentro de las filas de trabajo, así que las filas
    apartadas para la fidelidad no intervienen en el ajuste.
    """
    n = len(B)
    if n < 10:
        return 1e-2
    perm = np.random.RandomState(1).permutation(n)
    n_val = int(round(0.25 * n))
    val, fit = perm[:n_val], perm[n_val:]
    trials = []
    for cand in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        coef, const = _ridge(B[fit], Y[fit], cand, penalty)
        pred = B[val] @ coef + const
        trials.append((cand, float(np.mean((pred - Y[val]) ** 2))))
    best = min(t[1] for t in trials)
    return [t for t in trials if t[1] <= best * 1.05 + 1e-300][-1][0]


def _additive_fit(analysis, rcols, sample, grid_size, predictor, seed,
                  state, test, train, start=5):
    """Sustituto aditivo por mínimos cuadrados de las predicciones del modelo.

    Se ajusta con las filas de trabajo y su fidelidad se mide en las apartadas;
    lo que se dibuja es el ajuste con todas las filas.
    """
    adom = analysis.domain
    # el sistema lineal crece con el cubo de las columnas: con muchos atributos
    # se usan menos nudos por atributo numérico
    n_cont = sum(1 for a in adom.attributes if not a.is_discrete)
    n_disc = sum(len(a.values) for a in adom.attributes if a.is_discrete)
    room = max(0, MAX_FIT_COLUMNS - n_disc)
    if n_cont:
        grid_size = int(max(3, min(grid_size, room // n_cont)))
    specs = _approx_grids(adom, rcols, grid_size)
    _progress(state, start, "Ajuste aditivo")
    if analysis.real is None:
        analysis.real = predictor(sample)
        analysis.n_pd = 0
    real = analysis.real
    Bfull, slices = _fit_basis(specs, sample)
    penalty = _smooth_penalty(specs, slices, Bfull.shape[1])
    separate = len(test) < len(sample)
    # categorías sin ninguna fila en la muestra: no tienen efecto estimado
    unsupported = []
    for a, spec in enumerate(specs):
        if spec["kind"] == "discrete":
            col = sample[:, a]
            counts = np.bincount(col[~np.isnan(col)].astype(int),
                                 minlength=len(spec["grid"]))
            mask = counts[:len(spec["grid"])] == 0
            unsupported.append(mask if mask.any() else None)
        else:
            unsupported.append(None)
    raws, raw_nans = {}, {}
    holdout_recon = {}
    for space, target in _spaces(analysis.kind, real).items():
        if separate:
            lam = _choose_lambda(Bfull[train], target[train], penalty, seed)
            coef_tr, mean_tr = _ridge(Bfull[train], target[train], lam, penalty)
            holdout_recon[space] = Bfull[test] @ coef_tr + mean_tr
        else:
            lam = _choose_lambda(Bfull, target, penalty, seed)
        coef, mean = _ridge(Bfull, target, lam, penalty)
        raws[space], raw_nans[space] = [], []
        for (s, e, has_nan), spec in zip(slices, specs):
            c = coef[s:e]
            if has_nan:
                raws[space].append(c[:-1])
                raw_nans[space].append(c[-1])
            else:
                raws[space].append(c)
                raw_nans[space].append(None)
    _finish_variants(analysis, "fit", specs, raws, raw_nans, rcols, sample,
                     real, holdout=test, unsupported=unsupported)
    if separate:
        for (tag, space), var in analysis.variants.items():
            if tag == "fit":
                # la fidelidad, con el ajuste que no vio las filas apartadas
                var.recon[test] = holdout_recon[space]


def _smooth_penalty(specs, slices, ncols):
    """Penalización de segundas diferencias en las bases continuas (P-spline).

    Hace que los efectos ajustados sean curvas suaves en lugar de seguir el
    ruido de cada nudo; en las categóricas y en los desconocidos, cresta.
    """
    P = np.zeros((ncols, ncols))
    for (s, e, has_nan), spec in zip(slices, specs):
        m = e - s - (1 if has_nan else 0)
        if spec["kind"] != "discrete" and m >= 3:
            D = np.diff(np.eye(m), n=2, axis=0)
            P[s:s + m, s:s + m] += D.T @ D
            P[s:s + m, s:s + m] += 1e-3 * np.eye(m)
        else:
            P[s:s + m, s:s + m] += np.eye(m)
        if has_nan:
            P[e - 1, e - 1] += 1.0
    return P


def _ridge(B, Y, lam=1e-2, penalty=None):
    """Mínimos cuadrados penalizados; devuelve coeficientes y constante."""
    mean = Y.mean(axis=0)
    if B.shape[1] == 0:
        return np.zeros((0, Y.shape[1])), mean
    Bc = B - B.mean(axis=0)
    G = Bc.T @ Bc
    P = np.eye(G.shape[0]) if penalty is None else penalty
    reg = lam * (np.trace(G) / max(1, G.shape[0]) + 1e-12)
    coef = np.linalg.solve(G + reg * P + 1e-10 * reg * np.eye(G.shape[0]),
                           Bc.T @ (Y - mean))
    # constante para que las predicciones sean B @ coef + const
    const = mean - B.mean(axis=0) @ coef
    return coef, const


# ---------------------------------------------------------------------------
# Puntos y marcas
# ---------------------------------------------------------------------------

@dataclass
class PointScale:
    """Correspondencia entre puntuaciones y puntos de los efectos mostrados."""
    k: float                    # puntos por unidad de puntuación
    mins: dict                  # índice del atributo -> componente mínima
    maxs: dict
    offset: float               # puntuación con cero puntos

    def points(self, a, comp):
        return (np.asarray(comp) - self.mins[a]) * self.k

    def score_of_total(self, total):
        return self.offset + np.asarray(total) / self.k

    def total_of_score(self, score):
        return (np.asarray(score) - self.offset) * self.k


def display_range(view, effect, extra_values=()):
    """Componentes mínima y máxima de un efecto en lo que se dibuja."""
    comp = view.grid_component(effect)
    vals = list(np.ravel(comp))
    if effect.kind in ("linear", "curve"):
        ext = [v for v in extra_values if v is not None and np.isfinite(v)]
        if ext and effect.kind == "linear":
            vals += list(view.component(effect, ext))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return 0.0, 0.0
    return float(min(vals)), float(max(vals))


def point_scale(view, effects=None, extra=None, max_points=100.0):
    """Puntos de los efectos mostrados; el más amplio abarca ``max_points``.

    ``offset`` es la puntuación con cero puntos en todos los efectos mostrados
    (los que no se muestran no se incluyen: hay que sumar sus componentes).
    """
    effects = view.effects if effects is None else effects
    extra = extra or {}
    mins, maxs = {}, {}
    for e in effects:
        lo, hi = display_range(view, e, extra.get(e.index, ()))
        mins[e.index], maxs[e.index] = lo, hi
    spans = [maxs[a] - mins[a] for a in mins]
    top = max(spans) if spans else 0.0
    k = max_points / top if top > 1e-300 else 1.0
    offset = view.base + sum(mins.values())
    return PointScale(k=k, mins=mins, maxs=maxs, offset=offset)


def nice_ticks(lo, hi, n=6):
    """Marcas redondas en [lo, hi]."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return []
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo <= 1e-12 * max(1.0, abs(lo), abs(hi)):
        return [lo]
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = first
    while v <= hi + step * 1e-9:
        ticks.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return ticks


def format_number(v, digits=4):
    if v is None or not np.isfinite(v):
        return "?"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        txt = f"{v:.{max(1, digits - 1)}e}"
        mant, exp = txt.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}".replace(".", ",")
    decimals = max(0, digits - 1 - int(math.floor(math.log10(a))))
    txt = f"{v:.{decimals}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt.replace(".", ",").replace("-", "−")


def format_value(var, value):
    """Texto del valor de un atributo."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "?"
    if var.is_discrete:
        i = int(value)
        return var.values[i] if 0 <= i < len(var.values) else "?"
    if isinstance(var, TimeVariable):
        return var.repr_val(value)
    return format_number(value)


def span(values):
    """Rango (máx − mín) de los valores finitos; 0 si no hay."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if len(v) else 0.0


def is_monotone(values, tol_rel=0.03):
    """+1, -1 o 0 (no monótono, más allá de ``tol_rel`` del rango)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 1
    span = np.ptp(v)
    if span <= 1e-300:
        return 1
    d = np.diff(v)
    tol = tol_rel * span
    if np.all(d >= -tol):
        return 1
    if np.all(d <= tol):
        return -1
    return 0


def default_target(class_var):
    """Índice de la clase por defecto: la positiva si es binaria."""
    names = [v.lower() for v in class_var.values]
    if len(names) == 2:
        for pos in ("1", "yes", "true", "si", "sí", "positive", "pos", "y",
                    "s"):
            if pos in names:
                return names.index(pos)
        return 1
    return 0
