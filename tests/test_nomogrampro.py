# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Prueba sin interfaz gráfica de Nomogram Pro.

Se ejecuta con el Python de Orange (scripts/probar.sh lo hace con PYTHONPATH apuntando
al repo) y sale con 1 si alguna comprobación falla. Cubre la exactitud del modo exacto
(la suma de los puntos reproduce el modelo), el contraste con el Nomogram nativo, el
modo aproximado con los modelos del panel Model y los casos límite.
"""
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QEvent, QPointF
from AnyQt.QtWidgets import QApplication
from Orange.data import (
    ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, TimeVariable,
)
import Orange.modelling as Mo
from Orange.classification import (
    LogisticRegressionLearner, NaiveBayesLearner, SGDClassificationLearner, SVMLearner,
    LinearSVMLearner, KNNLearner, NNClassificationLearner, RandomForestLearner,
    CN2Learner, MajorityLearner, CalibratedLearner, ScoringSheetLearner,
)
from Orange.regression import (
    LinearRegressionLearner, RidgeRegressionLearner, LassoRegressionLearner,
    ElasticNetLearner, SGDRegressionLearner, SVRLearner, PLSRegressionLearner,
    KNNRegressionLearner, NNRegressionLearner, RandomForestRegressionLearner, MeanLearner,
    CurveFitLearner,
)
from Orange.ensembles.stack import StackedFitter
from Orange.preprocess import Normalize, Discretize
from Orange.preprocess.preprocess import ProjectPCA

from orangecontrib.orange_gualbe_toolbox.widgets import _nomo_engine as E
from orangecontrib.orange_gualbe_toolbox.widgets.ownomogrampro import (
    OWNomogramPro, SORT_RANGE, SORT_INSTANCE, UNITS, POINTS,
)

warnings.simplefilter("ignore")
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []


def _excepcion(*e):
    """Las excepciones de Qt se cuentan; una del propio script se ve y lo para."""
    texto = "".join(traceback.format_exception(*e))
    excepciones.append(texto)
    print("EXCEPCIÓN\n" + texto)
    sys.stdout.flush()


sys.excepthook = _excepcion

HEART = Table("heart_disease")
IRIS = Table("iris")
HOUSING = Table("housing")


def check(ok, msg):
    global fallos
    if not ok:
        print("FAIL " + msg)
    fallos += not ok
    return ok


def silent(fn, *args, **kw):
    """Entrena sin los avisos de convergencia de scikit-learn por la salida."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kw)


AJUSTES = dict(target_class_name="", scale_index=POINTS, sort_index=SORT_RANGE,
               n_attributes=8, show_all=False, method_index=E.METHOD_AUTO,
               space_index=E.SPACE_AUTO, sample_size=1000, grid_size=20,
               show_distribution=True, show_contributions=True, saved_values={},
               auto_commit=True)


class Banco:
    """Un widget real con todos los ajustes fijados y las salidas capturadas."""

    def __init__(self, **ajustes):
        stored = dict(AJUSTES, **ajustes)
        self.w = OWNomogramPro.__new__(OWNomogramPro, signal_manager=None,
                                       stored_settings=stored)
        self.w.__init__()
        self.w.resize(1200, 800)
        self.out = {}
        for name in ("features", "instance", "contributions"):
            getattr(self.w.Outputs, name).send = \
                (lambda n: (lambda v: self.out.__setitem__(n, v)))(name)

    def enviar(self, model=None, data=None, instance=None):
        self.w.set_model(model)
        self.w.set_data(data)
        self.w.set_instance(instance)
        self.w.handleNewSignals()
        self.esperar()

    def esperar(self, limite=300):
        t = time.time()
        while self.w.task is not None and time.time() - t < limite:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()

    def cerrar(self):
        self.w.onDeleteWidget()
        self.w.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def probs_of(model, values, adom):
    return E.predict_instance(model, adom, values)


def random_instances(analysis, n=40, seed=3, with_nan=True):
    """Instancias al azar en el rango de los datos, con algún desconocido."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        row = []
        for e in next(iter(analysis.variants.values())).effects:
            if with_nan and rng.rand() < 0.08:
                row.append(np.nan)
            elif e.var.is_discrete:
                row.append(float(rng.randint(len(e.var.values))))
            else:
                lo, hi = (e.lo, e.hi) if e.hi > e.lo else (e.lo - 1, e.lo + 1)
                row.append(float(rng.uniform(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))))
        rows.append(row)
    return np.array(rows)


# ---------------------------------------------------------------------------
print("1. Modo exacto: la suma de los puntos reproduce el modelo")
EXACTOS = [
    ("LR binaria", LogisticRegressionLearner(), HEART),
    ("NB binaria", NaiveBayesLearner(), HEART),
    ("LR multiclase", LogisticRegressionLearner(), IRIS),
    ("NB multiclase", NaiveBayesLearner(), IRIS),
    ("SVM lineal", SVMLearner(kernel="linear", probability=True), HEART),
    ("SGD (bisagra)", SGDClassificationLearner(), HEART),
    ("SVM lineal (LinearSVC)", LinearSVMLearner(), HEART),
    ("LR con Normalize", LogisticRegressionLearner(
        preprocessors=list(LogisticRegressionLearner.preprocessors) + [Normalize()]), HEART),
    ("LR con Discretize", LogisticRegressionLearner(
        preprocessors=[Discretize()] + list(LogisticRegressionLearner.preprocessors)), HEART),
    ("Regresión lineal", LinearRegressionLearner(), HOUSING),
    ("Ridge", RidgeRegressionLearner(), HOUSING),
    ("Lasso", LassoRegressionLearner(), HOUSING),
    ("Elastic Net", ElasticNetLearner(), HOUSING),
    ("SGD regresión", SGDRegressionLearner(), HOUSING),
    ("SVR lineal", SVRLearner(kernel="linear"), HOUSING),
    ("PLS", PLSRegressionLearner(), HOUSING),
]
for nombre, learner, data in EXACTOS:
    model = silent(learner, data)
    a = E.analyze(model)
    if not check(a.mode == E.EXACT, f"{nombre}: modo exacto ({a.mode}: {a.reason})"):
        continue
    inst = random_instances(a)
    kind = a.kind
    targets = range(len(a.class_var.values)) if kind == "classification" else [0]
    for t in targets:
        view = a.target_view(t)
        check(view.fidelity.get("max_error", 1) <= 1e-9,
              f"{nombre}: verificación {view.fidelity.get('max_error')}")
        for row in inst:
            score = view.score(row)
            real = probs_of(model, row, a.domain)
            if kind == "regression":
                ok = abs(score - real) <= 1e-8 * max(1, abs(real))
                if not check(ok, f"{nombre}: suma {score} ≠ modelo {real}"):
                    break
            else:
                # puntuación real del modelo: decisión o log-posterior
                if view.link.kind == "logistic":
                    p = float(view.link(score))
                    ok = abs(p - real[t]) <= 1e-9
                    if not check(ok, f"{nombre} clase {t}: P suma {p} ≠ P modelo {real[t]}"):
                        break
                elif view.link.kind == "monotone":
                    # con varias clases la escala de probabilidad es aproximada:
                    # se mide en los datos de referencia, no en instancias al azar
                    ok = view.fidelity.get("mae", 1) <= 0.05
                    if not check(ok, f"{nombre} clase {t}: error medio de la escala "
                                     f"de probabilidad {view.fidelity.get('mae')}"):
                        break
    # la puntuación exacta coincide con la del modelo en instancias nuevas
    if kind == "classification":
        from orangecontrib.orange_gualbe_toolbox.widgets._nomo_engine import exact_adapter
        adapter, _, _ = exact_adapter(model, a.domain, kind, len(a.class_var.values))
        rs = adapter.real_score(adapter.transform(inst))
        view = a.target_view(0)
        add = np.array([view.score(r) for r in inst])
        real = view._dot(rs)
        check(np.allclose(add, real, atol=1e-8 * max(1, np.abs(real).max())),
              f"{nombre}: puntuación aditiva ≠ puntuación del modelo en instancias nuevas")
    if nombre in ("LR binaria", "NB binaria"):
        view = a.target_view(1)
        check(view.link.kind == "logistic" and abs(view.link.a - 1) < 1e-6
              and abs(view.link.b) < 1e-6, f"{nombre}: el pie es la logística exacta")
    if nombre == "SGD (bisagra)" or "LinearSVC" in nombre:
        check(a.target_view(1).link.kind == "decision",
              f"{nombre}: sin probabilidades, el pie es la decisión")
    if nombre == "LR con Discretize":
        kinds = {e.kind for e in a.variants[("exact", "score")].effects}
        check("bins" in kinds, f"{nombre}: los atributos discretizados van por tramos {kinds}")
    if nombre == "NB binaria":
        kinds = {e.name: e.kind for e in a.variants[("exact", "score")].effects}
        check(kinds["age"] == "bins", f"NB: la edad discretizada va por tramos ({kinds['age']})")
    if nombre == "Regresión lineal":
        kinds = {e.kind for e in a.variants[("exact", "value")].effects}
        check(kinds <= {"linear", "discrete"}, f"regresión lineal: escalas lineales {kinds}")

# ---------------------------------------------------------------------------
print("2. Contraste con el Nomogram nativo (LR y NB binarias)")
from Orange.widgets.visualize.ownomogram import OWNomogram, SortBy

for nombre, learner in (("LR", LogisticRegressionLearner()), ("NB", NaiveBayesLearner())):
    model = silent(learner, HEART)
    nat = OWNomogram.__new__(OWNomogram, signal_manager=None,
                             stored_settings=dict(target_class_index=1, scale=0,
                                                  display_index=1, n_attributes=13,
                                                  sort_index=SortBy.ABSOLUTE,
                                                  normalize_probabilities=False,
                                                  cont_feature_dim_index=0))
    nat.__init__()
    nat.set_classifier(model)
    nat.target_class_index = 1
    native_order = [attr.name for _, attr in nat.get_ordered_attributes()]
    native_ptp = {}
    for i, attr in enumerate(nat.domain.attributes):
        native_ptp[attr.name] = float(np.ptp(nat.points[i][1]))
    a = E.analyze(model)
    view = a.target_view(1)
    mine_ptp = {e.name: float(np.ptp(view.grid_component(e))) for e in view.effects}
    same = all(abs(mine_ptp[n] - native_ptp[n]) <= 1e-6 * max(1, native_ptp[n])
               for n in native_ptp)
    if not check(same, f"{nombre}: rangos de puntos distintos del nativo"):
        for n in native_ptp:
            print("   ", n, native_ptp[n], mine_ptp[n])
    b = Banco(sort_index=SORT_RANGE, target_class_name="1", show_all=True)
    b.enviar(model)
    mine_order = [e.name for e in b.w._ordered_effects()]
    if nombre == "LR":
        check(mine_order == native_order,
              f"{nombre}: orden distinto del nativo {mine_order} vs {native_order}")
    else:
        # el nativo ordena las continuas de NB por su rango sin discretizar; se
        # comparan los rangos, que son lo que dibuja
        rank_native = sorted(native_ptp, key=lambda n: -native_ptp[n])
        check(mine_order == rank_native, f"{nombre}: orden por rango {mine_order}")
    check(b.w.analysis.mode == E.EXACT, f"{nombre}: el widget usa el modo exacto")
    nat.onDeleteWidget()
    nat.deleteLater()
    b.cerrar()

# ---------------------------------------------------------------------------
print("3. Modo aproximado: no falla, mide la fidelidad y muestra la predicción real")
APROX = [
    ("Árbol", Mo.TreeLearner(), HEART), ("Random Forest", RandomForestLearner(), HEART),
    ("kNN", KNNLearner(), HEART), ("SVM RBF", SVMLearner(probability=True), HEART),
    ("Red neuronal", NNClassificationLearner(), HEART),
    ("Gradient Boosting", Mo.GBLearner(), HEART), ("CN2", CN2Learner(), HEART),
    ("Mayoritaria", MajorityLearner(), HEART),
    ("Stacking", StackedFitter([LogisticRegressionLearner(), Mo.TreeLearner()]), HEART),
    ("Calibrado", CalibratedLearner(LogisticRegressionLearner()), HEART),
    ("Scoring Sheet", ScoringSheetLearner(), HEART),
    ("RF multiclase", RandomForestLearner(), IRIS),
    ("LR con PCA", LogisticRegressionLearner(
        preprocessors=list(LogisticRegressionLearner.preprocessors) + [ProjectPCA(n_components=3)]),
     HEART),
    ("Árbol reg.", Mo.TreeLearner(), HOUSING),
    ("RF reg.", RandomForestRegressionLearner(), HOUSING),
    ("kNN reg.", KNNRegressionLearner(), HOUSING),
    ("SVR RBF", SVRLearner(), HOUSING), ("Red neuronal reg.", NNRegressionLearner(), HOUSING),
    ("Media", MeanLearner(), HOUSING),
    ("AdaBoost", Mo.SklAdaBoostLearner(), HEART),
    ("AdaBoost reg.", Mo.SklAdaBoostLearner(), HOUSING),
    ("Curve Fit", CurveFitLearner(lambda x, a, b, c: a * x[:, 0] + b * x[:, 1] ** 2 + c,
                                  ["a", "b", "c"], ["LSTAT", "RM"]), HOUSING),
]
try:
    APROX.append(("XGBoost", Mo.XGBLearner(), HEART))
    APROX.append(("XGBoost reg.", Mo.XGBLearner(), HOUSING))
except Exception:
    pass
try:
    APROX.append(("CatBoost reg.", Mo.CatGBLearner(), HOUSING))
except Exception:
    pass
for nombre, learner, data in APROX:
    model = silent(learner, data)
    b = Banco(n_attributes=6)
    try:
        b.enviar(model, data)
        w = b.w
        a = w.analysis
        if not check(a is not None and not w.Error.active,
                     f"{nombre}: sin análisis {[str(m) for m in w.Error.active]}"):
            continue
        check(a.mode == E.APPROXIMATE,
              f"{nombre}: modo aproximado ({a.mode})")
        fid = w.view_.fidelity
        check(fid.get("rows", 0) > 0 and np.isfinite(fid.get("mae", np.nan)),
              f"{nombre}: fidelidad calculada {fid}")
        if nombre == "LR con PCA":
            check("combina" in a.reason, f"PCA: explica por qué no es exacto ({a.reason})")
        real = silent(E.predict_instance, model, a.domain, w.values)
        if a.kind == "classification":
            check(np.allclose(w.real, real), f"{nombre}: predicción mostrada ≠ modelo")
        else:
            check(abs(w.real - real) <= 1e-9 * max(1, abs(real)),
                  f"{nombre}: predicción mostrada ≠ modelo")
        # el texto del pie lleva la predicción real
        texts = [it.text() for it in w.scene.items() if hasattr(it, "text")]
        check(any("Modelo real" in t for t in texts), f"{nombre}: el pie muestra el modelo real")
        # arrastrar un marcador cambia el valor, la predicción y la salida
        row = w.rows[0]
        before = w.values[row.effect.index]
        target_x = row.right if abs(row.xs[np.argmin(np.abs(row.values - before))]
                                    - row.left) < 1 else row.left
        pos = QPointF(target_x, row.line_y if not row.curve else row.y + 12)
        w.set_from_position(row, pos, dragging=True)
        w.drag_finished()
        app.processEvents()
        check(w.values[row.effect.index] != before, f"{nombre}: el arrastre cambia el valor")
        inst = b.out.get("instance")
        check(inst is not None and inst.X[0, row.effect.index] == w.values[row.effect.index],
              f"{nombre}: la salida Instance lleva el valor arrastrado")
        real2 = silent(E.predict_instance, model, a.domain, w.values)
        check(np.allclose(np.atleast_1d(w.real), np.atleast_1d(real2)),
              f"{nombre}: la predicción real se actualiza al arrastrar")
    except Exception:
        check(False, f"{nombre}: excepción\n{traceback.format_exc()}")
    finally:
        b.cerrar()

# fidelidad: un modelo aditivo tratado como aproximado sale casi perfecto
m = silent(LinearRegressionLearner(), HOUSING)
a = E.analyze(m, allow_exact=False)
check(a.target_view(0).fidelity["r2"] > 0.9999, "lineal aproximada: fidelidad ≈ 1")
a = E.analyze(m, allow_exact=False, method=E.ADDITIVE_FIT)
check(a.target_view(0).fidelity["r2"] > 0.999, "lineal por ajuste aditivo: fidelidad ≈ 1")

# ---------------------------------------------------------------------------
print("4. Casos límite")
# 4a. modelo sin datos de referencia
m = silent(NaiveBayesLearner(), HEART)
m.original_data = None
a = E.analyze(m)
check(a.mode == E.EXACT and a.real is not None, "NB sin datos: exacto")
view = a.target_view(1)
row = random_instances(a, 1)[0]
p = float(view.link(view.score(row)))
check(abs(p - probs_of(m, row, a.domain)[1]) < 1e-9, "NB sin datos: probabilidad exacta")
m = silent(LogisticRegressionLearner(), HEART)
m.original_data = None
try:
    E.analyze(m)
    check(False, "LR sin datos: debe pedir datos")
except E.NomogramError as exc:
    check("Data" in str(exc), f"LR sin datos: mensaje claro ({exc})")
a = E.analyze(m, HEART)
check(a.mode == E.EXACT, "LR sin datos de entrenamiento, con Data: exacto")
m = silent(RandomForestLearner(), HEART)
m.original_data = None
b = Banco()
b.enviar(m)
check(b.w.Error.unsupported.is_shown() and "Data" in str(b.w.Error.unsupported),
      "RF sin datos: error que pide Data")
b.enviar(m, HEART)
check(not b.w.Error.active and b.w.analysis.mode == E.APPROXIMATE, "RF con Data: aproximado")
b.cerrar()

# 4b. dominio con metas, desconocidos y una sola variable
dom = Domain([ContinuousVariable("x"), DiscreteVariable("d", values=("a", "b", "c"))],
             DiscreteVariable("y", values=("no", "sí")),
             metas=[StringVariable("nombre"), ContinuousVariable("peso")])
rng = np.random.RandomState(0)
X = np.column_stack([rng.normal(size=300), rng.randint(3, size=300)]).astype(float)
X[rng.rand(300) < 0.1, 0] = np.nan
X[rng.rand(300) < 0.1, 1] = np.nan
y = ((np.nan_to_num(X[:, 0]) + (X[:, 1] == 2) + rng.normal(scale=0.5, size=300)) > 0.5)
metas = np.array([[f"f{i}", float(i)] for i in range(300)], dtype=object)
T = Table.from_numpy(dom, X, y.astype(float), metas=metas)
for nombre, learner in (("LR", LogisticRegressionLearner()), ("RF", RandomForestLearner()),
                        ("NB", NaiveBayesLearner())):
    m = silent(learner, T)
    b = Banco()
    b.enviar(m, T, T[5:6])
    w = b.w
    check(not w.Error.active, f"metas y desconocidos ({nombre}): sin error")
    check(w.view_ is not None and len(w.rows) == 2, f"metas ({nombre}): dos atributos")
    check(w.target_class_name == "sí", f"metas ({nombre}): clase positiva por defecto")
    # la instancia de la entrada se aplica, también con desconocidos
    same = np.array_equal(w.values, T.X[5], equal_nan=True)
    check(same, f"metas ({nombre}): valores de Instance {w.values} vs {T.X[5]}")
    # instancia con valor desconocido
    w.values[0] = np.nan
    w._values_changed(final=True)
    check(w.real is not None, f"desconocido ({nombre}): el modelo predice")
    b.cerrar()
T1 = Table.from_numpy(Domain([ContinuousVariable("x")], ContinuousVariable("y")),
                      rng.normal(size=(200, 1)), rng.normal(size=200))
for learner in (LinearRegressionLearner(), RandomForestRegressionLearner()):
    m = silent(learner, T1)
    b = Banco()
    b.enviar(m, T1)
    check(not b.w.Error.active and len(b.w.rows) == 1, "una sola variable")
    b.cerrar()

# 4c. cientos de atributos: solo los N más importantes, en un tiempo razonable
n_attr = 300
X = rng.normal(size=(1500, n_attr))
yb = (X[:, 0] - 2 * X[:, 1] + X[:, 2] * X[:, 3] > 0).astype(float)
Tm = Table.from_numpy(Domain([ContinuousVariable(f"a{i:03d}") for i in range(n_attr)],
                             DiscreteVariable("c", values=("0", "1"))), X, yb)
for nombre, learner in (("LR", LogisticRegressionLearner()),
                        ("RF", RandomForestLearner(n_estimators=30))):
    m = silent(learner, Tm)
    b = Banco(n_attributes=10)
    t0 = time.time()
    b.enviar(m, Tm)
    dt = time.time() - t0
    w = b.w
    check(not w.Error.active and len(w.rows) == 10 and len(w.hidden) == n_attr - 10,
          f"300 atributos ({nombre}): 10 mostrados")
    check(w.rows[0].effect.name in ("a000", "a001"),
          f"300 atributos ({nombre}): el más importante primero ({w.rows[0].effect.name})")
    check(dt < 300, f"300 atributos ({nombre}): {dt:.1f} s")
    print(f"   300 atributos, {nombre}: {dt:.1f} s ({w.analysis.mode})")
    b.cerrar()

# 4d. clase con muchos valores
K = 30
X = rng.normal(size=(1200, 4))
yk = np.clip(((X[:, 0] + 3) / 6 * K).astype(int), 0, K - 1).astype(float)
Tk = Table.from_numpy(Domain([ContinuousVariable(f"v{i}") for i in range(4)],
                             DiscreteVariable("k", values=tuple(f"c{i}" for i in range(K)))),
                      X, yk)
for learner in (LogisticRegressionLearner(), NaiveBayesLearner(), Mo.TreeLearner()):
    m = silent(learner, Tk)
    b = Banco(target_class_name="c15")
    b.enviar(m, Tk)
    w = b.w
    check(not w.Error.active and w.view_ is not None, f"30 clases ({type(m).__name__})")
    check(w.view_.target == 15, "30 clases: clase elegida")
    w.target_class_name = "c3"
    w._target_changed()
    check(w.view_.target == 3, "30 clases: cambiar la clase no recalcula")
    b.cerrar()

# 4e. regresión con objetivo de escala grande
X = rng.normal(size=(800, 3))
yl = 1e9 * (3 + X[:, 0] - 0.5 * X[:, 1]) + 1e7 * rng.normal(size=800)
Tl = Table.from_numpy(Domain([ContinuousVariable(f"z{i}") for i in range(3)],
                             ContinuousVariable("dinero")), X, yl)
for learner in (LinearRegressionLearner(), RandomForestRegressionLearner(n_estimators=30)):
    m = silent(learner, Tl)
    b = Banco()
    b.enviar(m, Tl)
    w = b.w
    check(not w.Error.active, f"escala grande ({type(m).__name__})")
    v = w.view_
    check(abs(v.score(w.values) - w.real) <= (1e-6 * abs(w.real)
                                              if w.analysis.mode == E.EXACT
                                              else 0.5 * abs(w.real)),
          f"escala grande: suma {v.score(w.values)} vs modelo {w.real}")
    check(w._band() is not None, "escala grande: banda de incertidumbre")
    b.cerrar()

# 4f. tabla dispersa, fecha y datos de entrada sin un atributo del modelo
Xs = sp.random(400, 6, density=0.3, random_state=1, format="csr")
ys = (np.asarray(Xs.sum(axis=1)).ravel() > 0.5).astype(float)
Tsp = Table.from_numpy(Domain([ContinuousVariable(f"s{i}") for i in range(6)],
                              DiscreteVariable("c", values=("0", "1"))), Xs, ys)
for learner in (LogisticRegressionLearner(), RandomForestLearner(n_estimators=20)):
    m = silent(learner, Tsp)
    b = Banco()
    b.enviar(m, Tsp)
    check(not b.w.Error.active and b.w.view_ is not None, "tabla dispersa")
    b.cerrar()
tv = TimeVariable("fecha", have_date=True)
Xt = np.column_stack([1.6e9 + rng.uniform(0, 3e7, 300), rng.normal(size=300)])
Tt = Table.from_numpy(Domain([tv, ContinuousVariable("x")], ContinuousVariable("y")),
                      Xt, Xt[:, 1] + (Xt[:, 0] - 1.6e9) / 1e7)
m = silent(RandomForestRegressionLearner(n_estimators=20), Tt)
b = Banco()
b.enviar(m, Tt)
texts = [it.text() for it in b.w.scene.items() if hasattr(it, "text")]
check(any("20" in t and "-" in t for t in texts), "fecha: las marcas se ven como fechas")
b.cerrar()
m = silent(LogisticRegressionLearner(), HEART)
parcial = HEART.transform(Domain(HEART.domain.attributes[:-1], HEART.domain.class_var))
b = Banco()
b.enviar(m, parcial)
check(b.w.view_ is not None and b.w.Information.notes.is_shown()
      and "entrenamiento" in str(b.w.Information.notes),
      "Data sin un atributo: usa los de entrenamiento y lo dice")
b.cerrar()

# ---------------------------------------------------------------------------
print("5. Widget: salidas, ajustes, escalas, informe")
m = silent(LogisticRegressionLearner(), HEART)
b = Banco(n_attributes=5, target_class_name="1")
b.enviar(m, HEART, HEART[10:11])
w = b.w
feats = b.out.get("features")
check(feats is not None and len(feats) == 5, "Features: los 5 mostrados")
contrib = b.out.get("contributions")
check(contrib is not None and len(contrib) == len(HEART.domain.attributes) + 1,
      "Contributions: una fila por atributo y la base")
inst = b.out.get("instance")
check(inst is not None and np.array_equal(inst.X[0], HEART.X[10], equal_nan=True),
      "Instance: los valores de la entrada")
# las contribuciones suman la puntuación menos la media
view = w.view_
total = sum(float(view.component(e, [w.values[e.index]])[0]) for e in view.effects)
check(abs(view.base + total - view.score(w.values)) < 1e-9, "contribuciones + base = puntuación")
pts = contrib.X[:, 0]
pts = pts[np.isfinite(pts)]
check(len(pts) == 5 and np.all(pts >= -1e-9) and np.nanmax(contrib.X[:, 2]) <= 100 + 1e-9,
      "Contributions: puntos entre 0 y 100 en los 5 mostrados")
# la suma de puntos y el pie: puntuación desde los puntos totales
comps, hidden, points, total_pts, score = w._current()
check(abs(w.scale.score_of_total(total_pts) + hidden - score) < 1e-9,
      "puntos totales → puntuación")
# escala en unidades y orden por contribución
w.scale_index = UNITS
w.relayout()
w.sort_index = SORT_INSTANCE
w._sort_changed()
check(len(w.rows) == 5, "orden por contribución a la instancia")
w.show_all = True
w._show_all_changed()
check(len(w.rows) == len(HEART.domain.attributes) and not w.hidden, "mostrar todos")
# guardar y recuperar los valores con el workflow
w.values[0] = 66.0
w._values_changed(final=True)
saved = w.settingsHandler.pack_data(w)
check(saved.get("saved_values", {}).get("age") == 66.0, "los valores se guardan")
b2 = Banco(**{k: saved[k] for k in ("saved_values", "target_class_name")})
b2.enviar(m, HEART)
check(b2.w.values[0] == 66.0, "los valores se recuperan al abrir")
b2.cerrar()
# editar un valor por texto (sin diálogo)
w.values[3] = np.nan
w._values_changed(final=True)
check(w.real is not None, "valor desconocido: el modelo predice")
# informe e imagen
try:
    w.send_report()
    ok = True
except Exception:
    ok = False
    traceback.print_exc()
check(ok, "send_report no falla")
check(w.graph_name == "scene" and w.scene.items(), "hay escena para guardar como imagen")
# quitar el modelo limpia salidas
b.enviar(None, None, None)
check(b.out.get("features") is None and b.out.get("instance") is None, "sin modelo: sin salidas")
# un modelo sin variable objetivo
b.enviar(silent(LinearRegressionLearner(), HOUSING), HEART)
check(not b.w.Error.active, "regresión con Data de otra tabla: usa los de entrenamiento")
b.cerrar()

# ---------------------------------------------------------------------------
print("7. Correcciones de la 0.7.1")
# 7.1 Contributions cuadra con lo que se ve, también con atributos ocultos
for nombre, learner, n_show in (("LR", LogisticRegressionLearner(), 4),
                                ("RF", RandomForestLearner(n_estimators=30), 5)):
    m = silent(learner, HEART)
    b = Banco(n_attributes=n_show, target_class_name="1")
    b.enviar(m, HEART, HEART[3:4])
    w = b.w
    t = b.out["contributions"]
    comps, hidden, points, total_pts, score = w._current()
    pts = t.X[:, 0]
    contrib = t.X[:, 1]
    mostrado = t.get_column("Mostrado")
    check(abs(np.nansum(pts) - total_pts) < 1e-9,
          f"{nombre}: suma de Puntos {np.nansum(pts)} = puntos totales {total_pts}")
    check(abs(contrib.sum() - score) < 1e-9,
          f"{nombre}: base + contribuciones {contrib.sum()} = puntuación {score}")
    check(int(np.nansum(mostrado)) == n_show and np.isnan(pts[mostrado == 0]).all(),
          f"{nombre}: solo los mostrados tienen puntos")
    check(t.metas[-1, 0].startswith("Base"), f"{nombre}: la última fila es la base")
    b.cerrar()

# 7.2 Data sin valores conocidos o con una columna constante: rangos de entrenamiento
m = silent(LogisticRegressionLearner(), HEART)
vacia = HEART.copy()
with vacia.unlocked():
    vacia.X[:] = np.nan
ref = E.analyze(m)
a = E.analyze(m, vacia)
rangos = {e.name: (e.lo, e.hi) for e in a.target_view(1).effects if not e.is_discrete}
ok = {e.name: (e.lo, e.hi) for e in ref.target_view(1).effects if not e.is_discrete}
check(rangos == ok, f"Data sin valores: rangos de entrenamiento {rangos}")
check(any("entrenamiento" in n for n in a.notes + a.warnings), "Data sin valores: se dice")
edad = HEART[HEART.get_column("age") == 54]
a = E.analyze(m, edad)
e_age = [e for e in a.target_view(1).effects if e.name == "age"][0]
check(e_age.lo < 54 < e_age.hi, f"age constante en Data: rango de entrenamiento {e_age.lo}–{e_age.hi}")
m2 = silent(LogisticRegressionLearner(), HEART)
m2.original_data = None
try:
    E.analyze(m2, vacia)
    check(False, "Data sin valores y sin entrenamiento: debe dar error")
except E.NomogramError as exc:
    check("valor conocido" in str(exc), f"Data sin valores y sin entrenamiento: error claro ({exc})")
b = Banco()
b.enviar(m2, vacia)
check(b.w.Error.unsupported.is_shown(), "Data sin valores y sin entrenamiento: el widget da error")
b.cerrar()
# una columna sin valores (las demás sí): rango de entrenamiento y aviso
parcial2 = HEART.copy()
with parcial2.unlocked():
    parcial2.X[:, 0] = np.nan
a = E.analyze(m2, parcial2)
check(a.warnings and "valores conocidos" in " ".join(a.warnings),
      f"una columna sin valores y sin entrenamiento: aviso ({a.warnings})")

# 7.3 ajuste aditivo: las categorías sin datos no tienen efecto inventado
rng3 = np.random.RandomState(5)
cats = tuple(f"p{i}" for i in range(40))
n3 = 3000
codes = np.where(rng3.rand(n3) < 0.9, 0, rng3.randint(1, 30, n3)).astype(float)
x3 = rng3.normal(size=n3)
y3 = ((codes % 3 == 1) + x3 + rng3.normal(scale=0.5, size=n3) > 0.5).astype(float)
dom3 = Domain([DiscreteVariable("pais", values=cats), ContinuousVariable("x")],
              DiscreteVariable("c", values=("0", "1")))
T3 = Table.from_numpy(dom3, np.column_stack([codes, x3]), y3)
m3 = silent(RandomForestLearner(n_estimators=30, random_state=0), T3)
a = E.analyze(m3, T3, method=E.ADDITIVE_FIT, sample_size=300)
v = a.target_view(1)
pais = [e for e in v.effects if e.name == "pais"][0]
presentes = set(np.unique(codes).astype(int))
comp = v.grid_component(pais)
sin_datos = [i for i in range(40) if i not in presentes]
check(pais.unsupported is not None and pais.unsupported[sin_datos].all()
      and not pais.unsupported[list(presentes)].any(),
      "categorías sin datos marcadas")
check(np.isnan(comp[sin_datos]).all() and np.isfinite(comp[list(presentes)]).all(),
      "sin efecto dibujado para las categorías sin datos")
check(abs(float(v.component(pais, [float(sin_datos[0])])[0])) < 1e-12,
      "una categoría sin datos cuenta como la media")
b = Banco(method_index=E.ADDITIVE_FIT, sample_size=300, show_all=True)
b.enviar(m3, T3, Table.from_numpy(dom3, np.array([[float(sin_datos[0]), 0.0]]),
                                   np.array([0.0])))
texts = [it.text() for it in b.w.scene.items() if hasattr(it, "text")]
check(any(t.startswith("sin datos") for t in texts), "el widget indica las categorías sin datos")
check(any("(sin datos)" in t for t in texts), "el valor sin datos se marca")
b.cerrar()

# 7.4 multiclase exacto: la escala de probabilidad solo si es buena en casi todas las filas
ZOO = Table("zoo")
m4 = silent(NaiveBayesLearner(), ZOO)
a = E.analyze(m4)
for t in range(len(ZOO.domain.class_var.values)):
    v = a.target_view(t)
    if v.link.kind == "monotone":
        check(v.fidelity["max_abs"] <= E.MULTI_MAX and v.fidelity["p95"] <= E.MULTI_P95,
              f"zoo clase {t}: escala aproximada con error máx {v.fidelity['max_abs']}")
reptil = list(ZOO.domain.class_var.values).index("reptile")
check(a.target_view(reptil).link.kind == "score", "zoo reptile: sin escala de probabilidad")
from orangecontrib.orange_gualbe_toolbox.widgets.ownomogrampro import fidelity_summary
html, _ = fidelity_summary(a.target_view(reptil))
check("Puntuación exacta" in html and "Nomograma exacto" not in html,
      "multiclase: la cabecera dice que la puntuación es exacta, no la probabilidad")
m5 = silent(LogisticRegressionLearner(), IRIS)
html, _ = fidelity_summary(E.analyze(m5).target_view(0))
check("Escala de probabilidad" in html and "aproximada" in html,
      "multiclase con escala: la cabecera la da por aproximada")

# 7.5 fidelidad de las dos aproximaciones en las mismas filas apartadas
m6 = silent(RandomForestRegressionLearner(n_estimators=30), HOUSING)
a = E.analyze(m6, HOUSING)
hold = [v.holdout for v in a.variants.values()]
check(all(h is not None and np.array_equal(h, hold[0]) for h in hold)
      and 0 < len(hold[0]) < len(a.sample),
      "dependencia parcial y ajuste se miden en las mismas filas apartadas")
check(all(v.fidelity["rows"] == len(hold[0]) for v in
          (E.TargetView(a, var, 0) for var in a.variants.values())),
      "la fidelidad cuenta solo las filas apartadas")

# 7.6 datos de referencia con muy pocas filas
for n_filas in (1, 5):
    m7 = silent(RandomForestLearner(n_estimators=10), HEART)
    a = E.analyze(m7, HEART[:n_filas])
    check(a.n_reference == len(HEART) and any("entrenamiento" in n for n in a.notes),
          f"Data de {n_filas} filas: referencia de entrenamiento")
    m7.original_data = None
    a = E.analyze(m7, HEART[:n_filas])
    check(any("no son fiables" in w for w in a.warnings),
          f"Data de {n_filas} filas sin entrenamiento: aviso")

# 7.7 tiempo con 300 atributos y un bosque de 100 árboles
X7 = rng.normal(size=(1000, 300))
y7 = (X7[:, 0] + np.sin(2 * X7[:, 1]) + 0.5 * X7[:, 2] * X7[:, 3] > 0).astype(float)
T7 = Table.from_numpy(Domain([ContinuousVariable(f"a{i}") for i in range(300)],
                             DiscreteVariable("y", values=("0", "1"))), X7, y7)
m8 = silent(RandomForestLearner(n_estimators=100, random_state=0), T7)
t0 = time.time()
a = E.analyze(m8, T7)
dt = time.time() - t0
print(f"   300 atributos, RF de 100 árboles: {dt:.1f} s, {a.n_pd} filas de fondo")
# límites holgados: solo detectan regresiones grandes, no la carga de la máquina
check(dt < 120, f"300 atributos: {dt:.1f} s")
check(a.n_pd >= 60, f"300 atributos: filas de fondo {a.n_pd}")

# ---------------------------------------------------------------------------
print("8. Correcciones de la 0.7.2")
# 8.1 el resultado no depende del orden de las filas (fondo al azar, no las primeras)
HS = HEART[np.argsort(HEART.Y, kind="stable")]
HR = HEART[np.random.RandomState(9).permutation(len(HEART))]
for nombre, learner in (("RF", RandomForestLearner(n_estimators=30, random_state=0)),
                        ("SVM", SVMLearner(probability=True))):
    m = silent(learner, HEART)
    res = []
    for d in (HEART, HS, HR):
        a = E.analyze(m, d, method=E.PARTIAL_DEPENDENCE)
        v = a.target_view(1)
        res.append((a.n_pd, round(v.fidelity["r2"], 12),
                    tuple(round(E.span(v.grid_component(e)), 12) for e in v.effects)))
    check(res[0] == res[1] == res[2], f"{nombre}: mismo nomograma con las filas reordenadas")
# un fondo pequeño de una tabla ordenada por la clase tiene las dos clases
big = Table.concatenate([HEART] * 20)
big = big[np.argsort(big.Y, kind="stable")]
mb = silent(RandomForestLearner(n_estimators=10, random_state=0), HEART)
a = E.analyze(mb, big, method=E.PARTIAL_DEPENDENCE, sample_size=2000)
check(0.3 < a.real[:, 1].mean() < 0.7, "muestra de una tabla ordenada: representativa")

# 8.2 Naive Bayes con una clase sin filas en el entrenamiento
IRIS2 = IRIS[:100]
m = silent(NaiveBayesLearner(), IRIS2)
a = E.analyze(m, IRIS2)
check(a.mode == E.EXACT and list(a.present) == [True, True, False],
      f"NB con una clase ausente: exacto con las dos clases ({a.mode}, {a.present})")
v = a.target_view(1)
X2 = IRIS2.X[:20]
sc = np.array([v.score(x) for x in X2])
check(np.isfinite(v.base) and np.isfinite(sc).all() and v.link.kind == "logistic",
      "NB con una clase ausente: puntuaciones finitas y probabilidad exacta")
real2 = np.asarray(m(IRIS2[:20], m.Probs))[:, 1]
check(np.allclose(v.link(sc), real2, atol=1e-9), "NB con una clase ausente: P exacta")
b = Banco(target_class_name="Iris-virginica")
b.enviar(m, IRIS2)
check(b.w.Error.no_target.is_shown(), "NB: elegir la clase ausente es un error claro")
b.w.target_class_name = "Iris-versicolor"
b.w._target_changed()
check(not b.w.Error.active and "exacto" in b.w.model_label.text(), "NB: la otra clase va bien")
b.cerrar()
# red de seguridad: un efecto no finito no se presenta como exacto
m = silent(NaiveBayesLearner(), HEART)
m.log_cont_prob[1] = m.log_cont_prob[1].copy()
m.log_cont_prob[1][0, 0] = -np.inf
a = E.analyze(m, HEART)
check(a.mode == E.APPROXIMATE and "no finitos" in a.reason,
      f"NB con un efecto infinito: pasa a aproximado ({a.mode}: {a.reason})")

# 8.3 las filas de categorías raras no cambian las proporciones de la muestra
rng8 = np.random.RandomState(0)
n8 = 10000
cat8 = np.where(rng8.rand(n8) < 0.9, 0, rng8.randint(1, 401, n8)).astype(float)
x8 = rng8.normal(size=n8)
y8 = ((x8 + 1.5 * (cat8 > 0) + 0.3 * rng8.normal(size=n8)) > 0.8).astype(float)
dom8 = Domain([DiscreteVariable("c", values=tuple(f"v{i}" for i in range(401))),
               ContinuousVariable("x")], DiscreteVariable("y", values=("0", "1")))
T8 = Table.from_numpy(dom8, np.column_stack([cat8, x8]), y8)
m8b = silent(RandomForestLearner(n_estimators=20, random_state=0), T8)
a = E.analyze(m8b, T8)
check(abs(np.mean(a.sample[:, 0] > 0) - np.mean(cat8 > 0)) < 0.03,
      f"categorías raras: la muestra mantiene su proporción ({np.mean(a.sample[:, 0] > 0):.3f})")
P8 = np.asarray(m8b(T8, m8b.Probs))[:, 1]
check(abs(a.real[:, 1].mean() - P8.mean()) < 0.03,
      f"categorías raras: P media de la muestra {a.real[:, 1].mean():.3f} ≈ datos {P8.mean():.3f}")
fit = E.TargetView(a, a.variants[("fit", "logit")], 1)
c8 = [e for e in fit.effects if e.name == "c"][0]
presentes8 = set(np.unique(cat8).astype(int))
check(c8.unsupported is None or not c8.unsupported[list(presentes8)].any(),
      "categorías raras con datos: tienen efecto estimado en el ajuste")

# 8.4 el tamaño del fondo no depende del reloj
m9 = silent(RandomForestLearner(n_estimators=20, random_state=0), HEART)
n_pds = {E.analyze(m9, HEART, method=E.PARTIAL_DEPENDENCE).n_pd for _ in range(3)}
a = E.analyze(m9, HEART, method=E.PARTIAL_DEPENDENCE)
rows = sum((len(e.var.values) if e.var.is_discrete else
            len(e.grid)) + 1 for e in a.target_view(1).effects)
esperado = min(int(round(0.7 * len(a.sample))), max(E.MIN_PD_ROWS, E.PD_ROW_BUDGET // rows))
# (heart tiene 13 atributos: el presupuesto es el básico)
check(n_pds == {esperado}, f"fondo determinista: {n_pds} (esperado {esperado})")

# ---------------------------------------------------------------------------
print("9. Correcciones de la 0.7.3")
# 9.1 cancelar, cambiar o cerrar a mitad de la dependencia parcial tarda < 1 s
rng9 = np.random.RandomState(4)
X9 = rng9.normal(size=(4000, 12))
y9 = (np.sin(X9[:, 0] * 2) + X9[:, 1] * X9[:, 2] + rng9.normal(scale=0.5, size=4000) > 0)
T9 = Table.from_numpy(Domain([ContinuousVariable(f"s{i}") for i in range(12)],
                             DiscreteVariable("y", values=("0", "1"))), X9, y9.astype(float))
svm9 = silent(SVMLearner(probability=True), T9)
tree9 = silent(Mo.TreeLearner(), T9)


def bombear(seg):
    t = time.time()
    while time.time() - t < seg:
        app.processEvents()
        time.sleep(0.005)


b = Banco()
b.w.set_model(svm9)
b.w.set_data(T9)
b.w.set_instance(None)
b.w.handleNewSignals()
bombear(1.5)
ocupado = b.w.task is not None
t0 = time.time()
b.w.set_model(tree9)
b.w.handleNewSignals()
b.esperar()
dt = time.time() - t0
print(f"   cambiar de modelo a mitad del cálculo: {dt:.2f} s")
check(ocupado, "la prueba de cancelación empieza con el cálculo en marcha")
check(dt < 1.0, f"cambiar de modelo a mitad: {dt:.2f} s")
check(b.w.analysis is not None and b.w.analysis.model is tree9, "el resultado es el del modelo nuevo")
b.w.set_model(svm9)
b.w.handleNewSignals()
bombear(1.5)
ocupado = b.w.task is not None
t0 = time.time()
b.w.onDeleteWidget()
dt = time.time() - t0
print(f"   cerrar a mitad del cálculo: {dt:.2f} s")
check(ocupado, "la prueba de cierre empieza con el cálculo en marcha")
check(dt < 1.0, f"cerrar a mitad: {dt:.2f} s")
b.w.deleteLater()
app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
# el reparto en trozos no cambia el resultado
m10 = silent(RandomForestLearner(n_estimators=20, random_state=0), HEART)
a1 = E.analyze(m10, HEART, method=E.PARTIAL_DEPENDENCE)
orig_tune = E.Predictor.tune
E.Predictor.tune = lambda self, per_row: setattr(self, "chunk", 777)
a2 = E.analyze(m10, HEART, method=E.PARTIAL_DEPENDENCE)
E.Predictor.tune = orig_tune
v1, v2 = a1.target_view(1), a2.target_view(1)
check(all(np.array_equal(v1.grid_component(e1), v2.grid_component(e2))
          for e1, e2 in zip(v1.effects, v2.effects)) and v1.base == v2.base,
      "el tamaño de los trozos no cambia el nomograma")

# 9.2 Data sin valores con un modelo aproximado: datos de entrenamiento y aviso
HU = HEART.copy()
with HU.unlocked():
    HU.X[:] = np.nan
for seed in range(3):
    mr = silent(RandomForestLearner(n_estimators=10, random_state=seed), HEART)
    b = Banco()
    b.enviar(mr, HU)
    w = b.w
    check(not w.Error.active and w.view_ is not None and w.rows,
          f"RF {seed} con Data sin valores: se dibuja ({[str(m) for m in w.Error.active]})")
    check(w.Warning.reference.is_shown() and "entrenamiento" in str(w.Warning.reference),
          f"RF {seed} con Data sin valores: aviso")
    check(w.analysis.n_reference == len(HEART), "la referencia son los de entrenamiento")
    b.cerrar()
mr.original_data = None
b = Banco()
b.enviar(mr, HU)
check(b.w.Error.unsupported.is_shown() and "valor conocido" in str(b.w.Error.unsupported),
      "RF con Data sin valores y sin entrenamiento: error claro")
b.cerrar()

# 9.3 la clave del orden distingue «?» de cualquier valor
Xk = np.array([[np.nan, 1.0], [-0.123456789, 1.0], [0.0, 1.0]])
rs = np.random.RandomState(12345)
r, q = rs.normal(size=2), rs.normal(size=2)
claves = np.nan_to_num(Xk, nan=0.0) @ r + np.isnan(Xk) @ q
check(len(set(np.round(claves, 12))) == 3, "«?», el centinela antiguo y 0 tienen claves distintas")
XX = np.array([[np.nan, 1.0], [0.0, 1.0]] * 50 + [[float(i), 2.0] for i in range(200)])
o1 = E.canonical_order(XX)
perm = np.random.RandomState(2).permutation(len(XX))
o2 = E.canonical_order(XX[perm])
check(np.array_equal(XX[o1], XX[perm][o2], equal_nan=True),
      "orden canónico igual con desconocidos y ceros, en cualquier orden de entrada")

# ---------------------------------------------------------------------------
print("6. Tiempos con una tabla mediana")
X = rng.normal(size=(5000, 20))
ym = X[:, 0] + np.sin(X[:, 1]) + X[:, 2] * X[:, 3] + rng.normal(scale=0.3, size=5000)
Tmed = Table.from_numpy(Domain([ContinuousVariable(f"m{i:02d}") for i in range(20)],
                               ContinuousVariable("y")), X, ym)
Tmedc = Table.from_numpy(Domain(Tmed.domain.attributes,
                                DiscreteVariable("c", values=("0", "1"))), X,
                         (ym > 0).astype(float))
for nombre, learner, data in (("RF clasificación", RandomForestLearner(), Tmedc),
                              ("RF regresión", RandomForestRegressionLearner(), Tmed),
                              ("Red neuronal clasificación", NNClassificationLearner(), Tmedc),
                              ("Red neuronal regresión", NNRegressionLearner(), Tmed)):
    model = silent(learner, data)
    t0 = time.time()
    a = E.analyze(model, data)
    dt = time.time() - t0
    view = a.target_view(1 if a.kind == "classification" else 0)
    print(f"   {nombre}: {dt:.1f} s, {view.variant.method}-{view.variant.space}, "
          f"R² = {view.fidelity['r2']:.3f}, filas de dependencia parcial {a.n_pd}")
    check(dt < 120, f"{nombre}: tiempo {dt:.1f} s")

print("\nFALLOS:", fallos)
if excepciones:
    print("EXCEPCIONES:\n" + "\n".join(excepciones))
    fallos += len(excepciones)
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if fallos else 0)
