# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Prueba sin interfaz gráfica del widget SMOTE.

Se ejecuta con el Python de Orange (scripts/probar.sh lo hace con PYTHONPATH
apuntando al repo) y sale con 1 si alguna comprobación falla. Recorre la matriz
completa: algoritmo × tipo de datos × forma de la clase × estrategia.
"""
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QEvent, QPointF, Qt
from AnyQt.QtGui import QMouseEvent
from AnyQt.QtWidgets import QApplication
from Orange.data import (
    ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table,
    TimeVariable,
)
from Orange.preprocess import Impute

from orangecontrib.orange_gualbe_toolbox.widgets import owsmote
from orangecontrib.orange_gualbe_toolbox.widgets.owsmote import (
    EQUAL, MINORITY, NOT_MINORITY, OWSmote, PERCENT, sampling_targets,
)

warnings.simplefilter("ignore", DeprecationWarning)
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []
sys.excepthook = lambda *e: excepciones.append("".join(traceback.format_exception(*e)))


def check(ok, msg):
    global fallos
    if not ok:
        print("FAIL " + msg)
    fallos += not ok
    return ok


class Banco:
    """Un widget real, con la salida capturada."""

    def __init__(self):
        self.w = OWSmote()
        self.salida = "sin enviar"
        self.w.Outputs.data.send = self._recibir
        self.w.k_neighbors = 5

    def _recibir(self, tabla):
        self.salida = tabla

    def esperar(self):
        inicio = time.time()
        while self.w.task is not None and time.time() - inicio < 120:
            app.processEvents()
            time.sleep(0.001)
        app.processEvents()
        check(self.w.task is None, "el cálculo termina")

    def poner(self, datos):
        self.salida = "sin enviar"
        self.w.set_data(datos)
        self.esperar()
        self.w.chart.grab()
        return self.salida

    def ajustar(self, **opciones):
        self.salida = "sin enviar"
        for nombre, valor in opciones.items():
            setattr(self.w, nombre, valor)
        self.w._on_settings_changed()
        self.esperar()
        self.w.chart.grab()
        return self.salida

    def errores(self):
        return [m for m in vars(type(self.w.Error)) if isinstance(
            getattr(self.w.Error, m, None), type(self.w.Error.empty))
            and getattr(self.w.Error, m).is_shown()]


def clase(n_valores):
    return DiscreteVariable("clase", values=[f"c{i}" for i in range(n_valores)])


def recuentos(tabla):
    return np.bincount(tabla.Y.astype(int), minlength=len(tabla.domain.class_var.values))


def tabla(tipo, conteos, con_metas=False, con_pesos=False, disperso=False, semilla=0):
    rng = np.random.RandomState(semilla)
    y = np.concatenate([np.full(c, i) for i, c in enumerate(conteos)]).astype(float)
    n = len(y)
    centro = y[:, None]
    numericos = rng.normal(size=(n, 3)) + centro
    # Las categorías dependen de la clase para que la distancia entre filas importe.
    categoricos = (rng.randint(0, 3, size=(n, 2)) + centro.astype(int)) % 3
    categoricos = categoricos.astype(float)
    num_vars = [ContinuousVariable(f"n{i}") for i in range(3)]
    cat_vars = [DiscreteVariable(f"d{i}", values=("x", "y", "z")) for i in range(2)]
    if tipo == "numérico":
        X, atributos = numericos, num_vars
    elif tipo == "un atributo":
        X, atributos = numericos[:, :1], num_vars[:1]
    elif tipo == "fecha":
        X, atributos = numericos, [TimeVariable("t")] + num_vars[1:]
        X[:, 0] = 1.6e9 + X[:, 0] * 86400
    elif tipo == "mixto":
        X, atributos = np.c_[numericos, categoricos], num_vars + cat_vars
    elif tipo == "mixto intercalado":
        X = np.c_[categoricos[:, 0], numericos[:, 0], categoricos[:, 1], numericos[:, 1]]
        atributos = [cat_vars[0], num_vars[0], cat_vars[1], num_vars[1]]
    elif tipo == "categórico":
        X, atributos = categoricos, cat_vars
    else:
        raise ValueError(tipo)
    metas, meta_vars = None, ()
    if con_metas:
        meta_vars = (StringVariable("nombre"), ContinuousVariable("edad"),
                     DiscreteVariable("grupo", values=("g", "h")))
        metas = np.empty((n, 3), dtype=object)
        metas[:, 0] = [f"fila{i}" for i in range(n)]
        metas[:, 1] = rng.normal(size=n)
        metas[:, 2] = rng.randint(0, 2, size=n).astype(float)
    dominio = Domain(atributos, clase(len(conteos)), meta_vars)
    pesos = rng.uniform(0.5, 2, size=n) if con_pesos else None
    t = Table.from_numpy(dominio, X, y, metas=metas, W=pesos)
    t.name = "mis datos"
    t.attributes = {"origen": "test"}
    return t.to_sparse() if disperso else t


def comprobar_salida(ent, sal, esperado, algoritmo, etiqueta):
    """Comprobaciones que valen para cualquier fila de la matriz."""
    n = len(ent)
    d = ent.domain
    if not check(isinstance(sal, Table), etiqueta + ": hay tabla de salida"):
        return
    check(sal.domain is d or sal.domain == d, etiqueta + ": conserva el dominio")
    check(list(recuentos(sal)) == list(esperado),
          f"{etiqueta}: recuentos {list(recuentos(sal))} != {list(esperado)}")
    check(len(sal) == sum(esperado), etiqueta + ": filas totales")
    check(sal.name == "mis datos" and sal.attributes.get("origen") == "test",
          etiqueta + ": conserva nombre y atributos de la tabla")

    denso = lambda m: m.toarray() if sp.issparse(m) else m
    check(sp.issparse(sal.X) == sp.issparse(ent.X), etiqueta + ": conserva la dispersión")
    Xs, Xe = denso(sal.X), denso(ent.X)
    check(np.array_equal(Xs[:n], Xe), etiqueta + ": filas originales intactas (X)")
    check(np.array_equal(sal.Y[:n], ent.Y), etiqueta + ": filas originales intactas (Y)")
    check(np.array_equal(sal.ids[:n], ent.ids), etiqueta + ": conserva los ids originales")
    check(len(set(sal.ids)) == len(sal), etiqueta + ": ids únicos")
    check(np.isfinite(Xs).all(), etiqueta + ": sin NaN en los atributos")

    nuevos = Xs[n:]
    y_nuevos = sal.Y[n:].astype(int)
    for j, var in enumerate(d.attributes):
        columna = nuevos[:, j]
        if var.is_discrete:
            check(np.array_equal(columna, np.round(columna))
                  and set(columna) <= set(range(len(var.values))),
                  f"{etiqueta}: {var.name} solo toma valores válidos")
        else:
            for valor in np.unique(y_nuevos):
                orig = Xe[ent.Y.astype(int) == valor, j]
                filas = columna[y_nuevos == valor]
                check(filas.min() >= orig.min() - 1e-6 and filas.max() <= orig.max() + 1e-6,
                      f"{etiqueta}: {var.name} interpola dentro del rango de la clase")

    if d.metas:
        check(np.array_equal(sal.metas[:n].astype(str), ent.metas.astype(str)),
              etiqueta + ": filas originales intactas (metas)")
        for j, var in enumerate(d.metas):
            columna = sal.metas[n:, j]
            if var.is_string:
                check(all(v == "" for v in columna), etiqueta + ": metas de texto vacías")
            else:
                check(np.isnan(columna.astype(float)).all(),
                      etiqueta + ": metas desconocidas")
    if ent.has_weights():
        check(np.array_equal(sal.W[:n], ent.W), etiqueta + ": pesos originales")
        check((sal.W[n:] == 1).all(), etiqueta + ": peso 1 en las sintéticas")
    check(algoritmo in banco.w.algorithm_label.text(), etiqueta + ": indica " + algoritmo)


ESTRATEGIAS = {
    "igualar": (EQUAL, 80),
    "minoritaria": (MINORITY, 80),
    "no minoritaria": (NOT_MINORITY, 80),
    "porcentaje 80": (PERCENT, 80),
    "porcentaje 20": (PERCENT, 20),
    "porcentaje 100": (PERCENT, 100),
}

ESPERADO = {
    (90, 22, 8): {"igualar": (90, 90, 90), "minoritaria": (90, 22, 90),
                  "no minoritaria": (90, 90, 8), "porcentaje 80": (90, 72, 72),
                  "porcentaje 20": (90, 22, 18), "porcentaje 100": (90, 90, 90)},
    (40, 12): {"igualar": (40, 40), "minoritaria": (40, 40),
               "no minoritaria": (40, 12), "porcentaje 80": (40, 32),
               "porcentaje 20": (40, 12), "porcentaje 100": (40, 40)},
    # Desordenada: la mayoritaria no es la primera y la minoritaria está en medio.
    (10, 30, 7, 15): {"igualar": (30, 30, 30, 30), "minoritaria": (10, 30, 30, 15),
                      "no minoritaria": (30, 30, 7, 30),
                      "porcentaje 80": (24, 30, 24, 24),
                      "porcentaje 20": (10, 30, 7, 15),
                      "porcentaje 100": (30, 30, 30, 30)},
}

MATRIZ = [
    ("numérico", "SMOTE", {}),
    ("numérico", "SMOTE", {"disperso": True}),
    ("numérico", "SMOTE", {"con_metas": True, "con_pesos": True}),
    ("un atributo", "SMOTE", {}),
    ("fecha", "SMOTE", {}),
    ("mixto", "SMOTENC", {}),
    ("mixto", "SMOTENC", {"disperso": True}),
    ("mixto", "SMOTENC", {"con_metas": True, "con_pesos": True}),
    ("mixto intercalado", "SMOTENC", {}),
    ("categórico", "SMOTEN", {}),
    ("categórico", "SMOTEN", {"disperso": True}),
    ("categórico", "SMOTEN", {"con_metas": True, "con_pesos": True}),
]

banco = Banco()
casos = 0
for tipo, algoritmo, opciones in MATRIZ:
    for conteos, por_estrategia in ESPERADO.items():
        for nombre, (estrategia, porcentaje) in ESTRATEGIAS.items():
            casos += 1
            etiqueta = f"{tipo} {opciones} {conteos} {nombre}"
            banco.w.strategy_index = estrategia
            banco.w.target_percent = porcentaje
            ent = tabla(tipo, conteos, **opciones)
            sal = banco.poner(ent)
            check(not banco.errores(), f"{etiqueta}: sin errores ({banco.errores()})")
            if por_estrategia[nombre] == conteos:
                check(sal is ent, etiqueta + ": sin nada que generar reenvía la entrada")
                check(banco.w.Information.nothing_to_add.is_shown(),
                      etiqueta + ": avisa de que no hay nada que añadir")
            else:
                comprobar_salida(ent, sal, por_estrategia[nombre], algoritmo, etiqueta)
                check(list(banco.w.chart.after) == list(por_estrategia[nombre]),
                      etiqueta + ": el gráfico muestra el resultado")
print(f"matriz: {casos} casos")
banco.w.strategy_index = EQUAL

# Las estrategias coinciden con las de imbalanced-learn.
from imblearn.over_sampling import SMOTE  # noqa: E402

rng = np.random.RandomState(1)
for conteos in ((90, 22, 8), (40, 12), (10, 30, 7, 15)):
    y = np.concatenate([np.full(c, i) for i, c in enumerate(conteos)])
    X = rng.normal(size=(len(y), 3))
    for propia, suya in ((EQUAL, "not majority"), (EQUAL, "all"), (EQUAL, "auto"),
                         (MINORITY, "minority"), (NOT_MINORITY, "not minority")):
        _, y2 = SMOTE(sampling_strategy=suya, random_state=0).fit_resample(X, y)
        check(list(sampling_targets(np.array(conteos), propia, 80))
              == list(np.bincount(y2, minlength=len(conteos))),
              f"la estrategia {suya} coincide con imbalanced-learn en {conteos}")

# Reproducibilidad y efecto de los parámetros.
ent = tabla("mixto", (40, 12))
a = banco.poner(ent)
b = banco.poner(ent)
check(np.array_equal(a.X, b.X), "misma semilla: mismo resultado")
c = banco.ajustar(random_seed=7)
check(not np.array_equal(a.X, c.X), "otra semilla: otro resultado")
d = banco.ajustar(random_seed=42, k_neighbors=2)
check(not np.array_equal(a.X, d.X), "otro k: otro resultado")
banco.ajustar(k_neighbors=5)

# Mensajes.
sal = banco.poner(None)
check(sal is None and not banco.errores(), "sin datos: salida vacía y sin errores")

iris = Table("iris")
sal = banco.poner(iris)
check(sal is iris and banco.w.Information.balanced.is_shown(),
      "ya balanceado: reenvía la entrada y lo indica")
check(banco.w.chart.after is not None and list(banco.w.chart.after) == [50, 50, 50],
      "ya balanceado: gráfico con las barras iguales")

sal = banco.poner(iris[:0])
check(sal is None and banco.errores() == ["empty"], "tabla vacía")

dominio = Domain(iris.domain.attributes)
sal = banco.poner(iris.transform(dominio))
check(sal is None and banco.errores() == ["no_class"], "sin clase")

sal = banco.poner(Table("housing"))
check(sal is None and banco.errores() == ["continuous_class"], "clase continua")

dos = Domain(iris.domain.attributes[:2],
             [clase(2), DiscreteVariable("otra", values=("a", "b"))])
sal = banco.poner(Table.from_numpy(dos, np.zeros((6, 2)), np.zeros((6, 2))))
check(sal is None and banco.errores() == ["many_classes"], "varias variables objetivo")

solo_clase = Domain([], clase(2))
sal = banco.poner(Table.from_numpy(solo_clase, np.zeros((6, 0)), np.array([0, 0, 0, 0, 1, 1.])))
check(sal is None and banco.errores() == ["no_attributes"], "sin atributos")

una = tabla("numérico", (30, 0))
sal = banco.poner(una)
check(sal is None and banco.errores() == ["single_class"]
      and "c0" in str(banco.w.Error.single_class),
      "una sola clase aunque el dominio declare dos")

ausente = tabla("numérico", (30, 10, 0))
sal = banco.poner(ausente)
check(list(recuentos(sal)) == [30, 30, 0] and not banco.errores(),
      "una clase declarada sin filas se queda en cero")

con_nan = tabla("numérico", (40, 12))
with con_nan.unlocked():
    con_nan.X[3, 1] = np.nan
sal = banco.poner(con_nan)
check(sal is None and banco.errores() == ["missing_attributes"]
      and "Impute" in str(banco.w.Error.missing_attributes),
      "atributo perdido: explica qué hacer")
check(banco.w.chart.before is not None and banco.w.chart.after is None,
      "atributo perdido: el gráfico muestra solo el antes")

con_inf = tabla("numérico", (40, 12))
with con_inf.unlocked():
    con_inf.X[3, 1] = np.inf
sal = banco.poner(con_inf)
check(sal is None and banco.errores() == ["missing_attributes"], "infinito en atributo")

cat_nan = tabla("categórico", (40, 12))
with cat_nan.unlocked():
    cat_nan.X[3, 0] = np.nan
sal = banco.poner(cat_nan)
check(sal is None and banco.errores() == ["missing_attributes"], "categórico perdido")

disperso_nan = tabla("numérico", (40, 12)).to_sparse()
with disperso_nan.unlocked():
    disperso_nan.X.data[5] = np.nan
sal = banco.poner(disperso_nan)
check(sal is None and banco.errores() == ["missing_attributes"], "perdido en tabla dispersa")

clase_nan = tabla("numérico", (40, 12))
with clase_nan.unlocked():
    clase_nan.Y[3] = np.nan
sal = banco.poner(clase_nan)
check(sal is None and banco.errores() == ["missing_class"]
      and "Select Rows" in str(banco.w.Error.missing_class), "clase perdida")

toda_nan = tabla("numérico", (10, 2))
with toda_nan.unlocked():
    toda_nan.Y[:] = np.nan
sal = banco.poner(toda_nan)
check(sal is None and banco.errores() == ["missing_class"], "toda la clase perdida")

# Clase con pocas instancias: frontera exacta k+1, y solo cuenta si se sobremuestrea.
for tipo in ("numérico", "mixto", "categórico"):
    ent = tabla(tipo, (40, 6))
    sal = banco.poner(ent)
    check(list(recuentos(sal)) == [40, 40] and not banco.errores(),
          f"{tipo}: con k+1 instancias funciona")
    ent = tabla(tipo, (40, 5))
    sal = banco.poner(ent)
    check(sal is None and banco.errores() == ["too_few"], f"{tipo}: con k instancias avisa")
    texto = str(banco.w.Error.too_few)
    check("c1" in texto and "k a 4" in texto, f"{tipo}: explica cómo resolverlo")
    sal = banco.ajustar(k_neighbors=4)
    check(list(recuentos(sal)) == [40, 40] and not banco.errores(),
          f"{tipo}: bajando k se resuelve")
    banco.ajustar(k_neighbors=5)

ent = tabla("numérico", (40, 1))
sal = banco.poner(ent)
check(sal is None and banco.errores() == ["single_instance"], "clase con una sola instancia")

# La clase pequeña no impide nada si la estrategia no la toca.
ent = tabla("numérico", (40, 20, 3))
banco.w.strategy_index = NOT_MINORITY
sal = banco.poner(ent)
check(list(recuentos(sal)) == [40, 40, 3] and not banco.errores(),
      "la clase pequeña no molesta si no se sobremuestrea")
banco.w.strategy_index = EQUAL

# Pesos: se avisa de que SMOTE no los usa.
banco.poner(tabla("numérico", (40, 12), con_pesos=True))
check(banco.w.Warning.weights_ignored.is_shown(), "avisa de que los pesos no se usan")
banco.poner(tabla("numérico", (40, 12)))
check(not banco.w.Warning.weights_ignored.is_shown(), "sin pesos no avisa")

# Datos reales.
titanic = Table("titanic")
sal = banco.poner(titanic)
check(list(recuentos(sal)) == [1490, 1490], "titanic: balancea con SMOTEN")
check("SMOTEN" in banco.w.algorithm_label.text(), "titanic: usa SMOTEN")

zoo = Table("zoo")
sal = banco.poner(zoo)
check(sal is None and banco.errores() == ["too_few"]
      and "amphibian" in str(banco.w.Error.too_few), "zoo: la clase pequeña se nombra")
sal = banco.ajustar(k_neighbors=3)
check(set(recuentos(sal)) == {41}, "zoo: con k=3 balancea las 7 clases y conserva las metas")
check(all(v == "" for v in sal.metas[len(zoo):, 0]), "zoo: el nombre queda vacío")
banco.ajustar(k_neighbors=5)

corazon = Table("heart_disease")
sal = banco.poner(corazon)
check(sal is None and banco.errores() == ["missing_attributes"], "heart_disease: valores perdidos")
sal = banco.poner(Impute()(corazon))
check(len(set(recuentos(sal))) == 1 and "SMOTENC" in banco.w.algorithm_label.text(),
      "heart_disease imputado: balancea con SMOTENC")

# Aplicación manual: vista previa sin enviar nada hasta pulsar Aplicar.
ent = tabla("numérico", (40, 12))
banco.poner(ent)
banco.salida = "sin enviar"
banco.w.auto_apply = False
banco.ajustar(strategy_index=PERCENT, target_percent=50)
check(banco.salida == "sin enviar", "sin aplicación automática no envía")
check(banco.w.chart.pending and list(banco.w.chart.after) == [40, 20],
      "sin aplicación automática muestra una vista previa")
banco.w.commit.now()
banco.esperar()
check(list(recuentos(banco.salida)) == [40, 20] and not banco.w.chart.pending,
      "al aplicar envía el resultado previsto")
banco.w.auto_apply = True
banco.ajustar(strategy_index=EQUAL)

# Segundo plano: no bloquea, y una entrada nueva descarta el cálculo anterior.
grande = tabla("numérico", (20000, 2000))
banco.salida = "sin enviar"
banco.w.set_data(grande)
check(banco.w.task is not None and banco.salida == "sin enviar",
      "el cálculo va en segundo plano")
pequeña = tabla("numérico", (40, 12))
banco.w.set_data(pequeña)
banco.esperar()
time.sleep(0.2)
app.processEvents()
check(len(banco.salida) == 80, "una entrada nueva descarta el cálculo anterior")

# Errores inesperados: mensaje en el widget, sin excepción.
original = owsmote.run_smote


def falla(*_a, **_k):
    raise RuntimeError("fallo simulado")


def sin_libreria(*_a, **_k):
    raise ImportError("sin imblearn")


owsmote.run_smote = falla
sal = banco.poner(tabla("numérico", (40, 12)))
check(sal is None and banco.errores() == ["failed"], "un fallo interno se muestra en el widget")
owsmote.run_smote = sin_libreria
sal = banco.poner(tabla("numérico", (40, 12)))
check(sal is None and banco.errores() == ["no_library"], "sin imbalanced-learn lo explica")
owsmote.run_smote = original

# Gráfico: muchas clases, nombres largos, pistas y dibujo en cada estado.
muchas = tabla("numérico", (40, 12))
nombres = [f"clase con un nombre muy largo número {i}" for i in range(30)]
muchas_var = DiscreteVariable("clase", values=nombres)
y = np.arange(600) % 30.0
y[y == 29] = 0
ent = Table.from_numpy(Domain(muchas.domain.attributes, muchas_var),
                       np.random.RandomState(0).normal(size=(600, 3)), y)
sal = banco.poner(ent)
check(sal is not None, "30 clases: genera salida")
banco.w.resize(800, 500)
imagen = banco.w.grab()
check(not imagen.isNull(), "30 clases: se dibuja")
check(banco.w.chart.minimumHeight() > 30 * banco.w.chart.MIN_ROW,
      "30 clases: el gráfico crece y se desplaza")
banco.poner(tabla("mixto", (40, 12)))
evento = QMouseEvent(QEvent.MouseMove, QPointF(40, banco.w.chart._rows_top + 2),
                     Qt.NoButton, Qt.NoButton, Qt.NoModifier)
banco.w.chart.mouseMoveEvent(evento)
check("c0" in banco.w.chart.toolTip() and "Sintéticas" in banco.w.chart.toolTip(),
      "la pista de la barra muestra los recuentos")
check("52 filas → 80 filas" in banco.w.chart.headline(), "el titular resume el cambio")

# Una entrada nula tras un resultado limpia la salida.
sal = banco.poner(None)
check(sal is None and banco.w.chart.before is None, "al desconectar se limpia todo")

banco.w.onDeleteWidget()
check(not excepciones, "sin excepciones sin capturar:\n" + "\n".join(excepciones))

print("\nFALLOS:", fallos)
# Salida sin el desmontaje de Qt por el recolector de Python, que a veces acaba en SIGBUS (código 138)
# aunque todas las comprobaciones hayan pasado.
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if fallos else 0)
