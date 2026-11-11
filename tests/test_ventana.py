"""Pruebas dirigidas de la validación por ventana deslizante de Test & Score Pro.

Script ejecutable con el Python de Orange; sale con 1 si alguna comprobación falla. Cubre
cada opción de la ventana y sus combinaciones, la ausencia de fugas de datos (con un modelo
y un preprocesador trampa), Diebold-Mariano frente a una implementación de referencia, el
multi-horizonte, las salidas, el informe, la robustez y el estado guardado.
"""
import os
import re
import sys
import tempfile
import time
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from scipy import stats

from AnyQt.QtWidgets import QApplication
from Orange.base import Learner, Model
from Orange.classification import LogisticRegressionLearner, MajorityLearner, TreeLearner
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table, TimeVariable
from Orange.preprocess.preprocess import Preprocess
from Orange.regression import LinearRegressionLearner, MeanLearner, TreeLearner as TreeRegression
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_engine as engine
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_pdf as pdf
from orangecontrib.orange_gualbe_toolbox.widgets._tsp_report import EXPLANATIONS as builder_explicaciones
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_window as tw
from orangecontrib.orange_gualbe_toolbox.widgets.owtestscorepro import OWTestScorePro

try:
    from AnyQt.QtPdf import QPdfDocument
except ImportError:  # pragma: no cover
    QPdfDocument = None

warnings.simplefilter("ignore")
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []
DIR = tempfile.mkdtemp(prefix="test-ventana-")
DIA = 86400.0
ORIGEN = 1577836800.0  # 2020-01-01


def _anotar(*error):
    texto = "".join(traceback.format_exception(*error))
    excepciones.append(texto)
    print("EXCEPCIÓN\n" + texto)


sys.excepthook = _anotar


def check(ok, msg):
    global fallos
    if not ok:
        fallos += 1
        print("FALLO:", msg)
    else:
        print("ok:", msg)


FECHA = TimeVariable("fecha", have_date=1)


def serie(n=100, objetivos=1, paso=DIA, repetidas=0, desorden=False, sin_fecha=0, semilla=0,
          discreta=False, fechas=None):
    """Serie con dos retardos como atributos, `objetivos` horizontes y la fecha como meta."""
    r = np.random.RandomState(semilla)
    base = np.cumsum(r.normal(size=n + objetivos + 2)) + 50
    X = np.column_stack([base[1:n + 1], base[:n]])
    Y = np.column_stack([base[2 + h:n + 2 + h] for h in range(objetivos)])
    if discreta:
        Y = (np.diff(np.column_stack([X[:, 0], Y]), axis=1) > 0).astype(float)
        clases = [DiscreteVariable(f"sube+{h + 1}", values=("no", "sí")) for h in range(objetivos)]
    else:
        clases = [ContinuousVariable(f"y+{h + 1}") for h in range(objetivos)]
    t = ORIGEN + np.arange(n) * paso if fechas is None else np.asarray(fechas, dtype=float)
    if repetidas:
        t = ORIGEN + (np.arange(n) // (repetidas + 1)) * paso
    if sin_fecha:
        t = t.copy()
        t[r.choice(n, sin_fecha, replace=False)] = np.nan
    orden = r.permutation(n) if desorden else np.arange(n)
    dominio = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")], clases, [FECHA])
    tabla = Table.from_numpy(dominio, X[orden], Y[orden] if objetivos > 1 else Y[orden].ravel(),
                             metas=t[orden][:, None])
    tabla.name = "serie"
    return tabla


def plan(data, **opciones):
    opciones.setdefault("time_var", FECHA)
    return tw.plan_windows(data, tw.WindowSpec(**opciones))


def tiempos(p, lo, hi):
    return p.times[lo:hi]


# ------------------------------------------------------------ plan de ventanas ----

datos = serie(100)
p = plan(datos, initial_mode=tw.INITIAL_PERCENT, initial_percent=30, test_size=10)
check(not p.error and p.iterations[0].train_lo == 0 and p.iterations[0].train_hi == 30,
      "entrenamiento inicial en porcentaje: 30 % de 100 filas")
check([it.test_lo for it in p.iterations] == [30, 40, 50, 60, 70, 80, 90] and
      all(it.n_test == 10 for it in p.iterations), "test justo después del entrenamiento y de su tamaño")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=25, test_size=10)
check(p.iterations[0].n_train == 25 and len(p.iterations) == 7, "entrenamiento inicial en filas")
corte = ORIGEN + 39 * DIA + 3600  # a media mañana del día 40
p = plan(datos, initial_mode=tw.INITIAL_DATE, initial_date=corte, test_size=10)
primera = p.iterations[0]
check(p.times[primera.train_hi - 1] <= corte < p.times[primera.train_hi] and primera.n_train == 40,
      "entrenamiento inicial hasta una fecha (incluida)")

p = plan(datos, start_mode=tw.START_ROW, start_row=11, initial_mode=tw.INITIAL_ROWS, initial_rows=20, test_size=10)
check(all(it.train_lo == 10 for it in p.iterations) and p.unused_counts()["before"] == 10,
      "inicio desde la fila N: las anteriores quedan fuera")
p = plan(datos, start_mode=tw.START_DATE, start_date=ORIGEN + 20 * DIA, initial_mode=tw.INITIAL_ROWS,
         initial_rows=20, test_size=10)
check(p.iterations[0].train_lo == 20 and p.times[p.iterations[0].train_lo] >= ORIGEN + 20 * DIA,
      "inicio desde una fecha")
p = plan(datos, start_mode=tw.START_DATE, start_date=ORIGEN + 20.5 * DIA, initial_mode=tw.INITIAL_ROWS,
         initial_rows=20, test_size=10)
check(p.iterations[0].train_lo == 21, "inicio desde una fecha entre dos filas: la primera posterior")

p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, advance=tw.EXPANDING)
check(all(it.train_lo == 0 for it in p.iterations) and
      [it.train_hi for it in p.iterations] == [30, 40, 50, 60, 70, 80, 90], "avance creciente con inicio fijo")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, advance=tw.ROLLING)
check(all(it.n_train == 30 for it in p.iterations) and [it.train_lo for it in p.iterations] == list(range(0, 70, 10)),
      "avance deslizante con tamaño fijo: inicio y fin progresan")

p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, gap=5)
check(all(it.test_lo == it.train_hi + 5 for it in p.iterations) and p.iterations[-1].test_hi <= 100,
      "hueco entre entrenamiento y test")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=5, step=20, step_same=False)
check([it.train_hi for it in p.iterations] == [30, 50, 70, 90] and all(it.n_test == 5 for it in p.iterations),
      "paso independiente del test (mayor)")
check(p.unused_counts()["between"] == 45, f"paso mayor que el test: filas sin evaluar entre tests ({p.unused_counts()})")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, step=5, step_same=False)
check(p.iterations[1].test_lo < p.iterations[0].test_hi and any("solapan" in n for n in p.notes),
      "paso menor que el test: tests solapados y avisados")

p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, end_mode=tw.END_DATE,
         end_date=ORIGEN + 64 * DIA)
check(all(p.times[it.test_hi - 1] <= ORIGEN + 64 * DIA for it in p.iterations) and len(p.iterations) == 3,
      "fin en una fecha: ningún test la sobrepasa")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=15, last=tw.LAST_DISCARD)
check(len(p.iterations) == 4 and all(it.complete for it in p.iterations) and p.unused_counts()["tail"] == 10,
      "último test incompleto descartado")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=15, last=tw.LAST_KEEP)
check(len(p.iterations) == 5 and not p.iterations[-1].complete and p.iterations[-1].n_test == 10,
      "último test incompleto usado, más corto")

mensual = [ORIGEN]
for _ in range(59):
    año, mes = divmod(int(np.datetime64(int(mensual[-1]), "s").astype("datetime64[M]").astype(int)) + 1, 12)
    mensual.append(float((np.datetime64(f"{1970 + año:04d}-{mes + 1:02d}-01") - np.datetime64("1970-01-01"))
                         / np.timedelta64(1, "s")))
datos_mes = serie(60, fechas=mensual)
p = plan(datos_mes, initial_mode=tw.INITIAL_ROWS, initial_rows=24, unit=tw.UNIT_TIME, time_unit=5,
         test_size=12, step_same=True)
check(len(p.iterations) == 3 and all(it.n_test == 12 for it in p.iterations) and all(it.complete for it in p.iterations),
      f"test y paso en meses (calendario): 3 años completos ({[(it.n_test, it.complete) for it in p.iterations]})")
p = plan(datos_mes, initial_mode=tw.INITIAL_ROWS, initial_rows=24, unit=tw.UNIT_TIME, time_unit=6,
         test_size=1, gap=0)
check([it.train_hi for it in p.iterations] == [24, 36, 48], "test en años")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, unit=tw.UNIT_TIME, time_unit=4, test_size=1, gap=1)
check(all(it.n_test == 7 and it.test_lo == it.train_hi + 7 for it in p.iterations), "test y hueco en semanas")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, unit=tw.UNIT_TIME, time_unit=3, test_size=10,
         advance=tw.ROLLING)
check(all(it.n_train == 30 for it in p.iterations), "deslizante medido en tiempo: tamaño constante")

# Calendario: con origen en los días 28 a 31 o el 29 de febrero, cada frontera se cuenta desde el
# origen. Con paso igual al test, los tests son contiguos y no se pierde ningún día.
import calendar
from datetime import date, datetime, timedelta, timezone


def mas_meses(dia, meses):
    """Suma meses y recorta al último día del mes, sin pandas (referencia independiente)."""
    total = dia.month - 1 + meses
    año, mes = dia.year + total // 12, total % 12 + 1
    return date(año, mes, min(dia.day, calendar.monthrange(año, mes)[1]))


def segundos(dia):
    return datetime(dia.year, dia.month, dia.day, tzinfo=timezone.utc).timestamp()


diario = serie(8 * 366, fechas=[segundos(date(2015, 1, 1) + timedelta(days=i)) for i in range(8 * 366)])
origenes = ([date(2016, 1, d) for d in range(1, 32)] + [date(2019, 1, d) for d in range(28, 32)] +
            [date(2016, 2, 28), date(2016, 2, 29), date(2019, 2, 28), date(2015, 8, 31), date(2015, 12, 31)])
casos, malos = 0, []
for origen in origenes:
    for unidad, factor in ((5, 1), (6, 12)):
        if unidad == 6 and origen.day < 28:
            continue
        for hueco in (0, 1):
            for tamaño in (1, 2):
                p = plan(diario, initial_mode=tw.INITIAL_DATE, initial_date=segundos(origen - timedelta(days=1)),
                         unit=tw.UNIT_TIME, time_unit=unidad, test_size=tamaño, gap=hueco, min_train=1,
                         max_iterations=10 ** 5)
                casos += 1
                problema = p.error
                for i, it in enumerate(p.iterations):
                    desde = mas_meses(origen, factor * (i * tamaño + hueco))
                    hasta = mas_meses(origen, factor * (i * tamaño + hueco + tamaño))
                    if p.times[it.test_lo] != segundos(desde):
                        problema = f"iteración {i + 1}: empieza el {p.label(it.test_lo)} en vez del {desde}"
                        break
                    if it.test_hi < p.n_rows and p.times[it.test_hi] != segundos(hasta):
                        problema = f"iteración {i + 1}: acaba antes del {p.label(it.test_hi)} en vez del {hasta}"
                        break
                contiguos = all(b.test_lo == a.test_hi for a, b in zip(p.iterations, p.iterations[1:]))
                if problema or not contiguos or p.unused_counts()["between"]:
                    malos.append((origen, "meses" if unidad == 5 else "años", hueco, tamaño,
                                  problema or "días sin evaluar entre tests"))
check(not malos, f"calendario: {casos} casos con origen en los días 1 a 31 y el 29 de febrero, en meses y años, "
                 f"con y sin hueco: tests contiguos y en su fecha {malos[:3]}")
p = plan(diario, initial_mode=tw.INITIAL_DATE, initial_date=segundos(date(2020, 1, 30)), unit=tw.UNIT_TIME,
         time_unit=5, test_size=1, min_train=1, max_iterations=10 ** 5)
inicios = [p.label(it.test_lo) for it in p.iterations[:4]]
check(inicios == ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"] and p.unused_counts()["between"] == 0,
      f"origen el 31 de enero de 2020: los tests empiezan a fin de cada mes y no se pierde el 30 de marzo ({inicios})")
p = plan(diario, initial_mode=tw.INITIAL_DATE, initial_date=segundos(date(2016, 2, 28)), unit=tw.UNIT_TIME,
         time_unit=6, test_size=1, min_train=1, max_iterations=10 ** 5)
check([p.label(it.test_lo) for it in p.iterations][:5] ==
      ["2016-02-29", "2017-02-28", "2018-02-28", "2019-02-28", "2020-02-29"] and p.unused_counts()["between"] == 0,
      "origen el 29 de febrero de 2016, en años: el 2020-02-28 se evalúa")

con_huecos = serie(80, fechas=np.r_[ORIGEN + np.arange(40) * DIA, ORIGEN + (np.arange(40) + 100) * DIA])
p = plan(con_huecos, initial_mode=tw.INITIAL_ROWS, initial_rows=20, unit=tw.UNIT_TIME, time_unit=3, test_size=10)
check(any("irregulares" in n for n in p.notes) and p.skipped_empty > 0 and any("sin filas de test" in n for n in p.notes),
      "huecos en el tiempo: aviso e iteraciones vacías omitidas")

repetidas = serie(90, repetidas=2)  # tres filas por fecha
for unidad in (tw.UNIT_ROWS, tw.UNIT_TIME):
    for avance in (tw.EXPANDING, tw.ROLLING):
        p = plan(repetidas, initial_mode=tw.INITIAL_ROWS, initial_rows=31, test_size=7, unit=unidad, gap=1,
                 advance=avance)
        partidas = [it for it in p.iterations
                    if set(tiempos(p, it.train_lo, it.train_hi)) & set(tiempos(p, it.test_lo, it.test_hi))
                    or (it.train_lo > 0 and p.times[it.train_lo] == p.times[it.train_lo - 1])]
        check(p.iterations and not partidas, f"fechas iguales nunca separadas (unidad {unidad}, avance {avance})")
check(any("misma fecha" in n for n in p.notes), "fechas repetidas: se avisa")

desordenada = serie(100, desorden=True, sin_fecha=7)
p = plan(desordenada, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10)
check(np.all(np.diff(p.times) > 0) and p.n_rows == 93 and p.missing_time == 7,
      "filas desordenadas: se ordenan; sin fecha: fuera")
check(any("ordenadas" in n for n in p.notes) and any("sin valor" in n for n in p.notes), "se avisa del orden y de las fechas perdidas")

p = tw.plan_windows(datos, tw.WindowSpec(time_var=None, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10))
check(not p.error and p.times is None and p.label(0) == "fila 1", "orden de las filas sin variable de tiempo")
p = tw.plan_windows(datos, tw.WindowSpec(time_var=None, initial_mode=tw.INITIAL_DATE))
check("variable de tiempo" in p.error, "las fechas exigen variable de tiempo")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=100)
check("no deja nada para el test" in p.error, "ventana mayor que los datos: error con solución")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=95, test_size=10)
check("No cabe ninguna iteración" in p.error, "cero iteraciones: error con solución")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=95, test_size=10, last=tw.LAST_KEEP)
check(not p.error and len(p.iterations) == 1, "con el último test incompleto sí cabe una")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=10, test_size=1, max_iterations=50)
check("90 iteraciones" in p.error and "límite es 50" in p.error, "demasiadas iteraciones: límite y aviso")
p = plan(datos, start_mode=tw.START_DATE, start_date=ORIGEN + 500 * DIA)
check("posterior al último dato" in p.error, "fecha de inicio fuera de rango: error")
p = plan(datos, end_mode=tw.END_DATE, end_date=ORIGEN - DIA)
check("no es posterior al inicio" in p.error, "fecha de fin anterior al inicio: error")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=5, min_train=10)
check("mínimo" in p.error, "entrenamiento inicial por debajo del mínimo")
p = plan(datos, initial_mode=tw.INITIAL_ROWS, initial_rows=30, test_size=10, advance=tw.ROLLING, min_train=10,
         unit=tw.UNIT_TIME, time_unit=3)
check(not p.error, "mínimo de entrenamiento respetado")

# ------------------------------------------------------------- Diebold-Mariano ----

def dm_referencia(e1, e2, h, potencia):
    """Traducción literal de forecast::dm.test (R), dos colas, sin vectorizar."""
    d = [abs(a) ** potencia - abs(b) ** potencia for a, b in zip(e1, e2)]
    n = len(d)
    media = sum(d) / n
    cov = []
    for k in range(h):
        cov.append(sum((d[i] - media) * (d[i - k] - media) for i in range(k, n)) / n)
    var = (cov[0] + 2 * sum(cov[1:])) / n
    if var <= 0 and h > 1:
        return dm_referencia(e1, e2, 1, potencia)
    estad = media / np.sqrt(var)
    k = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    estad *= k
    return estad, 2 * stats.t.cdf(-abs(estad), n - 1), h


r = np.random.RandomState(1)
maximo = 0.0
for prueba in range(40):
    n = int(r.randint(20, 120))
    h = int(r.randint(1, 5))
    e1, e2 = r.normal(size=n) * 1.2, r.normal(size=n)
    for potencia in (1, 2):
        esperado, p_esperado, _ = dm_referencia(e1, e2, h, potencia)
        mio = st.diebold_mariano(np.abs(e1) ** potencia, np.abs(e2) ** potencia, h)
        # Por dentro d = pérdida(B) − pérdida(A): el estadístico va con el signo contrario al de R.
        maximo = max(maximo, abs(-mio.statistic - esperado), abs(mio.p_value - p_esperado))
check(maximo < 1e-10, f"Diebold-Mariano (HLN) coincide con forecast::dm.test en 80 casos (máx. dif. {maximo:.1e})")
# A mano: d = [1, 3, 2, 4, 5], h = 2. Media 3; centrada [−2, 0, −1, 1, 2]; γ0 = 10/5 = 2;
# γ1 = (0·(−2) + (−1)·0 + 1·(−1) + 2·1)/5 = 0,2. Varianza a largo plazo 2 + 2·0,2 = 2,4;
# DM = 3/√(2,4/5) = 3/√0,48; HLN = √((5 + 1 − 4 + 2·1/5)/5) = √0,48; estadístico = 3, t con 4 g. l.
mano = st.diebold_mariano(np.zeros(5), np.array([1.0, 3, 2, 4, 5]), 2)
check(abs(mano.statistic - 3.0) < 1e-12 and abs(mano.p_value - 2 * stats.t.sf(3.0, 4)) < 1e-12,
      f"Diebold-Mariano calculado a mano (estadístico {mano.statistic:.6f} = 3)")
negativa = st.diebold_mariano(np.zeros(8), np.array([5.0, -5, 5, -5, 5, -5, 5, -4]), 2)
check(any("se usa h = 1" in w for w in negativa.warnings), "varianza negativa: se repite con h = 1, como R")
iguales = st.diebold_mariano(np.ones(30), np.ones(30))
check(iguales.decision == st.SAME and iguales.p_value == 1.0, "pérdidas idénticas: sin diferencia")
mejor = st.diebold_mariano(np.full(60, 1.0) + r.normal(0, .1, 60), np.full(60, 2.0) + r.normal(0, .1, 60))
mejor.p_adjusted = mejor.p_value
st.decide_frequentist(mejor, 0.05)
check(mejor.direction == 1 and mejor.decision == "A mejor" and mejor.effect < 1, "A con menos pérdida: A mejor")

# Tamaño bajo H0 con diferencias MA(1) (pronóstico a 2 pasos): HLN cerca del 5 %, ignorar la
# autocorrelación (h = 1) da demasiados positivos.
r = np.random.RandomState(7)
rechazos = np.zeros(2)
simulaciones = 2000
for _ in range(simulaciones):
    ruido = r.normal(size=61)
    d = ruido[1:] + 0.9 * ruido[:-1]
    for i, h in enumerate((2, 1)):
        rechazos[i] += st.diebold_mariano(np.zeros(60), d, h).p_value < 0.05
tasa = rechazos / simulaciones
check(0.03 <= tasa[0] <= 0.08 and tasa[1] > tasa[0] + 0.04,
      f"tamaño bajo H0 con MA(1): {tasa[0]:.1%} con h = 2 frente a {tasa[1]:.1%} ignorando la autocorrelación")

# ----------------------------------------------------- sin fugas de datos ----

VISTOS = []      # (máx. fecha de entrenamiento, mín. fecha de test, columnas del modelo)
PREPROCESADO = []


class ModeloTrampa(Model):
    def __init__(self, domain, ultima, media):
        super().__init__(domain)
        self.ultima, self.media = ultima, media

    def predict_storage(self, data):
        fechas = tw.column(data, FECHA)
        VISTOS.append((self.ultima, float(np.nanmin(fechas)), tuple(v.name for v in self.domain.attributes)))
        return np.full(len(data), self.media)


class Trampa(Learner):
    """Detecta si alguna fila de test es anterior o igual a la última de entrenamiento."""
    name = "trampa"

    def fit_storage(self, data):
        return ModeloTrampa(data.domain, float(np.nanmax(tw.column(data, FECHA))), float(np.nanmean(data.Y)))


class PreTrampa(Preprocess):
    def __call__(self, data):
        PREPROCESADO.append((len(data), float(np.nanmax(tw.column(data, FECHA)))))
        return data


class Banco:
    SALIDAS = ("predictions evaluations_results scores fold_scores comparisons ranks best_learner "
               "horizon_scores").split()

    def __init__(self, ajustes=None, **opciones):
        self.w = OWTestScorePro.__new__(OWTestScorePro, signal_manager=None, stored_settings=ajustes)
        self.w.__init__()
        self.salida = {}
        for nombre in self.SALIDAS:
            getattr(self.w.Outputs, nombre).send = self._capturar(nombre)
        if ajustes is None:
            for nombre, valor in dict(random_seed=42, parallel_models=True, comparison_metric="", alpha_index=1,
                                      correction_method="holm", rope_auto=True, rope_value=0.01, hidden_metrics=[],
                                      dm_horizon=0, window_start_mode=0, window_initial_mode=0,
                                      window_initial_percent=50, window_advance=0, window_unit=0, window_test=10,
                                      window_gap=0, window_step=10, window_step_same=True, window_end_mode=0,
                                      window_last=0, window_min_train=10, window_max_iterations=200).items():
                setattr(self.w, nombre, valor)
            self.w.alpha_combo.setCurrentIndex(1)
            self.w.resampling = OWTestScorePro.WINDOW
        for nombre, valor in opciones.items():
            setattr(self.w, nombre, valor)
        self.w._sync_window_values()
        self.w._sync_window_controls()
        self.n = 0

    def _capturar(self, nombre):
        def recibir(valor):
            self.salida[nombre] = valor
        return recibir

    def esperar(self):
        inicio = time.time()
        app.processEvents()
        while (self.w.task is not None or self.w._after_task.isActive() or self.w._debounce.isActive()
               or self.w._window_timer.isActive()) and time.time() - inicio < 300:
            app.processEvents()
            time.sleep(0.002)
        app.processEvents()

    def datos(self, data):
        self.w.set_train_data(data)
        return self

    def modelos(self, *learners):
        for learner in learners:
            self.w.insert_learner(self.n, learner)
            self.n += 1
        return self

    def correr(self):
        self.w.handleNewSignals()
        self.esperar()
        return self

    def opcion(self, nombre, valor):
        self.w._window_option(nombre, valor)
        self.esperar()
        return self

    def errores(self):
        return " ".join(str(m) for m in self.w.Error.active)

    def avisos(self):
        return " ".join(str(m) for m in self.w.Warning.active)

    def cerrar(self):
        self.w.onDeleteWidget()


combinaciones = [
    dict(window_initial_mode=tw.INITIAL_PERCENT, window_initial_percent=40),
    dict(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=30, window_advance=tw.ROLLING),
    dict(window_initial_mode=tw.INITIAL_DATE, window_initial_date=ORIGEN + 35 * DIA, window_gap=3),
    dict(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=30, window_unit=tw.UNIT_TIME, window_time_unit=4,
         window_test=1, window_step_same=False, window_step=2, window_advance=tw.ROLLING),
    dict(window_start_mode=tw.START_ROW, window_start_row=8, window_initial_mode=tw.INITIAL_ROWS,
         window_initial_rows=25, window_end_mode=tw.END_DATE, window_end_date=ORIGEN + 85 * DIA, window_last=tw.LAST_KEEP,
         window_test=12),
    dict(window_start_mode=tw.START_DATE, window_start_date=ORIGEN + 10 * DIA, window_initial_mode=tw.INITIAL_PERCENT,
         window_initial_percent=30, window_step_same=False, window_step=4, window_test=8, window_gap=2),
]
for numero, opciones in enumerate(combinaciones, 1):
    VISTOS.clear()
    PREPROCESADO.clear()
    datos = serie(120, repetidas=1 if numero == 3 else 0, semilla=numero)
    banco = Banco(**opciones).datos(datos).modelos(Trampa(), LinearRegressionLearner())
    banco.w.set_preprocessor(PreTrampa())
    banco.correr()
    plan_usado = banco.w._sampling_used.plan
    resultados = banco.salida.get("evaluations_results")
    check(resultados is not None and not banco.errores(), f"combinación {numero}: evalúa sin errores {banco.errores()}")
    if resultados is None:
        banco.cerrar()
        continue
    esperado = tw.plan_windows(datos, banco.w._window_spec())
    check([(it.train_lo, it.train_hi, it.test_lo, it.test_hi) for it in esperado.iterations] ==
          [(it.train_lo, it.train_hi, it.test_lo, it.test_hi) for it in plan_usado.iterations] and
          banco.w._preview_plan is not None and len(banco.w._preview_plan.iterations) == len(esperado.iterations),
          f"combinación {numero}: la vista previa y la evaluación usan el mismo plan")
    check(len(resultados.folds) == len(plan_usado.iterations), f"combinación {numero}: una partición por iteración")
    fugas = [v for v in VISTOS if v[1] <= v[0]]
    check(VISTOS and not fugas, f"combinación {numero}: ningún test anterior o igual al entrenamiento ({len(VISTOS)} predicciones)")
    if opciones.get("window_gap"):
        check(all(it.test_lo - it.train_hi >= opciones["window_gap"] for it in plan_usado.iterations),
              f"combinación {numero}: el hueco separa entrenamiento y test")
    pre = [(it.n_train, plan_usado.times[it.train_hi - 1]) for it in plan_usado.iterations]
    check(sorted(PREPROCESADO) == sorted(pre),
          f"combinación {numero}: el preprocesador se ajusta solo con el entrenamiento de cada iteración")
    # Las predicciones coinciden con entrenar a mano cada iteración.
    tabla = resultados.data
    lineal = resultados.predicted[1]
    diferencia = 0.0
    for fold, it in zip(resultados.folds, plan_usado.iterations):
        modelo = LinearRegressionLearner()(tabla[it.train_lo:it.train_hi])
        diferencia = max(diferencia, float(np.max(np.abs(modelo(tabla[it.test_lo:it.test_hi]) - lineal[fold]))))
    check(diferencia < 1e-9, f"combinación {numero}: predicciones iguales a entrenar a mano ({diferencia:.1e})")
    predicciones = banco.salida["predictions"]
    nombres = [v.name for v in predicciones.domain.metas]
    check(all(n in nombres for n in ("Iteración", "Inicio entrenamiento", "Fin entrenamiento", "Inicio test",
                                     "Fin test", "Objetivo")),
          f"combinación {numero}: Predictions lleva iteración, fechas de la ventana y objetivo")
    fin_train = predicciones.get_column("Fin entrenamiento") if hasattr(predicciones, "get_column") else None
    if fin_train is not None:
        check(np.all(fin_train < tw.column(predicciones, FECHA)), f"combinación {numero}: cada fila de test es posterior a su entrenamiento")
    banco.cerrar()

# ---------------------------------------------------- multi-horizonte ----

ENTRENOS = [0]


class Contador(LinearRegressionLearner):
    def __call__(self, data, progress_callback=None):
        ENTRENOS[0] += 1
        return super().__call__(data)


VISTOS.clear()
multi = serie(150, objetivos=3, semilla=4)
banco = Banco(window_initial_mode=tw.INITIAL_PERCENT, window_initial_percent=40, window_test=15)
banco.datos(multi).modelos(Contador(), MeanLearner(), Trampa()).correr()
check(not banco.errores(), f"multi-horizonte: evalúa sin errores {banco.errores()}")
n_it = len(banco.w._sampling_used.plan.iterations)
check(ENTRENOS[0] == 3 * n_it, f"multi-horizonte: un entrenamiento por iteración y objetivo ({ENTRENOS[0]} = 3 × {n_it})")
check(all(v[2] == ("x1", "x2") for v in VISTOS), "multi-horizonte: los otros objetivos nunca se usan como atributos")
horizontes = banco.salida["horizon_scores"]
check(horizontes is not None and len(horizontes) == 9, "salida Horizon Scores: modelo × objetivo")
iteraciones = banco.salida["fold_scores"]
check(iteraciones is not None and len(iteraciones) == 3 * 3 * n_it and
      {v.name for v in iteraciones.domain.metas} >= {"Iteración", "Objetivo", "Inicio test", "Fin test"},
      "salida Fold Scores: modelo × iteración × objetivo, con las fechas")
check(banco.salida["evaluations_results"].domain.class_var.name == "y+1", "Evaluation Results del primer objetivo")
for posicion in range(3):
    referencia = LinearRegressionLearner()
    tabla = tw.target_tables(multi, banco.w._sampling_used.plan)[posicion]
    it = banco.w._sampling_used.plan.iterations[0]
    pred = referencia(tabla[it.train_lo:it.train_hi])(tabla[it.test_lo:it.test_hi])
    valores = banco.w.learners[0].outcome.part(posicion).results
    check(np.allclose(valores.predicted[0][valores.folds[0]], pred), f"objetivo {posicion + 1}: se entrena contra su propia variable")
antes = ENTRENOS[0]
banco.w._horizon_target_chosen(2)
banco.esperar()
check(ENTRENOS[0] == antes, "cambiar de objetivo no reentrena")
check(banco.salida["evaluations_results"].domain.class_var.name == "y+3" and banco.w._options().horizon == 3,
      "Evaluation Results y horizonte de DM siguen al objetivo elegido")
dm = banco.w._analysis.pairs_of(st.DIEBOLD_MARIANO)
check(dm and "h = 3" in dm[0].assumption and banco.w._analysis.primary == st.DIEBOLD_MARIANO,
      "Diebold-Mariano es el test principal, con h = 3 para el tercer objetivo")
banco.w.comparison_metric = "MAE"
banco.w._options_changed()
banco.esperar()
check(ENTRENOS[0] == antes and banco.w._analysis.pairs_of(st.DIEBOLD_MARIANO)[0].evidence == "error absoluto por fila",
      "cambiar de métrica no reentrena y DM usa la pérdida de esa métrica")
check(st.TTEST_CORRECTED not in banco.w._analysis.available, "Nadeau-Bengio no se ofrece con ventanas")
check(all(p.assumption == engine.ITERATION_ASSUMPTION for p in banco.w._analysis.pairs_of(st.WILCOXON)),
      "los tests sobre iteraciones se marcan como orientativos")
for grafico in banco.w.charts:
    grafico.figure(7, 4)
ruta_html = os.path.join(DIR, "ventana.html")
banco.w.export_report(ruta_html)
banco.esperar()
html = open(ruta_html, encoding="utf-8").read()
check("Ventana deslizante" in html and "Horizontes" in html and "Diebold-Mariano" in html and "Iteraciones" in html,
      "informe HTML: ventana, iteraciones, horizontes y Diebold-Mariano")
informe = banco.w._analysis
pareado = informe.pairs_of(st.TTEST_PAIRED)
check(pareado and any("Diebold-Mariano" in w for w in pareado[0].warnings) and
      not any("t corregido" in w for w in pareado[0].warnings), "t sin corregir: remite a Diebold-Mariano, no al t corregido")
lecturas = [st.interpret(p, informe.names, informe.metric_label, informe.alpha) for p in informe.pairs_of(st.DIEBOLD_MARIANO)]
ganadas = [t for t in lecturas if t.startswith("Hay evidencia")]
check(ganadas and all(" mejor que " in t for t in ganadas) and len(set(ganadas)) == len(ganadas),
      f"la interpretación de Diebold-Mariano nombra al rival ({ganadas[:1]})")
leyenda = banco.w.chart_scores.figure(7, 4).axes[0].get_xlabel()
check("puntos: iteraciones" in leyenda, f"la comparación de modelos habla de iteraciones ({leyenda!r})")
check("no se aplican el t corregido" in banco.avisos() and "McNemar" in banco.avisos(),
      "el widget dice que el t corregido y McNemar no se aplican a la ventana")
check("no se aplican el t corregido" in html and "ρ = 0" in html, "el informe dice lo que no aplica y que el bayesiano usa ρ = 0")
bayes = informe.pairs_of(st.BAYES)
check(bayes and "ρ = 0" in bayes[0].assumption and "ρ = 0" in builder_explicaciones[st.BAYES],
      "el t bayesiano explica que con ventana va sin correlación")
ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.cerrar()


def revisar_pdf(ruta, layout, etiqueta):
    items = layout.items
    for item in items:
        check(pdf.BODY_TOP - .5 <= item.top and item.bottom <= pdf.BODY_BOTTOM + .5, f"{etiqueta}: {item.kind} «{item.label[:30]}» dentro de la página")
    for actual, siguiente in zip(items, items[1:]):
        if actual.kind == "heading" or actual.keep_next:
            check(actual.page == siguiente.page, f"{etiqueta}: «{actual.label[:40]}» con su contenido")
        if actual.page == siguiente.page:
            check(actual.bottom <= siguiente.top + .5, f"{etiqueta}: sin solapes tras «{actual.label[:30]}»")
    tablas = {}
    for item in items:
        if item.kind == "table":
            if item.fragment == 1:
                actual_tabla = []
                tablas[id(actual_tabla)] = actual_tabla
            actual_tabla.append(item)
    for partes in tablas.values():
        check(sum(p.rows for p in partes) == partes[0].total_rows, f"{etiqueta}: «{partes[0].label}» conserva todas las filas")
        for parte in partes[1:]:
            check(parte.repeats_header, f"{etiqueta}: «{parte.label}» repite la cabecera")
        for parte in partes:
            check(parte.rows >= min(pdf.MIN_ROWS_ON_PAGE, partes[0].total_rows), f"{etiqueta}: «{parte.label}» sin filas sueltas")
    etiquetas = " ".join(item.label for item in items)
    for texto in ("Ventana deslizante", "Ventanas de entrenamiento y test", "Iteraciones", "Diebold-Mariano"):
        check(texto in etiquetas, f"{etiqueta}: incluye «{texto}»")
    if QPdfDocument is not None:
        documento = QPdfDocument(None)
        documento.load(ruta)
        textos = [documento.getAllText(p).text() for p in range(documento.pageCount())]
        check(documento.pageCount() == layout.pages, f"{etiqueta}: páginas coherentes")
        check(not any(re.search(r"\d,\s*\n\s*\d{2,}", texto) for texto in textos), f"{etiqueta}: ningún número partido")
        documento.close()


restaurado = Banco(ajustes=ajustes).datos(multi).modelos(LinearRegressionLearner(), MeanLearner(),
                                                         TreeRegression(max_depth=3)).correr()
check(restaurado.w.resampling == OWTestScorePro.WINDOW and restaurado.w.window_test == 15 and
      restaurado.w.window_initial_percent == 40 and restaurado.w.horizon_target == "y+3" and
      restaurado.w.comparison_metric == "MAE" and restaurado.w.window_time_var is not None and
      restaurado.w.window_time_var.name == "fecha",
      "estado guardado con el workflow: ventana, objetivo, métrica y variable de tiempo")
check(restaurado.w.test_spin.value() == 15 and restaurado.w.initial_percent_spin.value() == 40,
      "los controles muestran los valores restaurados")
ruta_pdf = os.path.join(DIR, "ventana_multi.pdf")
restaurado.w.export_report(ruta_pdf)
restaurado.esperar()
check(not restaurado.w.Error.report.is_shown() and os.path.getsize(ruta_pdf) > 50000, "informe PDF multi-horizonte")
revisar_pdf(ruta_pdf, restaurado.w._report_layout, "PDF multi-horizonte")
destino = os.environ.get("MUESTRA_PDF")
if destino:
    import shutil
    shutil.copy(ruta_pdf, destino)
restaurado.cerrar()

# ----------------------------------------------- anomalías y robustez ----

mezclados = Table.from_numpy(Domain([ContinuousVariable("x")], [ContinuousVariable("a"),
                                                                 DiscreteVariable("b", values=("0", "1"))], [FECHA]),
                             np.arange(50.0)[:, None], np.column_stack([np.arange(50.0), np.arange(50) % 2]),
                             metas=(ORIGEN + np.arange(50) * DIA)[:, None])
banco = Banco().datos(mezclados).modelos(LinearRegressionLearner()).correr()
check("mezclados" in banco.errores(), "objetivos numéricos y categóricos mezclados: error claro")
banco.cerrar()

banco = Banco(resampling=OWTestScorePro.KFOLD).datos(serie(60, objetivos=2)).modelos(LinearRegressionLearner()).correr()
check("Ventana deslizante" in banco.errores(), "varios objetivos con otro muestreo: indica cómo resolverlo")
banco.cerrar()

# Filas sin valor en algún objetivo: se descartan antes de formar las ventanas, para todos a la vez.


def horizontes_con_cola(n=120, objetivos=3, nombre="Fortified", fechas=None, semilla=11):
    """Como en la captura: el objetivo t+h no tiene valor en las h últimas filas."""
    r = np.random.RandomState(semilla)
    base = np.cumsum(r.normal(size=n + 2)) + 50
    X = np.column_stack([base[1:n + 1], base[:n]])
    Y = np.full((n, objetivos), np.nan)
    for h in range(1, objetivos + 1):
        Y[:n - h, h - 1] = base[2 + h:n + 2]
    t = ORIGEN + np.arange(n) * DIA if fechas is None else np.asarray(fechas[:n], dtype=float)
    dominio = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")],
                     [ContinuousVariable(f"{nombre}_t+{h}") for h in range(1, objetivos + 1)], [FECHA])
    tabla = Table.from_numpy(dominio, X, Y, metas=t[:, None])
    tabla.name = "captura"
    return tabla


esperado = ("Se han descartado 3 filas sin valor en algún objetivo (Fortified_t+1: 1; Fortified_t+2: 2; "
            "Fortified_t+3: 3).")
mensual_largo = [ORIGEN]
while len(mensual_largo) < 120:
    siguiente = mas_meses(datetime.fromtimestamp(mensual_largo[-1], timezone.utc).date(), 1)
    mensual_largo.append(segundos(siguiente))
for etiqueta, opciones, fechas in (
        ("filas", dict(window_unit=tw.UNIT_ROWS), None),
        ("meses", dict(window_unit=tw.UNIT_TIME, window_time_unit=5), mensual_largo)):
    for avance in (tw.EXPANDING, tw.ROLLING):
        captura = horizontes_con_cola(fechas=fechas)
        banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=56, window_test=12,
                      window_last=tw.LAST_KEEP, window_advance=avance, **opciones).datos(captura)
        banco.modelos(LinearRegressionLearner(), MeanLearner()).correr()
        nombre = f"captura en {etiqueta}, avance {'creciente' if avance == tw.EXPANDING else 'fijo'}"
        informacion = " ".join(str(m) for m in banco.w.Information.active)
        check(not banco.errores() and banco.salida.get("evaluations_results") is not None,
              f"{nombre}: evalúa sin errores {banco.errores()}")
        check(esperado in informacion, f"{nombre}: informa de las filas descartadas ({informacion[:160]})")
        check(len(banco.w.data) == 117 and not np.isnan(banco.w.data.Y).any(), f"{nombre}: quedan 117 filas, todas con objetivo")
        plan_usado = banco.w._sampling_used.plan
        previa = banco.w._preview_plan
        check(previa is not None and [(i.train_lo, i.test_lo, i.test_hi) for i in previa.iterations] ==
              [(i.train_lo, i.test_lo, i.test_hi) for i in plan_usado.iterations],
              f"{nombre}: la vista previa refleja los datos que se usan")
        check(plan_usado.iterations[-1].n_test == 1 and not plan_usado.iterations[-1].complete,
              f"{nombre}: la última iteración tiene un test de una sola fila")
        partes = [banco.w.learners[0].outcome.part(t).results for t in range(3)]
        check(all(p is not None and np.array_equal(p.row_indices, partes[0].row_indices) for p in partes),
              f"{nombre}: todos los objetivos se evalúan sobre las mismas filas")
        check(len(banco.salida["evaluations_results"].folds) == len(plan_usado.iterations),
              f"{nombre}: ninguna iteración perdida")
        if etiqueta == "filas" and avance == tw.EXPANDING:
            ruta = os.path.join(DIR, "descartadas.html")
            banco.w.export_report(ruta)
            banco.esperar()
            check(esperado in open(ruta, encoding="utf-8").read(), "el informe dice qué filas se han descartado")
        banco.cerrar()

for avance in (tw.EXPANDING, tw.ROLLING):
    medio = serie(120, semilla=3)
    with medio.unlocked():
        medio.Y[[30, 31, 77]] = np.nan
    banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=40, window_advance=avance).datos(medio)
    banco.modelos(LinearRegressionLearner(), MeanLearner()).correr()
    informacion = " ".join(str(m) for m in banco.w.Information.active)
    check(not banco.errores() and "Se han descartado 3 filas sin valor en la variable objetivo «y+1»." in informacion,
          f"un objetivo con valores perdidos en medio (avance {avance}): se descartan y se informa")
    check(np.isfinite(banco.salida["evaluations_results"].actual).all(), "sin valores perdidos en el test")
    banco.cerrar()

    varios = serie(120, objetivos=3, semilla=3)
    with varios.unlocked():
        varios.Y[[50, 51], 1] = np.nan
        varios.Y[[51, 70], 2] = np.nan
    banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=40, window_test=7,
                  window_advance=avance).datos(varios)
    banco.modelos(LinearRegressionLearner(), MeanLearner()).correr()
    informacion = " ".join(str(m) for m in banco.w.Information.active)
    check(not banco.errores() and "Se han descartado 3 filas sin valor en algún objetivo (y+2: 2; y+3: 2)." in informacion,
          f"varios objetivos con valores perdidos en medio (avance {avance}): se descarta cada fila una vez ({informacion[:120]})")
    partes = [banco.w.learners[0].outcome.part(t).results for t in range(3)]
    check(all(np.array_equal(p.row_indices, partes[0].row_indices) for p in partes) and len(banco.w.data) == 117,
          "todos los objetivos con las mismas filas")
    banco.cerrar()

# Sin limpiar (motor usado directamente): una iteración vacía para un objetivo se omite, no para nada.
sucia = serie(100, objetivos=2, semilla=8)
with sucia.unlocked():
    sucia.Y[60:70, 1] = np.nan
muestreo = engine.Sampling(kind=engine.WINDOW, window=tw.WindowSpec(time_var=FECHA, initial_mode=tw.INITIAL_ROWS,
                                                                    initial_rows=40, test_size=10))
muestreo, notas, problema = engine.check_sampling(sucia, None, muestreo)
omitidas = [n for n in notas if n.startswith("Se omite")]
check(problema is None and len(muestreo.plan.iterations) == 5 and omitidas and "la 3 (test" in omitidas[0]
      and "«y+2»" in omitidas[0], f"iteración vacía para un objetivo: se omite y se identifica ({omitidas[:1]})")
salidas = engine.evaluate(sucia, None, [(0, LinearRegressionLearner())], muestreo)
check(all(parte.results is not None for parte in salidas[0].per_target) and
      len(salidas[0].part(1).results.folds) == 5, "el resto de iteraciones se evalúa")
with sucia.unlocked():
    sucia.Y[40:, 1] = np.nan
muestreo = engine.Sampling(kind=engine.WINDOW, window=tw.WindowSpec(time_var=FECHA, initial_mode=tw.INITIAL_ROWS,
                                                                    initial_rows=40, test_size=10))
_, _, problema = engine.check_sampling(sucia, None, muestreo)
check(problema and "Ninguna iteración" in problema, "si no queda ninguna iteración, error que lo explica")

# Con un solo objetivo y otro muestreo: igual que el Test & Score nativo, que también quita esas filas.
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
con_hueco = serie(90, semilla=6)
with con_hueco.unlocked():
    con_hueco.Y[[5, 44, 89]] = np.nan
nativo = OWTestAndScore.__new__(OWTestAndScore, signal_manager=None, stored_settings=None)
nativo.__init__()
capturado = {}
nativo.Outputs.evaluations_results.send = lambda valor: capturado.__setitem__("r", valor)
nativo.resampling, nativo.n_folds = 0, 1
nativo.set_train_data(con_hueco)
nativo.insert_learner(0, LinearRegressionLearner())
nativo.handleNewSignals()
inicio = time.time()
while nativo._OWTestAndScore__state.name == "Running" and time.time() - inicio < 60:
    app.processEvents()
    time.sleep(0.002)
app.processEvents()
banco = Banco(resampling=OWTestScorePro.KFOLD, n_folds=1).datos(con_hueco).modelos(LinearRegressionLearner()).correr()
mio = banco.salida["evaluations_results"]
check(capturado.get("r") is not None and np.array_equal(capturado["r"].row_indices, mio.row_indices) and
      np.allclose(capturado["r"].predicted, mio.predicted) and len(banco.w.data) == 87,
      "un objetivo con valores perdidos y validación cruzada: igual que el nativo")
nativo.onDeleteWidget()
banco.cerrar()

constante = serie(80, semilla=2)
with constante.unlocked():
    constante.Y[:40] = 5.0
banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=30).datos(constante)
banco.modelos(LinearRegressionLearner(), MeanLearner()).correr()
check(not banco.errores() and "constante en el entrenamiento" in banco.avisos(), "objetivo constante en una ventana: aviso, sin fallo")
banco.cerrar()

clasif = serie(120, discreta=True, semilla=5)
with clasif.unlocked():
    clasif.Y[:35] = 0.0
banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=30).datos(clasif)
banco.modelos(LogisticRegressionLearner(), MajorityLearner(), TreeLearner()).correr()
avisos = banco.avisos()
check("una sola clase" in avisos and "no aparecen en su entrenamiento" in avisos,
      "clasificación: una sola clase en el entrenamiento y clases nuevas en el test, avisadas")
check("logistic regression ha fallado en la iteración 1" in avisos, f"el modelo que falla dice en qué iteración ({avisos[:200]})")
check(banco.salida.get("evaluations_results") is not None and len(banco.w._analysis.names) == 2,
      "los demás modelos siguen")
matriz = OWConfusionMatrix()
matriz.set_results(banco.salida["evaluations_results"])
matriz.handleNewSignals() if hasattr(matriz, "handleNewSignals") else None
check(not matriz.Error.active, "Evaluation Results de la ventana se conecta a Confusion Matrix")
matriz.onDeleteWidget()
dm = banco.w._analysis.pairs_of(st.DIEBOLD_MARIANO)
check(dm and dm[0].evidence.startswith("aciertos por fila"), "clasificación: DM sobre la pérdida 0/1")
check(st.MCNEMAR not in banco.w._analysis.available, "McNemar no se ofrece con ventanas (filas autocorrelacionadas)")
banco.cerrar()

banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=95).datos(serie(100))
banco.modelos(LinearRegressionLearner()).correr()
check("No cabe ninguna iteración" in banco.errores() and banco.salida.get("evaluations_results") is None,
      "cero iteraciones en el widget: error y sin salidas")
banco.opcion("window_last", tw.LAST_KEEP)
check(not banco.errores() and banco.salida.get("evaluations_results") is not None, "al usar el test incompleto, evalúa")
banco.cerrar()

# Evaluación grande: no arranca sola.
grande = serie(3000, semilla=9)
banco = Banco(window_initial_mode=tw.INITIAL_ROWS, window_initial_rows=500, window_test=2,
              window_max_iterations=2000).datos(grande)
banco.modelos(LinearRegressionLearner(), MeanLearner(), TreeRegression(max_depth=2)).correr()
check(banco.w.evaluate_button.isVisibleTo(banco.w) and "Evaluar" in banco.w.banner.text() and banco.w.task is None,
      "más de 2.000 entrenamientos: espera a que se pulse Evaluar")
check("1250 iteraciones" in banco.w.window_summary.text() and "3.750 entrenamientos" in banco.w.window_summary.text(),
      "la estimación de iteraciones y entrenamientos se ve antes de lanzar")
inicio = time.time()
banco.w._confirm_large()
time.sleep(0.05)
app.processEvents()
check(banco.w.task is not None, "al pulsar Evaluar, empieza")
banco.w.set_train_data(serie(1200, semilla=10))   # cambio de datos a media evaluación
banco.w.handleNewSignals()
banco.esperar()
check(not banco.errores() and len(banco.salida["evaluations_results"].data) == 1200,
      "cambiar los datos a media evaluación descarta la anterior")
banco.cerrar()

# Sin variable de tiempo: orden de las filas.
sin_tiempo = Table.from_numpy(Domain([ContinuousVariable("x")], ContinuousVariable("y")),
                              np.arange(80.0)[:, None], np.arange(80.0) * 2)
banco = Banco().datos(sin_tiempo).modelos(LinearRegressionLearner(), MeanLearner()).correr()
check(not banco.errores() and banco.w.window_time_var is None and banco.salida["predictions"] is not None,
      "sin variable de tiempo: usa el orden de las filas")
check(not banco.w.start_combo.model().item(tw.START_DATE).isEnabled(), "sin variable de tiempo, las fechas no se ofrecen")
banco.cerrar()

# Tablas de Orange3-Timeseries: la variable de tiempo se elige sola.
try:
    from orangecontrib.timeseries import Timeseries
    pasajeros = Timeseries.from_file("airpassengers")
except Exception:  # pylint: disable=broad-except
    pasajeros = None
if pasajeros is not None:
    banco = Banco(window_test=12).datos(pasajeros).modelos(LinearRegressionLearner(), MeanLearner()).correr()
    check(banco.w.window_time_var is pasajeros.time_variable and not banco.errores(),
          "tabla de Timeseries: se ordena por su variable de tiempo")
    banco.cerrar()

# Los demás muestreos siguen igual.
banco = Banco(resampling=OWTestScorePro.KFOLD, n_folds=3).datos(serie(100)).modelos(LinearRegressionLearner(), MeanLearner()).correr()
check(banco.w._analysis.primary == st.TTEST_CORRECTED and not banco.w.window_box.isVisibleTo(banco.w),
      "validación cruzada intacta: t corregido y sin panel de ventana")
banco.cerrar()

check(not excepciones, "sin excepciones sin capturar:\n" + "\n".join(excepciones))
print("\nFALLOS:", fallos)
sys.exit(1 if fallos else 0)
