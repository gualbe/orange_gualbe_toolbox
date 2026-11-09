"""Prueba sin interfaz gráfica de Test & Score Pro.

Script ejecutable con el Python de Orange; sale con 1 si alguna comprobación falla.
Cubre la validación estadística, la comparación con el Test & Score nativo y la matriz de
muestreos × tipos de problema × anomalías.
"""
import os
import sys
import tempfile
import threading
import time
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import baycomp
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from AnyQt.QtWidgets import QApplication
import Orange.base
from Orange.base import Learner
from Orange import modelling
from Orange.classification import (
    LogisticRegressionLearner, MajorityLearner, NaiveBayesLearner, RandomForestLearner, TreeLearner,
)
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from Orange.regression import LinearRegressionLearner, MeanLearner, RidgeRegressionLearner, TreeLearner as TreeRegression
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_engine as engine
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_report as builder
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st
from orangecontrib.orange_gualbe_toolbox.widgets.owtestscorepro import OWTestScorePro

warnings.simplefilter("ignore")
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []
DIR = tempfile.mkdtemp(prefix="test-tsp-")


def _anotar(*error):
    texto = "".join(traceback.format_exception(*error))
    excepciones.append(texto)
    print("EXCEPCIÓN\n" + texto)


sys.excepthook = _anotar


def check(ok, msg):
    global fallos
    if not ok:
        print("FAIL " + msg)
    fallos += not ok
    return ok


def cerca(a, b, tol=1e-9):
    return bool(abs(a - b) <= tol)


# ------------------------------------------------- validación estadística ----

# Demšar (2006), «Statistical comparisons of classifiers over multiple data sets», ej. de la sección 3.2.3.
RANGOS_DEMSAR = [3.143, 2.000, 2.893, 1.964]
chi2 = st.friedman_from_mean_ranks(RANGOS_DEMSAR, 14)
check(cerca(round(chi2, 2), 9.28), f"Demšar: χ² = 9,28 ({chi2})")
check(cerca(round(st.iman_davenport(chi2, 14, 4), 2), 3.69), "Demšar: F de Iman-Davenport = 3,69")
check(cerca(round(st.critical_difference(4, 14, 0.05), 2), 1.25), "Demšar: CD(0,05) = 1,25")
check(cerca(round(st.critical_difference(4, 14, 0.10), 2), 1.12), "Demšar: CD(0,10) = 1,12")
# Tabla 5 de Demšar: valores críticos q_α de Nemenyi para k = 2…10.
for alpha, tabla in ((0.05, (1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164)),
                     (0.10, (1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920))):
    for k, publicado in enumerate(tabla, 2):
        check(abs(st.nemenyi_q(k, alpha) - publicado) <= 0.002, f"q Nemenyi k={k} α={alpha}")

# Wilcoxon de la sección 3.1.3 de Demšar (C4.5 frente a C4.5+m): R+ = 93, R- = 12.
c45 = np.array([.763, .599, .954, .628, .882, .936, .661, .583, .775, 1, .940, .619, .972, .957])
c45m = np.array([.768, .591, .971, .661, .888, .931, .668, .583, .838, 1, .962, .666, .981, .978])
check(st.signed_ranks(c45m - c45, "zsplit") == (93.0, 12.0), "Demšar: sumas de rangos 93 y 12")

# Valor p exacto de Wilcoxon a mano: 5 diferencias positivas distintas → 2·(1/2)^5.
exacto = st.wilcoxon_test(np.array([1., 2, 3, 4, 5]), np.zeros(5))
check(cerca(exacto.p_value, 0.0625), "Wilcoxon: p exacto n=5 = 0,0625")
check(any("nunca podrá" in w for w in exacto.warnings), "Wilcoxon: avisa de que n=5 no alcanza α")
check(cerca(exacto.effect, 1.0), "Wilcoxon: efecto r = 1 si todo es positivo")

rng = np.random.RandomState(0)
for semilla in range(30):
    a = rng.normal(.8, .03, 10)
    b = a - rng.normal(.01, .02, 10)
    r = st.corrected_ttest(a, b, 0.0)
    t = stats.ttest_rel(a, b)
    check(cerca(r.statistic, t.statistic) and cerca(r.p_value, t.pvalue), "t sin corregir = ttest_rel")
    d = a - b
    manual = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 10 + 1 / 9))
    r = st.corrected_ttest(a, b, 1 / 9)
    check(cerca(r.statistic, manual) and cerca(r.p_value, 2 * stats.t.sf(abs(manual), 9)), "t de Nadeau-Bengio a mano")
    medio = stats.t.ppf(.975, 9) * np.sqrt(d.var(ddof=1) * (1 / 10 + 1 / 9))
    check(cerca(r.ci_low, d.mean() - medio) and cerca(r.ci_high, d.mean() + medio), "IC corregido a mano")
    check(cerca(r.effect, d.mean() / d.std(ddof=1) * (1 - 3 / (4 * 9 - 1))), "g de Hedges a mano")
    ref = baycomp.two_on_single(a, b, rope=0.01)
    r = st.bayes_correlated(a, b, 0.01, 0.1)
    check(cerca(r.p_a, ref[0]) and cerca(r.p_rope, ref[1]) and cerca(r.p_b, ref[2]), "bayesiano = baycomp")
    propia = st._student_posterior(d.mean(), d.var(ddof=1), 10, 0.01, 0.1)
    check(all(cerca(x, y, 1e-12) for x, y in zip(propia, ref)), "posterior propia = baycomp")
    ref2 = baycomp.two_on_single(a, b)
    r0 = st.bayes_correlated(a, b, 0.0, 0.1)
    check(cerca(r0.p_a, ref2[0]) and cerca(r0.p_b, ref2[1]) and r0.p_rope == 0, "bayesiano sin ROPE")
    w = st.wilcoxon_test(a, b)
    ref3 = stats.wilcoxon(a - b, method="exact")
    check(cerca(w.p_value, ref3.pvalue), "Wilcoxon = scipy")

for bb, cc in ((3, 8), (30, 12), (10, 14), (0, 5), (100, 70), (24, 0), (13, 12)):
    A = np.r_[np.ones(bb, bool), np.zeros(cc, bool), np.ones(25, bool), np.zeros(15, bool)]
    B = np.r_[np.zeros(bb, bool), np.ones(cc, bool), np.ones(25, bool), np.zeros(15, bool)]
    r = st.mcnemar_test(A, B)
    ref = sm_mcnemar([[25, bb], [cc, 15]], exact=bb + cc < 25, correction=True)
    check(cerca(r.p_value, ref.pvalue) and cerca(r.statistic, ref.statistic), f"McNemar = statsmodels ({bb},{cc})")

valores = rng.uniform(size=40)
for metodo in ("holm", "bonferroni", "fdr_bh"):
    ref = multipletests(valores, method=metodo)[1]
    check(np.allclose(st.adjust_pvalues(valores, metodo), ref), f"{metodo} = statsmodels")
    con_nan = np.r_[valores[:5], np.nan]
    salida = st.adjust_pvalues(con_nan, metodo)
    check(np.isnan(salida[5]) and np.allclose(salida[:5], multipletests(valores[:5], method=metodo)[1]),
          f"{metodo} ignora los NaN")

for bloques in (6, 15):
    datos = rng.normal(size=(4, bloques))
    datos[1, :3] = datos[0, :3]  # empates
    r = st.friedman_test(datos, on_folds=False)
    ref = stats.friedmanchisquare(*(-datos))
    check(cerca(r.chi2, ref.statistic) and cerca(r.p_value, ref.pvalue), "Friedman con empates = scipy")

# Monte Carlo: sin diferencia real y con la correlación de una validación de 10 particiones,
# el t clásico da demasiados positivos y el corregido respeta α.
sim = np.random.RandomState(1)
k, n_sim, rho = 10, 40000, 1 / 10
comun = sim.normal(size=(n_sim, 1))
propio = sim.normal(size=(n_sim, k))
d = np.sqrt(rho) * comun + np.sqrt(1 - rho) * propio
media, var = d.mean(axis=1), d.var(axis=1, ddof=1)
t_clasico = media / np.sqrt(var / k)
t_corregido = media / np.sqrt(var * (1 / k + 1 / (k - 1)))
limite = stats.t.ppf(.975, k - 1)
fp_clasico = np.mean(np.abs(t_clasico) > limite)
fp_corregido = np.mean(np.abs(t_corregido) > limite)
print(f"Monte Carlo α=0,05: t clásico {fp_clasico:.3f}, t corregido {fp_corregido:.3f}")
check(fp_clasico > 0.12, f"el t clásico da demasiados positivos ({fp_clasico:.3f})")
check(abs(fp_corregido - 0.05) < 0.01, f"el t corregido respeta α ({fp_corregido:.3f})")

# Casos degenerados.
iguales = np.array([.8, .9, .85, .7])
r = st.corrected_ttest(iguales, iguales, 0.1)
check(r.p_value == 1 and r.decision == st.SAME, "t: modelos idénticos")
r = st.corrected_ttest(iguales, iguales - .1, 0.1)
check(np.isnan(r.p_value) and any("Varianza nula" in w for w in r.warnings), "t: varianza nula avisa")
r = st.corrected_ttest(np.array([.8]), np.array([.7]), 0.1)
check(np.isnan(r.p_value) and r.warnings, "t: una sola partición avisa")
r = st.corrected_ttest(np.array([.8, np.nan, .7, .6]), np.array([.7, .5, np.nan, .5]), 0.1)
check(r.n == 2 and any("descartado" in w for w in r.warnings), "t: particiones sin definir se descartan")
r = st.wilcoxon_test(iguales, iguales)
check(r.p_value == 1 and r.decision == st.SAME, "Wilcoxon: idénticos")
r = st.wilcoxon_test(np.array([.1, .2, .2, .3, 0.]), np.zeros(5))
check(any("empat" in w for w in r.warnings), "Wilcoxon: avisa de empates")
r = st.bayes_correlated(iguales, iguales, 0.01, .25)
check(r.p_rope == 1 and any("Varianza nula" in w for w in r.warnings), "bayesiano: varianza nula")
r = st.bayes_correlated(iguales, iguales - .1, 0.01, .25)
check(r.p_a == 1, "bayesiano: varianza nula con diferencia")
r = st.mcnemar_test(np.ones(5, bool), np.ones(5, bool))
check(r.p_value == 1 and r.decision == st.SAME, "McNemar: mismos aciertos")
r = st.friedman_test(np.ones((3, 6)))
check(r.p_value == 1 and any("empatan" in w for w in r.warnings), "Friedman: todo empatado")
r = st.friedman_test(np.ones((2, 6)))
check(np.isnan(r.p_value) and any("3 modelos" in w for w in r.warnings), "Friedman: 2 modelos")
r = st.friedman_test(rng.normal(size=(3, 4)))
check(any("concibieron" in w for w in r.warnings), "Friedman: dice que son particiones")
check(any("recomiendan" in w for w in r.warnings), "Friedman: pocos bloques avisa")


# Defectos corregidos en la 0.3.1: ruido de coma flotante.
a = np.array([14, 13, 15, 13, 14, 14, 13, 12, 14, 13]) / 15
b = np.array([13, 12, 14, 12, 13, 13, 12, 11, 13, 12]) / 15
ruido = st.corrected_ttest(a, b, 1 / 9)
check(np.isnan(ruido.p_value) and ruido.decision == st.NOT_AVAILABLE, f"varianza casi nula: no evaluable ({ruido.p_value})")
check(any("Varianza nula" in w for w in ruido.warnings), "varianza casi nula: avisa")
limpio = st.corrected_ttest(a, a - 1 / 15, 1 / 9)
check(ruido.decision == limpio.decision and cerca(ruido.mean_diff, 1 / 15), "varianza casi nula = varianza exacta 0")
w = st.wilcoxon_test(a, b)
referencia = 2 * stats.norm.sf((27.5 - 0.5) / np.sqrt(10 * 11 * 21 / 24 - (10 ** 3 - 10) / 48))
check(cerca(w.p_value, referencia, 1e-9) and abs(w.p_value - 0.00190) < 1e-5, f"Wilcoxon con empates bien tratados ({w.p_value})")
bay = st.bayes_correlated(a, b, 0.01, 0.1)
check(bay.p_a == 1 and any("Varianza nula" in x for x in bay.warnings), "bayesiano: varianza casi nula es un punto")
izquierda, derecha = st.corrected_interval(np.full(10, 1 / 3) + np.array([0, 1e-17] * 5), 1 / 9)
check(cerca(izquierda, derecha), "intervalo sin varianza real: es un punto")
empates = np.array([[0.1 + 0.2, 0.3, 0.5], [0.3, 0.1 + 0.2, 0.5], [0.1, 0.1, 0.1]]).T
rangos, _ = st.rank_blocks(empates)
check(rangos[0, 0] == rangos[1, 0] and rangos[0, 1] == rangos[1, 1], "Friedman: empate con ruido es empate")

# Tolerancia por valor: una escala enorme en una partición no borra las demás.
a = np.array([1e15, .21, .25, .19, .22, .26, .2, .24, .23, .18])
b = np.array([1e15, .23, .28, .2, .25, .27, .23, .26, .24, .2])
r = st.corrected_ttest(a, b, 1 / 9)
d = a - b
manual = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 10 + 1 / 9))
check(cerca(r.statistic, manual) and r.p_value < 1 and r.decision != st.SAME, f"escala enorme: t real ({r.statistic}, {r.p_value})")
check(not any("idénticas" in w for w in r.warnings), "escala enorme: no se dicen idénticas")
w = st.wilcoxon_test(a, b)
# Las diferencias reales son múltiplos de 0,01: hay empates de verdad entre ellas.
referencia = stats.wilcoxon(np.round((a - b)[1:], 2), method="approx", correction=True).pvalue
check(cerca(w.p_value, referencia, 1e-12) and w.n == 10 and w.decision != st.SAME, f"escala enorme: Wilcoxon real ({w.p_value})")
check(any("1 empates" in x for x in w.warnings), f"escala enorme: solo la partición idéntica es empate {w.warnings}")
bloques = np.array([[1e12, 1e12 * 1.5, 1e12 * .9], [.08, .09, .1], [.2, .25, .3]])
rangos, _ = st.rank_blocks(-bloques)
check(np.array_equal(rangos[1], [1, 1, 1]) and np.array_equal(rangos[2], [2, 2, 2]) and np.array_equal(rangos[0], [3, 3, 3]),
      f"rangos: un valor enorme no empata a los demás {rangos.tolist()}")

# El caso del redondeo sigue siendo no evaluable y las varianzas reales pequeñas, evaluables.
a = np.array([14, 13, 15, 13, 14, 14, 13, 12, 14, 13]) / 15
b = np.array([13, 12, 14, 12, 13, 13, 12, 11, 13, 12]) / 15
r = st.corrected_ttest(a, b, 1 / 9)
check(np.isnan(r.p_value) and r.decision == st.NOT_AVAILABLE, "redondeo: sigue no evaluable")
check(abs(st.wilcoxon_test(a, b).p_value - 0.001904) < 5e-6, "redondeo: Wilcoxon 0,001904")
aleatorio = np.random.RandomState(3)
base = aleatorio.uniform(.7, .9, 10)
for sd in (1e-6, 1e-9, 1e-11):
    ruido = np.abs(aleatorio.normal(0, sd, 10)) + 5 * sd
    otro = base - ruido
    r = st.corrected_ttest(base, otro, 1 / 9)
    w = st.wilcoxon_test(base, otro)
    check(np.isfinite(r.p_value) and r.decision != st.SAME, f"sd {sd}: t evaluable")
    check(not any("empat" in x for x in w.warnings), f"sd {sd}: Wilcoxon sin empates falsos {w.warnings}")
    check(abs(w.p_value - 2 / 1024) < 1e-9, f"sd {sd}: Wilcoxon exacto ({w.p_value})")
for escala in (1e6, 1e-6):
    x, y = base + aleatorio.normal(0, .01, 10), base + aleatorio.normal(0, .01, 10)
    uno, dos = st.corrected_ttest(x, y, 1 / 9), st.corrected_ttest(x * escala, y * escala, 1 / 9)
    check(cerca(uno.statistic, dos.statistic, 1e-6) and cerca(st.wilcoxon_test(x, y).p_value, st.wilcoxon_test(x * escala, y * escala).p_value),
          f"la escala {escala} no cambia el resultado")

# Diferencia constante con un cero o con escalas muy distintas.
cnt = np.array([1, 954, 256, 563, 146, 241, 978, 730, 74, 188])
r = st.corrected_ttest(cnt / 1000, (cnt - 1) / 1000, 1 / 9)
check(np.isnan(r.p_value) and r.decision == st.NOT_AVAILABLE, f"constante con un cero: no evaluable ({r.p_value})")
barrido = np.random.RandomState(11)
escapes = 0
for _ in range(2000):
    c = barrido.randint(1, 1000, 10)
    c[0] = 1
    escapes += np.isfinite(st.corrected_ttest(c / 1000, (c - 1) / 1000, 1 / 9).p_value)
check(escapes == 0, f"constante con ceros: barrido sin escapes ({escapes} de 2000)")
b = np.array([7442.349368719807, 623.6650188352655, 5.8732187084815095, 482.1421028527768, 0.05124441683726436,
              69.10765046043326, 0.07246356970270898, 4656.005689076002, 13.523525694052873, 3.0758959477348573])
r = st.corrected_ttest(b + .02, b, 1 / 9)
check(np.isnan(r.p_value) and r.decision == st.NOT_AVAILABLE and cerca(r.mean_diff, .02, 1e-9), f"constante con escalas distintas: no evaluable ({r.p_value})")
escapes = 0
for _ in range(2000):
    b = 10 ** barrido.uniform(-2, 4, 10)
    escapes += np.isfinite(st.corrected_ttest(b + .02, b, 1 / 9).p_value)
check(escapes == 0, f"escalas 1e-2 a 1e4: barrido sin escapes ({escapes} de 2000)")
escapes = 0
for denominador in (3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 29, 31):
    for _ in range(1940):
        k = barrido.randint(5, 50, 10)
        paso = barrido.randint(1, 5)
        x, y = k / denominador, (k - paso) / denominador
        escapes += np.isfinite(st.corrected_ttest(x, y, 1 / 9).p_value)
check(escapes == 0, f"fracciones: barrido de 23 280 sin escapes ({escapes})")
check(cerca(st.bayes_correlated(cnt / 1000, (cnt - 1) / 1000, 0.0, .1).p_a, 1.0), "bayesiano: constante positiva")

# Una partición enorme no resuelve la diferencia: no impide ver que es constante.
b = np.r_[1e15, np.array([.2, .3, .25, .4, .35, .22, .28, .31, .27])]
a = b + .02
r = st.corrected_ttest(a, b, 1 / 9)
check(a[0] == b[0] and np.isnan(r.p_value) and r.decision == st.NOT_AVAILABLE, f"constante con partición enorme: no evaluable ({r.p_value})")
check(cerca(st.corrected_ttest(a, b, 1 / 9).mean_diff, .02, 1e-9), "constante con partición enorme: la diferencia es la de las demás")

# El ruido de una partición enorme no pesa en el t ni en el bayesiano.
a = np.array([1e16 + 8, .21, .25, .19, .22, .26, .2, .24, .23, .18])
b = np.array([1e16, .23, .28, .2, .25, .27, .23, .26, .24, .2])
r = st.corrected_ttest(a, b, 1 / 9)
d = np.r_[0.0, (a - b)[1:]]
manual = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 10 + 1 / 9))
check(cerca(r.statistic, manual, 1e-9) and round(r.statistic, 2) == -3.79, f"partición 1e16: t = -3,79 ({r.statistic})")
referencia = st.corrected_ttest(np.r_[.5, a[1:]], np.r_[.5, b[1:]], 1 / 9)
check(cerca(r.p_value, referencia.p_value, 1e-9), "partición 1e16: igual que si valiera cero")
bayes_a = st.bayes_correlated(a, b, 0.0, .1)
bayes_b = st.bayes_correlated(np.r_[.5, a[1:]], np.r_[.5, b[1:]], 0.0, .1)
check(cerca(bayes_a.p_a, bayes_b.p_a, 1e-9), "partición 1e16: bayesiano sin ruido")

# Lo ya corregido sigue igual.
a = np.array([1e15, .21, .25, .19, .22, .26, .2, .24, .23, .18])
b = np.array([1e15, .23, .28, .2, .25, .27, .23, .26, .24, .2])
r = st.corrected_ttest(a, b, 1 / 9)
check(round(r.statistic, 3) == -3.793 and round(r.p_value, 4) == 0.0043, f"partición 1e15: t = -3,793, p = 0,0043 ({r.statistic}, {r.p_value})")

# El suelo no se traga diferencias reales pequeñas.
base = np.random.RandomState(4).uniform(.7, .9, 10)
aleatorio = np.random.RandomState(5)
for sd in (1e-6, 1e-9, 1e-11):
    otro = base - (np.abs(aleatorio.normal(0, sd, 10)) + 5 * sd)
    r = st.corrected_ttest(base, otro, 1 / 9, floor=st.RELATIVE_TOLERANCE)
    w = st.wilcoxon_test(base, otro, floor=st.RELATIVE_TOLERANCE)
    check(np.isfinite(r.p_value) and r.decision != st.SAME, f"suelo con sd {sd}: t evaluable")
    check(abs(w.p_value - 2 / 1024) < 1e-9, f"suelo con sd {sd}: Wilcoxon exacto ({w.p_value})")
ruido_puro = np.random.RandomState(6).normal(0, 1e-15, 10)
r = st.corrected_ttest(1e-15 + ruido_puro, 1e-15 * np.ones(10), 1 / 9, floor=1e-12)
check(r.decision == st.SAME, "suelo: el ruido puro no es diferencia")
check(np.array_equal(st.rank_blocks(np.array([[1e-15], [2e-15], [.5]]), floor=1e-12)[0].ravel(), [2.5, 2.5, 1]), "suelo: rangos empatan el ruido")

# Cada test decide con su propio estadístico.
sesgado = np.array([-.01, -.011, -.012, -.013, -.014, -.015, -.016, -.017, -.018, -.019, -.021, .6])
w = st.wilcoxon_test(sesgado, np.zeros(12))
w.p_adjusted = w.p_value
st.decide_frequentist(w, 0.05)
check(abs(w.p_value - 0.0342) < 0.001 and cerca(round(w.effect, 2), -0.69), "Wilcoxon: p y r del caso sesgado")
check(sesgado.mean() > 0 and w.decision == st.B_BETTER and w.direction == -1, f"Wilcoxon decide por rangos ({w.decision})")

# El IC de McNemar depende de α.
A = np.r_[np.ones(30, bool), np.zeros(10, bool), np.ones(25, bool)]
B = np.r_[np.zeros(30, bool), np.ones(10, bool), np.ones(25, bool)]
ic05, ic01 = st.mcnemar_test(A, B, alpha=0.05), st.mcnemar_test(A, B, alpha=0.01)
check(ic01.ci_high - ic01.ci_low > ic05.ci_high - ic05.ci_low, "McNemar: α = 0,01 da un IC más ancho")
check(cerca((ic01.ci_high - ic01.ci_low) / (ic05.ci_high - ic05.ci_low), stats.norm.ppf(.995) / stats.norm.ppf(.975)),
      "McNemar: IC proporcional al valor crítico de α")

# Iman-Davenport sin definir: concordancia perfecta.
perfecto = np.array([np.full(8, .9), np.full(8, .8), np.full(8, .5)])
r = st.friedman_test(perfecto)
check(np.isnan(r.iman_davenport) and np.isnan(r.p_iman_davenport) and np.isfinite(r.p_value), "Iman-Davenport indefinido con concordancia perfecta")
check(any("Iman-Davenport" in x for x in r.warnings), "Iman-Davenport indefinido: avisa")

# Métricas de una partición con una sola clase.
from Orange.evaluation import Results  # noqa: E402
from Orange.evaluation.scoring import LogLoss, Specificity  # noqa: E402

dominio_bin = Domain([ContinuousVariable("x")], DiscreteVariable("c", values=("a", "b")))
una_clase = Results()
una_clase.domain = dominio_bin
una_clase.actual = np.zeros(2)
una_clase.predicted = np.array([[0.0, 1.0]])
una_clase.probabilities = np.array([[[.8, .2], [.8, .2]]])
check(cerca(float(engine._score_fold(LogLoss(), LogLoss, una_clase, {})[0]), 0.2231435513), "LogLoss con una clase: a mano 0,2231")
spec = engine._score_fold(Specificity(), Specificity, una_clase, dict(target=1, average="weighted"))
check(cerca(float(spec[0]), 0.5), "Specificity definida si hay negativos")
spec = engine._score_fold(Specificity(), Specificity, una_clase, dict(target=None, average="weighted"))
check(np.isnan(spec[0]), "Specificity promediada con una sola clase: no definida")

# --------------------------------------------------------------------- banco ----

class Banco:
    SALIDAS = "predictions evaluations_results scores fold_scores comparisons ranks best_learner".split()

    def __init__(self, ajustes=None, **opciones):
        self.w = OWTestScorePro.__new__(OWTestScorePro, signal_manager=None, stored_settings=ajustes)
        self.w.__init__()
        self.salida = {}
        for nombre in self.SALIDAS:
            getattr(self.w.Outputs, nombre).send = self._capturar(nombre)
        # Sin ajustes de un workflow, Orange aplica los guardados por el usuario: se fijan aquí.
        if ajustes is None:
            for nombre, valor in dict(resampling=0, n_folds=3, cv_stratified=True, n_repeats=3, sample_size=9,
                                      shuffle_stratified=True, random_seed=42, parallel_models=True,
                                      comparison_metric="", alpha_index=1, correction_method="holm",
                                      rope_auto=True, rope_value=0.01, hidden_metrics=[]).items():
                setattr(self.w, nombre, valor)
            self.w.alpha_combo.setCurrentIndex(1)
        for nombre, valor in opciones.items():
            setattr(self.w, nombre, valor)
        self.n = 0

    def _capturar(self, nombre):
        def recibir(valor):
            self.salida[nombre] = valor
        return recibir

    def esperar(self):
        inicio = time.time()
        app.processEvents()
        while (self.w.task is not None or self.w._after_task.isActive() or self.w._debounce.isActive()) \
                and time.time() - inicio < 180:
            app.processEvents()
            time.sleep(0.002)
        app.processEvents()
        check(self.w.task is None, "el cálculo termina")

    def datos(self, data, test=None):
        self.w.set_train_data(data)
        if test is not None:
            self.w.set_test_data(test)
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

    def dibujar(self):
        for grafico in self.w.charts:
            grafico.figure(6, 4)
        return self

    def errores(self):
        return [str(m) for m in self.w.Error.active]

    def avisos(self):
        return " ".join(str(m) for m in self.w.Warning.active)

    def cerrar(self):
        self.w.onDeleteWidget()


def sintetico(n=120, clases=2, semilla=0, peso_clase=None):
    r = np.random.RandomState(semilla)
    y = r.choice(clases, n, p=peso_clase).astype(float)
    x = r.normal(size=(n, 4)) + y[:, None]
    dom = Domain([ContinuousVariable(f"x{i}") for i in range(4)],
                 DiscreteVariable("c", values=tuple(f"v{i}" for i in range(clases))),
                 [DiscreteVariable("grupo", values=("g1", "g2", "g3", "g4"))])
    t = Table.from_numpy(dom, x, y, metas=(np.arange(n) % 4)[:, None].astype(float))
    t.name = "sintético"
    return t


def regresion(n=120, semilla=0):
    r = np.random.RandomState(semilla)
    x = r.normal(size=(n, 4))
    y = x @ [1, 2, -1, .5] + r.normal(size=n)
    dom = Domain([ContinuousVariable(f"x{i}") for i in range(4)], ContinuousVariable("y"),
                 [DiscreteVariable("grupo", values=("g1", "g2", "g3", "g4"))])
    t = Table.from_numpy(dom, x, y, metas=(np.arange(n) % 4)[:, None].astype(float))
    t.name = "regresión"
    return t


class Falla(Learner):
    name = "falla"

    def fit_storage(self, data):
        raise ValueError("este modelo siempre falla")


class Contador(LogisticRegressionLearner):
    name = "contador"
    veces = 0
    cerrojo = threading.Lock()

    def __call__(self, data, progress_callback=None):
        with Contador.cerrojo:
            Contador.veces += 1
        return super().__call__(data, progress_callback)


# Comparación con el Test & Score nativo real, muestreo a muestreo.
class Nativo:
    def __init__(self):
        self.w = OWTestAndScore.__new__(OWTestAndScore, signal_manager=None, stored_settings=None)
        self.w.__init__()
        self.salida = {}
        self.w.Outputs.predictions.send = lambda v: self.salida.__setitem__("predictions", v)
        self.w.Outputs.evaluations_results.send = lambda v: self.salida.__setitem__("evaluations_results", v)
        for nombre, valor in dict(resampling=0, n_folds=2, cv_stratified=True, n_repeats=3, sample_size=9,
                                  shuffle_stratified=True, class_selection=OWTestAndScore.TARGET_AVERAGE).items():
            setattr(self.w, nombre, valor)

    def ejecutar(self, data, test, learners):
        self.w.set_train_data(data)
        if test is not None:
            self.w.set_test_data(test)
        for i, learner in enumerate(learners):
            self.w.insert_learner(i, learner)
        self.w.handleNewSignals()
        inicio = time.time()
        while self.w._OWTestAndScore__state.name == "Running" and time.time() - inicio < 120:
            app.processEvents()
            time.sleep(0.002)
        app.processEvents()


clasif = Table("heart_disease")
clasif.name = "heart_disease"
tres = sintetico(150, 3)
reg = regresion(150)
MUESTREOS = {
    "kfold": (OWTestScorePro.KFOLD, dict(n_folds=2, cv_stratified=True)),
    "por característica": (OWTestScorePro.FEATURE, {}),
    "aleatorio": (OWTestScorePro.SHUFFLE, dict(n_repeats=2, sample_size=9, shuffle_stratified=True)),
    "dejar uno fuera": (OWTestScorePro.LOO, {}),
    "entrenamiento": (OWTestScorePro.ON_TRAIN, {}),
    "test": (OWTestScorePro.ON_TEST, {}),
}
CLAS = [LogisticRegressionLearner, NaiveBayesLearner, MajorityLearner]
REGR = [LinearRegressionLearner, MeanLearner, TreeRegression]
for problema, base, modelos in (("clasificación", tres, CLAS), ("regresión", reg, REGR)):
    tren, prueba = base[:110], base[110:]
    for nombre, (tipo, ajustes) in MUESTREOS.items():
        datos_kw = tren[:60] if tipo == OWTestScorePro.LOO else tren
        etiqueta = f"{problema} / {nombre}"
        nativo = Nativo()
        nativo.w.resampling = tipo
        for clave, valor in ajustes.items():
            setattr(nativo.w, clave, valor)
        nativo.ejecutar(datos_kw, prueba if tipo == OWTestScorePro.ON_TEST else None, [m() for m in modelos])

        banco = Banco(resampling=tipo, **ajustes)
        banco.datos(datos_kw, prueba if tipo == OWTestScorePro.ON_TEST else None).modelos(*[m() for m in modelos]).correr().dibujar()
        check(not banco.errores(), f"{etiqueta}: sin errores {banco.errores()}")
        propia, suya = banco.salida.get("evaluations_results"), nativo.salida.get("evaluations_results")
        if not (check(propia is not None and suya is not None, f"{etiqueta}: hay resultados")):
            continue
        check(np.array_equal(propia.predicted, suya.predicted), f"{etiqueta}: mismas predicciones que el nativo")
        check(np.array_equal(propia.row_indices, suya.row_indices), f"{etiqueta}: mismas filas")
        if suya.probabilities is not None:
            check(np.allclose(propia.probabilities, suya.probabilities), f"{etiqueta}: mismas probabilidades")
        check(propia.learner_names == suya.learner_names, f"{etiqueta}: mismos nombres")
        tp, ts = banco.salida["predictions"], nativo.salida["predictions"]
        check(tp.domain == ts.domain and np.array_equal(tp.X, ts.X) and len(tp) == len(ts), f"{etiqueta}: tabla de predicciones")
        for escala, slot in zip(range(len(modelos)), nativo.w.learners.values()):
            for scorer, valor in zip(nativo.w.scorers, slot.stats):
                if valor.success:
                    propio = banco.w._scoreset.overall[scorer.__name__][escala]
                    check(np.isclose(propio, valor.value[0], equal_nan=True), f"{etiqueta}: {scorer.__name__} = nativo")
        informe = banco.w._analysis
        check(len(informe.pairs) > 0, f"{etiqueta}: hay comparaciones")
        check(informe.best >= 0 and banco.salida["best_learner"] is not None, f"{etiqueta}: hay mejor modelo")
        check(banco.salida["scores"].X.shape[0] == len(modelos), f"{etiqueta}: tabla de puntuaciones")
        check(banco.salida["ranks"] is not None and banco.salida["comparisons"] is not None, f"{etiqueta}: rangos y comparaciones")
        tabla_folds = banco.salida["fold_scores"]
        if tipo in engine.FOLD_KINDS:
            check(tabla_folds is not None and len(tabla_folds) == len(modelos) * len(propia.folds), f"{etiqueta}: tabla por particiones")
            check(st.BAYES in informe.available and st.TTEST_CORRECTED in informe.available and informe.primary == st.TTEST_CORRECTED,
                  f"{etiqueta}: tests por particiones")
            check(informe.friedman is not None and informe.friedman.n_blocks > 0, f"{etiqueta}: Friedman")
            check(np.isfinite(informe.ci_low).all(), f"{etiqueta}: intervalos")
        else:
            check(tabla_folds is None, f"{etiqueta}: sin tabla por particiones")
            check(st.TTEST_CORRECTED not in informe.available, f"{etiqueta}: sin t por particiones")
            esperado = st.MCNEMAR if problema == "clasificación" else st.ROW_WILCOXON
            check(informe.primary == esperado, f"{etiqueta}: test principal por filas ({informe.primary})")
        if tipo == OWTestScorePro.SHUFFLE and problema == "clasificación":
            check(st.MCNEMAR not in informe.available, f"{etiqueta}: sin McNemar (filas repetidas)")
        if tipo == OWTestScorePro.ON_TRAIN:
            check("optimistas" in banco.avisos(), f"{etiqueta}: avisa de que es optimista")
        banco.cerrar()
        nativo.w.onDeleteWidget()

# Compatibilidad con los widgets de evaluación que consumen la salida.
banco = Banco(n_folds=2).datos(clasif).modelos(LogisticRegressionLearner(), TreeLearner()).correr()
resultados = banco.salida["evaluations_results"]
for clase in (OWConfusionMatrix, OWROCAnalysis):
    receptor = clase.__new__(clase, signal_manager=None, stored_settings=None)
    receptor.__init__()
    receptor.set_results(resultados)
    receptor.handleNewSignals()
    app.processEvents()
    check(not list(receptor.Error.active), f"{clase.__name__} acepta la salida")
    receptor.onDeleteWidget()

# Clase objetivo: mismos valores que el nativo con esa clase.
nativo = Nativo()
nativo.w.n_folds = 2
nativo.ejecutar(tres, None, [LogisticRegressionLearner(), NaiveBayesLearner()])
nativo.w.class_selection = "v1"
nativo.w.update_stats_model()
banco = Banco(n_folds=2).datos(tres).modelos(LogisticRegressionLearner(), NaiveBayesLearner()).correr()
banco.w.class_selection = "v1"
banco.w._scoreset = None
banco.w._start_analysis()
banco.esperar()
modelo = nativo.w.score_table.model
check(modelo.columnCount() > 4, "clase objetivo: el nativo ha dado puntuaciones")
for columna in range(3, modelo.columnCount()):
    nombre = modelo.horizontalHeaderItem(columna).text()
    scorer = next(s for s in nativo.w.scorers if s.name == nombre or s.long_name == nombre)
    valor = float(modelo.item(0, columna).data(0))
    check(np.isclose(banco.w._scoreset.overall[scorer.__name__][0], valor), f"clase objetivo: {nombre} = nativo")
por_fold = banco.w._scoreset.folds["CA"].mean(axis=1)
check(np.isfinite(por_fold).all(), "clase objetivo: particiones")
banco.cerrar()
nativo.w.onDeleteWidget()

# Rigor: el t corregido sale de las particiones reales, con su ρ.
banco = Banco(n_folds=2).datos(clasif).modelos(LogisticRegressionLearner(), TreeLearner()).correr()
scoreset, informe = banco.w._scoreset, banco.w._analysis
check(cerca(scoreset.rho, 1 / 5) and cerca(scoreset.test_train_ratio, 1 / 4), "k=5: ρ=1/5 y n_test/n_train=1/4")
pareja = informe.pairs_of(st.TTEST_CORRECTED)[0]
filas = scoreset.folds["AUC"]
esperado = st.corrected_ttest(filas[0], filas[1], 1 / 4)
check(cerca(pareja.p_value, esperado.p_value), "t corregido sobre las particiones reales")
banco.w.set_train_data(clasif)
banco.modelos(NaiveBayesLearner()).correr()
informe = banco.w._analysis
for test in (st.TTEST_CORRECTED, st.WILCOXON):
    parejas = informe.pairs_of(test)
    check(np.allclose([p.p_adjusted for p in parejas], multipletests([p.p_value for p in parejas], method="holm")[1]),
          f"{test}: el p ajustado es el de Holm")
scoreset = banco.w._scoreset
filas = scoreset.folds["AUC"]
pareja = next(p for p in informe.pairs_of(st.TTEST_CORRECTED) if (p.a, p.b) == (0, 1))
check(cerca(pareja.p_value, st.corrected_ttest(filas[0], filas[1], 1 / 4).p_value), "t corregido con 3 modelos")
ref = baycomp.two_on_single(filas[0], filas[1], rope=informe.rope)
bayes = next(p for p in informe.pairs_of(st.BAYES) if (p.a, p.b) == (0, 1))
check(cerca(bayes.p_a, ref[0]), "bayesiano sobre las particiones reales = baycomp")
banco.cerrar()
banco = Banco(resampling=OWTestScorePro.SHUFFLE, n_repeats=3, sample_size=9).datos(clasif).modelos(LogisticRegressionLearner(), TreeLearner()).correr()
scoreset = banco.w._scoreset
n_test = len(clasif) - round(len(clasif) * .66)
check(abs(scoreset.rho - n_test / len(clasif)) < 0.01, f"aleatorio: ρ es la fracción de test ({scoreset.rho})")
banco.cerrar()

# Orientación de la diferencia: siempre métrica(A) − métrica(B).
vivienda = Table("housing")
banco = Banco(n_folds=2, comparison_metric="RMSE").datos(vivienda).modelos(LinearRegressionLearner(), MeanLearner()).correr()
informe, scoreset = banco.w._analysis, banco.w._scoreset
filas = scoreset.folds["RMSE"]
d = filas[0] - filas[1]
for test in (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.BAYES):
    pareja = informe.pairs_of(test)[0]
    check(cerca(pareja.mean_diff, d.mean()) and pareja.mean_diff < 0, f"{test}: diferencia es RMSE(A) − RMSE(B)")
pareja = informe.pairs_of(st.TTEST_CORRECTED)[0]
check(pareja.ci_low < d.mean() < pareja.ci_high and pareja.ci_high < 0, "IC en la misma orientación")
esperado = st.corrected_ttest(-filas[0], -filas[1], scoreset.test_train_ratio)
check(cerca(pareja.statistic, -esperado.statistic), "estadístico en la misma orientación")
check(pareja.decision == st.A_BETTER and pareja.effect < 0, "RMSE: A mejor aunque la diferencia sea negativa")
comparar = banco.salida["comparisons"]
columna = comparar.get_column("Diferencia (A − B)")
check(cerca(columna[0], d.mean()), "salida: Diferencia (A − B)")
ruta = os.path.join(DIR, "orientacion.html")
banco.w.export_report(ruta)
banco.esperar()
contenido = open(ruta, encoding="utf-8").read()
check("Diferencia (A − B)" in contenido and "valor negativo significa que A es mejor" in contenido, "informe: orientación explícita")
banco.w.details_combo.setCurrentIndex(banco.w.details_combo.findData(st.TTEST_CORRECTED))
banco.w._fill_details()
check("Diferencia (A − B)" in banco.w.details_model.headers and "negativo" in banco.w.notes_browser.toPlainText(), "tabla: orientación explícita")
banco.cerrar()

# McNemar y los tests por filas dicen sobre qué aportan evidencia.
banco = Banco(resampling=OWTestScorePro.LOO, comparison_metric="AUC").datos(clasif[:60]).modelos(LogisticRegressionLearner(), MajorityLearner()).correr()
informe = banco.w._analysis
texto = " ".join(informe.recommendation)
check("Supera con evidencia" not in texto, f"AUC con McNemar: no afirma evidencia sobre AUC ({texto})")
check("no mide AUC" in texto or "no en AUC" in texto, f"AUC con McNemar: lo aclara ({texto})")
check(informe.pairs_of(st.MCNEMAR)[0].evidence == "aciertos por fila", "McNemar: evidencia sobre aciertos")
check("aciertos por fila" in st.interpret(informe.pairs_of(st.MCNEMAR)[0], informe.names, "AUC", .05), "interpretación: nombra los aciertos")
banco.w.comparison_metric = "CA"
banco.w._options_changed()
banco.esperar()
texto = " ".join(banco.w._analysis.recommendation)
check("no mide" not in texto, f"CA con McNemar: la evidencia sí es sobre la métrica ({texto})")
banco.cerrar()
banco = Banco(resampling=OWTestScorePro.LOO, comparison_metric="RMSE").datos(reg[:60]).modelos(LinearRegressionLearner(), MeanLearner()).correr()
informe = banco.w._analysis
texto = " ".join(informe.recommendation)
check("Supera con evidencia" not in texto and "no mide RMSE" in texto, f"RMSE por filas: lo aclara ({texto})")
pareja = informe.pairs_of(st.ROW_WILCOXON)[0]
check(pareja.evidence == "error absoluto por fila" and pareja.mean_diff < 0, "por filas: error absoluto A − B")
banco.w.comparison_metric = "MAE"
banco.w._options_changed()
banco.esperar()
check("Supera con evidencia" in " ".join(banco.w._analysis.recommendation), "MAE por filas: sí es evidencia sobre MAE")
banco.cerrar()

# La recomendación no da por buena una varianza de ruido.
banco = Banco(n_folds=3).datos(clasif).modelos(LogisticRegressionLearner(), MajorityLearner()).correr()
scoreset = banco.w._scoreset
a = np.array([14, 13, 15, 13, 14, 14, 13, 12, 14, 13]) / 15
b = np.array([13, 12, 14, 12, 13, 13, 12, 11, 13, 12]) / 15
scoreset.folds["CA"] = np.vstack((a, b))
scoreset.overall["CA"] = np.array([a.mean(), b.mean()])
opciones = engine.TestOptions(metric="CA")
con_ruido = engine.run_tests(scoreset, banco.w._sampling_used, opciones)
scoreset.folds["CA"] = np.vstack((a, a - 1 / 15))
sin_ruido = engine.run_tests(scoreset, banco.w._sampling_used, opciones)
check(con_ruido.recommendation == sin_ruido.recommendation, "recomendación igual con y sin ruido")
check("No se ha podido evaluar" in " ".join(con_ruido.recommendation) and "Supera" not in " ".join(con_ruido.recommendation),
      "recomendación: varianza nula no es evidencia")
banco.cerrar()

# Ajuste perfecto: RMSE de 1e-15, las diferencias son ruido numérico.
semilla_p = np.random.RandomState(9)
X = semilla_p.normal(size=(200, 3))
perfecta = Table.from_numpy(Domain([ContinuousVariable(f"x{i}") for i in range(3)], ContinuousVariable("y")),
                            X, X @ [1, 2, -1] + 5)
banco = Banco(comparison_metric="RMSE").datos(perfecta).modelos(LinearRegressionLearner(), RidgeRegressionLearner(alpha=1e-13)).correr()
informe = banco.w._analysis
check(np.nanmax(banco.w._scoreset.folds["RMSE"]) < 1e-12, "ajuste perfecto: RMSE casi cero")
for test in (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.WILCOXON):
    pareja = informe.pairs_of(test)[0]
    check(pareja.decision == st.SAME and pareja.p_value == 1, f"ajuste perfecto: {test} no ve diferencia ({pareja.decision}, {pareja.p_value})")
bayes = informe.pairs_of(st.BAYES)[0]
check(bayes.p_a < .95 and bayes.p_b < .95, f"ajuste perfecto: el bayesiano no decide por ruido ({bayes.p_a}, {bayes.p_b})")
check("Supera" not in " ".join(informe.recommendation), "ajuste perfecto: sin evidencia en la recomendación")
banco.cerrar()
banco = Banco(comparison_metric="MSE").datos(perfecta).modelos(LinearRegressionLearner(), RidgeRegressionLearner(alpha=1e-13)).correr()
check(banco.w._analysis.pairs_of(st.WILCOXON)[0].decision == st.SAME, "ajuste perfecto: tampoco con MSE")
banco.cerrar()
banco = Banco(comparison_metric="RMSE").datos(perfecta).modelos(LinearRegressionLearner(), MeanLearner()).correr()
check(banco.w._analysis.pairs_of(st.TTEST_CORRECTED)[0].decision == st.A_BETTER, "ajuste perfecto: una diferencia real sigue viéndose")
banco.cerrar()
banco = Banco(resampling=OWTestScorePro.LOO, comparison_metric="MAE").datos(perfecta[:40]).modelos(LinearRegressionLearner(), RidgeRegressionLearner(alpha=1e-13)).correr()
check(banco.w._analysis.pairs_of(st.ROW_WILCOXON)[0].decision == st.SAME, "ajuste perfecto: por filas tampoco")
banco.cerrar()
for nombre, valor in (("RMSE", 1e-13 * 12), ("MAE", 1e-13 * 12), ("AUC", 1e-13), ("MAPE", 1e-11), ("Desconocida", 0.0)):
    check(cerca(engine.noise_floor(type(nombre, (), {}), np.array([0.5, -12.0])), valor, 1e-20), f"suelo de {nombre}")
delta = 1e-13 * 12
valores = np.array([0.0, 4.0, 9410.0])
check(np.allclose(engine.noise_floor(type("MSE", (), {}), np.array([.5, -12.0]), valores), 2 * delta * np.sqrt(valores) + delta ** 2, atol=0),
      "suelo de MSE: 2·δ·RMSE + δ²")

# MSE con y muy desplazada: el suelo no oculta diferencias reales.
aleatorio_mse = np.random.RandomState(21)
X = aleatorio_mse.normal(size=(300, 3))
atributos_mse = [ContinuousVariable(f"x{i}") for i in range(3)]


def tabla_y(y):
    return Table.from_numpy(Domain(atributos_mse, ContinuousVariable("y")), X, y)


def comparar_mse(y, segundo, etiqueta):
    """Evalúa con MSE, y comprueba los tests contra el cálculo directo sobre las particiones."""
    banco = Banco(comparison_metric="MSE").datos(tabla_y(y)).modelos(LinearRegressionLearner(), segundo).correr()
    scoreset, informe = banco.w._scoreset, banco.w._analysis
    filas = scoreset.folds["MSE"]
    d = filas[0] - filas[1]
    t_manual = d.mean() / np.sqrt(d.var(ddof=1) * (1 / len(d) + scoreset.test_train_ratio))
    pareja = informe.pairs_of(st.TTEST_CORRECTED)[0]
    check(cerca(pareja.statistic, t_manual, 1e-6 * abs(t_manual)) and pareja.decision != st.SAME, f"{etiqueta}: t = {t_manual:.3f} ({pareja.statistic}, {pareja.decision})")
    wilcoxon = informe.pairs_of(st.WILCOXON)[0]
    referencia = stats.wilcoxon(-d, method="exact").pvalue
    check(cerca(wilcoxon.p_value, referencia, 1e-12) and wilcoxon.p_value < 1, f"{etiqueta}: Wilcoxon = scipy ({wilcoxon.p_value})")
    check(not any("empat" in x for x in wilcoxon.warnings), f"{etiqueta}: sin empates falsos {wilcoxon.warnings}")
    bayes = informe.pairs_of(st.BAYES)[0]
    check(bayes.p_rope < 1 and bayes.decision != "equivalentes", f"{etiqueta}: el bayesiano no las da por equivalentes")
    banco.cerrar()
    return pareja


senal = X @ [1, 2, -1] / 3
ruido_mse = aleatorio_mse.normal(size=300)
y = 1.7e9 + 1000 * senal + 100 * ruido_mse
pareja = comparar_mse(y, TreeRegression(max_depth=4), "MSE con y desplazada, árbol 4")
check(pareja.decision == st.A_BETTER and pareja.p_value < 1e-3, f"MSE desplazada: gana la lineal ({pareja.p_value})")
comparar_mse(y, TreeRegression(max_depth=2), "MSE con y desplazada, árbol 2")
centinela = y.copy()
centinela[0] = 999999999.0
comparar_mse(centinela, TreeRegression(max_depth=4), "MSE con centinela")
for escala in (1e-9, 1e9):
    comparar_mse(escala * (1000 * senal + 100 * ruido_mse), TreeRegression(max_depth=4), f"MSE con escala {escala}")
for desplazamiento in (1e6, 1e8, 1e10):
    comparar_mse(desplazamiento + 1000 * senal + 100 * ruido_mse, TreeRegression(max_depth=4), f"MSE desplazada {desplazamiento:g}")
# RMSE y MAE tienen un error de redondeo que no crece con su valor: mismo suelo.
banco = Banco(comparison_metric="RMSE").datos(tabla_y(y)).modelos(LinearRegressionLearner(), TreeRegression(max_depth=4)).correr()
check(banco.w._analysis.pairs_of(st.TTEST_CORRECTED)[0].decision == st.A_BETTER, "RMSE con y desplazada: gana la lineal")
banco.cerrar()

# Los textos dicen lo que es cierto.
casi = np.array([.8, .9, .85, .7, .75])
ruido = casi + 1e-15
for prueba in (st.corrected_ttest(casi, ruido, .25), st.wilcoxon_test(casi, ruido)):
    check(prueba.numeric_tie and prueba.decision == st.SAME and "precisión numérica" in " ".join(prueba.warnings), f"{prueba.test}: igualdad numérica")
    prueba.a, prueba.b = 0, 1
    check("precisión numérica" in st.interpret(prueba, ["A", "B"], "CA", .05) and "exactamente" not in st.interpret(prueba, ["A", "B"], "CA", .05),
          f"{prueba.test}: la interpretación no dice «exactamente»")
for prueba in (st.corrected_ttest(casi, casi.copy(), .25), st.wilcoxon_test(casi, casi.copy())):
    prueba.a, prueba.b = 0, 1
    check(not prueba.numeric_tie and "idénticas" in " ".join(prueba.warnings) and "exactamente" in st.interpret(prueba, ["A", "B"], "CA", .05), f"{prueba.test}: igualdad exacta")
w = st.wilcoxon_test(np.array([.5, .3, .9, .4, .7]), np.array([.5 + 1e-15, .2, .7, .1, .3]))
check(any("precisión numérica" in x for x in w.warnings), "Wilcoxon: empate por precisión, lo dice")
w = st.wilcoxon_test(np.array([.5, .3, .9, .4, .7]), np.array([.5, .2, .7, .1, .3]))
check(any("diferencia cero" in x for x in w.warnings), "Wilcoxon: empate exacto, lo dice")
# Suelo por partición alineado con las particiones descartadas.
r = st.corrected_ttest(np.array([.8, np.nan, .7, .6]), np.array([.7, .5, .65, .5]), .1, floor=np.array([1e-13, 1e-13, 1e-13, 1e-13]))
check(r.n == 3 and np.isfinite(r.p_value), "suelo por partición con valores sin definir")

# Casos reales con escalas extremas.
semilla = np.random.RandomState(0)
X = semilla.normal(size=(150, 3))
y = X @ [1, 2, -1] + semilla.normal(size=150) * .5 + 10
y[7] = 1e-15
atributos = [ContinuousVariable(f"x{i}") for i in range(3)]
extrema = Table.from_numpy(Domain(atributos, ContinuousVariable("y")), X, y)
banco = Banco(comparison_metric="MAPE").datos(extrema).modelos(LinearRegressionLearner(), MeanLearner()).correr()
scoreset, informe = banco.w._scoreset, banco.w._analysis
pareja = informe.pairs_of(st.WILCOXON)[0]
crudas = -(scoreset.folds["MAPE"][0] - scoreset.folds["MAPE"][1])
referencia = stats.wilcoxon(crudas, method="exact").pvalue
check(cerca(pareja.p_value, referencia, 1e-12) and pareja.p_value < 1, f"MAPE con y≈0: Wilcoxon real ({pareja.p_value})")
check(pareja.n == 10 and np.abs(crudas).max() > 1e10, "MAPE con y≈0: la escala es extrema de verdad")
check(not any("empat" in x for x in pareja.warnings), f"MAPE con y≈0: sin empates falsos {pareja.warnings}")
banco.cerrar()


class Divergente(Learner):
    name = "divergente"

    class Modelo(Orange.base.Model):
        def predict(self, X):
            return np.full(len(X), 1e6)

    def fit_storage(self, data):
        return Divergente.Modelo(data.domain)


X = semilla.normal(size=(200, 3))
y = np.clip(.5 + .15 * (X @ [1, 2, -1]) + semilla.normal(size=200) * .05, 0, 1)
acotada = Table.from_numpy(Domain(atributos, ContinuousVariable("y")), X, y)
banco = Banco(comparison_metric="MSE").datos(acotada).modelos(LinearRegressionLearner(), MeanLearner(), Divergente()).correr()
informe = banco.w._analysis
check(list(informe.mean_ranks) == [1.0, 2.0, 3.0], f"modelo divergente: rangos 1, 2, 3 ({list(informe.mean_ranks)})")
nemenyi = next(p for p in informe.pairs_of(st.NEMENYI) if (p.a, p.b) == (0, 1))
check(nemenyi.p_value < 1 and nemenyi.decision != st.SAME, f"modelo divergente: Nemenyi no empata ({nemenyi.p_value})")
corregido = next(p for p in informe.pairs_of(st.TTEST_CORRECTED) if (p.a, p.b) == (0, 1))
check(corregido.p_value < 0.05 and corregido.decision == st.A_BETTER, "modelo divergente: el t corregido ve la diferencia")
banco.cerrar()

# McNemar dice que su efecto es una razón.
banco = Banco(resampling=OWTestScorePro.LOO, comparison_metric="AUC").datos(clasif[:60]).modelos(LogisticRegressionLearner(), TreeLearner(), MajorityLearner()).correr()
banco.w.details_combo.setCurrentIndex(banco.w.details_combo.findData(st.MCNEMAR))
banco.w._fill_details()
check("Razón A/B" in banco.w.details_model.headers and "Efecto (A − B)" not in banco.w.details_model.headers, "McNemar: la columna dice razón A/B")
check(banco.w._analysis.pairs_of(st.MCNEMAR)[0].effect_name == "razón de discordancias (A/B)", "McNemar: nombre del efecto")
texto = " ".join(banco.w._analysis.recommendation)
check("comete menos errores" not in texto, f"redacción natural ({texto})")
if "supera a" in texto:
    check("En aciertos por fila" in texto, "McNemar: redacción de la evidencia")
ruta = os.path.join(DIR, "razon.html")
banco.w.export_report(ruta)
banco.esperar()
check("razón de discordancias (A/B)" in open(ruta, encoding="utf-8").read(), "McNemar: el informe dice que es una razón")
banco.cerrar()

# Los modelos que llegan desde el lienzo: varios son Fitter, sin atributo `params`.
class ParametrosRotos(LogisticRegressionLearner):
    name = "parámetros rotos"

    @property
    def params(self):
        raise RuntimeError("no se pueden leer")

    @params.setter
    def params(self, _value):
        pass


for problema, datos_fit, ajeno in (("clasificación", clasif, "C=1.0"), ("regresión", reg, "C=1.0")):
    modelos_fit = [modelling.SVMLearner(), modelling.TreeLearner(), modelling.RandomForestLearner(n_estimators=5),
                   modelling.KNNLearner(), modelling.ConstantLearner()]
    nativo = Nativo()
    nativo.ejecutar(datos_fit, None, [modelling.SVMLearner(), modelling.KNNLearner()])
    banco = Banco(n_folds=2, parallel_models=False).datos(datos_fit).modelos(modelling.SVMLearner(), modelling.KNNLearner()).correr()
    check(np.array_equal(banco.salida["evaluations_results"].predicted, nativo.salida["evaluations_results"].predicted),
          f"Fitter {problema}: mismas predicciones que el nativo")
    banco.cerrar()
    nativo.w.onDeleteWidget()
    banco = Banco(n_folds=1).datos(datos_fit).modelos(*modelos_fit).correr().dibujar()
    check(not banco.errores() and not banco.avisos().count("ha fallado"), f"Fitter {problema}: evaluación sin fallos {banco.avisos()}")
    ruta = os.path.join(DIR, f"fitter_{problema}.html")
    banco.w.export_report(ruta)
    banco.esperar()
    check(os.path.exists(ruta) and not banco.errores(), f"Fitter {problema}: informe exportado {banco.errores()}")
    check(ajeno in open(ruta, encoding="utf-8").read(), f"Fitter {problema}: los parámetros salen en el informe")
    check(not banco.w.cancel_button.isVisibleTo(banco.w) and banco.w.report_button2.isEnabled(), f"Fitter {problema}: botones tras exportar")
    banco.w.send_report()
    check(len(banco.w.report_html) > 2000, f"Fitter {problema}: informe nativo de Orange")
    banco.cerrar()
check(builder.learner_parameters(modelling.SVMLearner(), clasif.domain).startswith("C=1.0"), "Fitter: parámetros del tipo de problema")
check(builder.learner_parameters(ParametrosRotos(), clasif.domain) == "—", "parámetros ilegibles: no rompen el informe")
check(builder.learner_parameters(modelling.SVMLearner()) == "—", "Fitter sin dominio: no rompe el informe")

banco = Banco(n_folds=1).datos(clasif).modelos(ParametrosRotos(), NaiveBayesLearner()).correr()
ruta = os.path.join(DIR, "rotos.html")
banco.w.export_report(ruta)
banco.esperar()
check(os.path.exists(ruta) and not banco.errores(), "parámetros ilegibles: se exporta el informe")
original_contexto = banco.w._report_context
banco.w._report_context = lambda: 1 / 0
banco.w.export_report(os.path.join(DIR, "fallo.html"))
check(banco.w.Error.report.is_shown() and banco.w.task is None, "fallo al preparar el informe: mensaje en el widget")
check(not banco.w.cancel_button.isVisibleTo(banco.w) and banco.w.report_button2.isEnabled(), "fallo al preparar el informe: botones intactos")
banco.w._report_context = original_contexto
banco.w.export_report(ruta)
banco.esperar()
check(not banco.w.Error.report.is_shown(), "tras un fallo se puede volver a exportar")
banco.cerrar()

# Modelos con el mismo nombre: igual que el nativo.
nativo = Nativo()
nativo.ejecutar(clasif, None, [LogisticRegressionLearner(), LogisticRegressionLearner()])
banco = Banco(n_folds=2).datos(clasif).modelos(LogisticRegressionLearner(), LogisticRegressionLearner()).correr()
propia, suya = banco.salida["evaluations_results"], nativo.salida["evaluations_results"]
check(propia.learner_names == suya.learner_names == ["logistic regression"] * 2, f"mismo nombre: learner_names ({propia.learner_names})")
check(banco.salida["predictions"].domain == nativo.salida["predictions"].domain, "mismo nombre: columnas iguales al nativo")
check(banco.w._analysis.names[1].endswith("(2)"), "mismo nombre: la vista los distingue")
banco.cerrar()
nativo.w.onDeleteWidget()

# Iman-Davenport en iris con modelos idénticos, tal como aparecía.
iris = Table("iris")
banco = Banco(n_folds=2).datos(iris).modelos(LogisticRegressionLearner(), LogisticRegressionLearner(), MajorityLearner()).correr()
fried = banco.w._analysis.friedman
check(np.isnan(fried.p_iman_davenport) and np.isfinite(fried.p_value) and fried.p_value > 0, f"iris: sin p = 0 falso ({fried.p_iman_davenport}, {fried.p_value})")
ruta = os.path.join(DIR, "iman.html")
banco.w.export_report(ruta)
banco.esperar()
contenido = open(ruta, encoding="utf-8").read()
check("Iman-Davenport no está definido" in contenido, "iris: el informe lo explica")

# Diagrama de diferencias críticas: lleva su aviso.
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots  # noqa: E402

figura = plots.critical_difference_figure(banco.w._analysis, 8, 4, plots.LIGHT)
check(any("no son independientes" in texto.get_text() for texto in figura.texts), "diagrama: avisa en la propia figura")
banco.w._analysis.friedman.on_folds = False
figura = plots.critical_difference_figure(banco.w._analysis, 8, 4, plots.LIGHT)
check(not any("no son independientes" in texto.get_text() for texto in figura.texts), "diagrama: sin aviso si no son particiones")
banco.w._analysis.friedman.on_folds = True
for ancho, alto in ((6.89, 4.6), (4.0, 3.0), (9.0, 5.0)):
    figura = plots.critical_difference_figure(banco.w._analysis, ancho, alto, plots.LIGHT)
    figura.canvas.draw()
    pie = next(t for t in figura.texts if "no son independientes" in t.get_text())
    caja = pie.get_window_extent(figura.canvas.get_renderer())
    w_fig, h_fig = figura.canvas.get_width_height()
    check(caja.x0 >= 0 and caja.x1 <= w_fig and caja.y0 >= 0, f"pie entero a {ancho} pulgadas ({caja.x0:.0f}-{caja.x1:.0f} de {w_fig})")
    ejes = figura.axes[0].get_tightbbox(figura.canvas.get_renderer())
    check(caja.y1 <= ejes.y0 + 6, f"el pie no pisa el diagrama a {ancho} pulgadas ({caja.y1:.0f}, {ejes.y0:.0f})")
banco.cerrar()

# n_train real con muestreo por característica y valores perdidos.
base_grupos = clasif.transform(Domain(clasif.domain.attributes, clasif.domain.class_var,
                                      [DiscreteVariable("g", values=tuple("abcde"))]))
with base_grupos.unlocked():
    base_grupos.metas[:, 0] = np.arange(len(base_grupos)) % 5
    base_grupos.metas[:40, 0] = np.nan
banco = Banco(resampling=OWTestScorePro.FEATURE).datos(base_grupos).modelos(LogisticRegressionLearner(), MajorityLearner()).correr()
scoreset = banco.w._scoreset
check(abs(scoreset.test_train_ratio - 0.25) < 0.001, f"por característica: ratio real 0,25 ({scoreset.test_train_ratio:.3f})")
check(int(scoreset.fold_sizes.sum()) == 263, "por característica: solo cuenta las filas usadas")
banco.cerrar()

# LogLoss se calcula aunque una partición tenga una sola clase.
rara = sintetico(40)
with rara.unlocked():
    rara.Y[:] = np.r_[np.zeros(39), 1]
banco = Banco(n_folds=2).datos(rara).modelos(MajorityLearner(), NaiveBayesLearner()).correr()
scoreset = banco.w._scoreset
check(scoreset.undefined["LogLoss"] == 0 and np.isfinite(scoreset.folds["LogLoss"]).all(), "LogLoss definida en todas las particiones")
check(scoreset.undefined["AUC"] > 0, "AUC sigue sin definir donde no lo está")
banco.cerrar()

# Otras métricas de error: menor es mejor y el mejor modelo sale bien.
banco = Banco(n_folds=2, comparison_metric="RMSE").datos(reg).modelos(MeanLearner(), LinearRegressionLearner()).correr()
informe = banco.w._analysis
check(informe.lower_better and informe.names[informe.best].startswith("linear"), "RMSE: gana la regresión lineal")
pareja = informe.pairs_of(st.TTEST_CORRECTED)[0]
check(pareja.decision == st.B_BETTER and pareja.mean_diff > 0, f"RMSE: B (lineal) mejor ({pareja.decision})")
check(informe.rope > 0.001 and informe.rope != 0.01, "RMSE: ROPE relativo a la escala")
banco.cerrar()

# ------------------------------------------------------------------ anomalías ----

def error_de(tabla, *textos, **opciones):
    banco = Banco(**opciones).datos(tabla).modelos(LogisticRegressionLearner()).correr()
    mensaje = " ".join(banco.errores())
    check(mensaje, f"debe dar error con {textos}: {banco.avisos()}")
    for texto in textos:
        check(texto in mensaje, f"{texto!r} no está en {mensaje!r}")
    check(banco.salida.get("evaluations_results") is None, f"sin salida con {textos}")
    banco.cerrar()


error_de(clasif[:0], "vacía")
error_de(clasif.transform(Domain(clasif.domain.attributes)), "variable objetivo")
error_de(clasif.transform(Domain([], clasif.domain.class_var)), "atributos")
dos_clases = Domain(clasif.domain.attributes[:3], [clasif.domain.class_var, DiscreteVariable("z", values=("a", "b"))])
error_de(Table.from_numpy(dos_clases, np.zeros((6, 3)), np.zeros((6, 2))), "varias variables objetivo")
constante = sintetico(30)
with constante.unlocked():
    constante.Y[:] = 0
error_de(constante, "un valor")
toda_nan = sintetico(30)
with toda_nan.unlocked():
    toda_nan.Y[:] = np.nan
error_de(toda_nan, "ningún valor")
error_de(sintetico(4), "particiones", n_folds=3)  # 10 particiones con 4 filas

perdidos = sintetico(60)
with perdidos.unlocked():
    perdidos.Y[:5] = np.nan
banco = Banco(n_folds=1).datos(perdidos).modelos(LogisticRegressionLearner()).correr()
check("5 filas" in banco.avisos() and banco.salida["evaluations_results"] is not None, "filas sin clase: se quitan y avisa")
banco.cerrar()

rara = sintetico(40)
with rara.unlocked():
    rara.Y[:] = np.r_[np.zeros(39), 1]
banco = Banco(n_folds=3).datos(rara).modelos(MajorityLearner(), NaiveBayesLearner()).correr().dibujar()
check("estratificar" in banco.avisos() and banco.salida["evaluations_results"] is not None, "estratificación imposible: sigue y avisa")
indefinidas = banco.w._scoreset.undefined["AUC"]
check(indefinidas > 0, f"AUC indefinido en particiones con una sola clase ({indefinidas})")
check("no está definida" in banco.avisos(), "AUC indefinido: se indica")
banco.cerrar()
banco = Banco(resampling=OWTestScorePro.SHUFFLE, n_repeats=1, sample_size=2).datos(rara).modelos(MajorityLearner()).correr()
check("estratificar" in banco.avisos() and banco.salida["evaluations_results"] is not None, "aleatorio: estratificación imposible")
banco.cerrar()

banco = Banco(resampling=OWTestScorePro.FEATURE).datos(clasif).modelos(LogisticRegressionLearner()).correr()
check(banco.errores() and "atributo" in " ".join(banco.errores()), "por característica sin atributo: mensaje")
banco.cerrar()
falta = sintetico(60)
falta_dom = Domain(falta.domain.attributes, falta.domain.class_var, [DiscreteVariable("grupo", values=("g1", "g2"))])
falta = Table.from_numpy(falta_dom, falta.X, falta.Y, metas=np.zeros((60, 1)))
banco = Banco(resampling=OWTestScorePro.FEATURE).datos(falta).modelos(LogisticRegressionLearner()).correr()
check("dos valores" in " ".join(banco.errores()), "por característica con un solo valor: mensaje")
banco.cerrar()

# Modelo que falla: no tumba a los demás y se indica cuál y por qué.
banco = Banco(n_folds=1).datos(clasif).modelos(LogisticRegressionLearner(), Falla(), TreeLearner()).correr().dibujar()
check("falla" in banco.avisos() and "siempre falla" in banco.avisos(), "modelo que falla: se indica")
check(banco.w._analysis is not None and len(banco.w._analysis.names) == 2, "modelo que falla: siguen los demás")
check(banco.salida["evaluations_results"].predicted.shape[0] == 2, "modelo que falla: salida con los demás")
banco.cerrar()
banco = Banco(n_folds=1).datos(clasif).modelos(Falla()).correr()
check("falla" in banco.avisos() and banco.salida["evaluations_results"] is None, "todos fallan: sin salida")
banco.cerrar()
banco = Banco(n_folds=1).datos(reg).modelos(LogisticRegressionLearner(), LinearRegressionLearner()).correr()
check("logistic regression" in banco.avisos() and banco.salida["evaluations_results"] is not None, "modelo incompatible con regresión: se indica")
banco.cerrar()

# Datos de test.
banco = Banco(resampling=OWTestScorePro.ON_TEST).datos(clasif).modelos(LogisticRegressionLearner()).correr()
check("Falta" in banco.avisos() and banco.salida["evaluations_results"] is None, "sin datos de test: avisa")
banco.cerrar()
banco = Banco(resampling=OWTestScorePro.ON_TEST).datos(clasif, tres).modelos(LogisticRegressionLearner()).correr()
check("objetivo distintas" in " ".join(banco.errores()), "test con otra clase: mensaje")
banco.cerrar()
banco = Banco(n_folds=1).datos(clasif, clasif).modelos(LogisticRegressionLearner()).correr()
check("no se usan" in banco.avisos(), "datos de test sin usar: avisa")
banco.cerrar()
otro = clasif.transform(Domain(clasif.domain.attributes[:6], clasif.domain.class_var))
banco = Banco(resampling=OWTestScorePro.ON_TEST).datos(clasif, otro).modelos(LogisticRegressionLearner()).correr()
check(banco.w.Information.test_transformed.is_shown(), "test con otros atributos: se indica")
banco.cerrar()
sin_filas = Banco(resampling=OWTestScorePro.ON_TEST).datos(clasif)
sin_filas.w.set_test_data(clasif[:0])
sin_filas.modelos(LogisticRegressionLearner()).correr()
check("vacía" in " ".join(sin_filas.errores()), "test vacío: mensaje")
sin_filas.cerrar()

# Modelos idénticos, uno solo, dos, muchos; datos dispersos.
banco = Banco(n_folds=2).datos(clasif).modelos(LogisticRegressionLearner(), LogisticRegressionLearner()).correr().dibujar()
check(banco.w._analysis.names == ["logistic regression", "logistic regression (2)"], "modelos idénticos: nombres únicos")
pareja = banco.w._analysis.pairs_of(st.TTEST_CORRECTED)[0]
check(pareja.p_value == 1 and pareja.decision == st.SAME, "modelos idénticos: p=1")
check(banco.w._analysis.friedman.n_models == 2 and st.NEMENYI not in banco.w._analysis.available, "dos modelos: sin Friedman")
banco.cerrar()
banco = Banco(n_folds=1).datos(clasif).modelos(LogisticRegressionLearner()).correr().dibujar()
check(banco.w._analysis is not None and not banco.w._analysis.pairs, "un modelo: sin comparaciones")
check(banco.salida["comparisons"] is None and banco.salida["best_learner"] is not None, "un modelo: salidas coherentes")
banco.cerrar()
muchos = [MajorityLearner() if i % 3 == 0 else NaiveBayesLearner() for i in range(25)]
banco = Banco(n_folds=1).datos(tres).modelos(*muchos).correr().dibujar()
check(len(banco.w._analysis.names) == 25 and len(banco.salida["scores"]) == 25, "25 modelos")
check(banco.w.task is None and not banco.errores(), "25 modelos: sin errores")
banco.cerrar()
continuo = sintetico(150).to_sparse()
banco = Banco(n_folds=1).datos(continuo).modelos(LogisticRegressionLearner(), NaiveBayesLearner()).correr()
check(banco.salida.get("evaluations_results") is not None, f"datos dispersos: hay resultado {banco.avisos()}")
check(not banco.errores(), f"datos dispersos: sin errores {banco.errores()}")
banco.cerrar()

# Reutilización: las opciones no reentrenan y un modelo nuevo solo entrena el nuevo.
Contador.veces = 0
banco = Banco(n_folds=1).datos(clasif).modelos(Contador(), NaiveBayesLearner()).correr()
base = Contador.veces
check(base == 3, f"entrena una vez por partición ({base})")
banco.w.comparison_metric = "CA"
banco.w._options_changed()
banco.esperar()
check(banco.w._analysis.metric == "CA" and Contador.veces == base, "cambiar la métrica no reentrena")
banco.w.alpha_index = 2
banco.w._options_changed()
banco.esperar()
check(banco.w._analysis.alpha == 0.1 and Contador.veces == base, "cambiar α no reentrena")
banco.w.correction_method = "bonferroni"
banco.w._options_changed()
banco.esperar()
check(Contador.veces == base, "cambiar la corrección no reentrena")
banco.modelos(TreeLearner()).correr()
check(Contador.veces == base and len(banco.w._analysis.names) == 3, "añadir un modelo no reentrena los otros")
banco.w.remove_learner(2)
banco.n -= 1
banco.correr()
check(Contador.veces == base and len(banco.w._analysis.names) == 2, "quitar un modelo no reentrena")
banco.cerrar()

# Paralelo = secuencial, y el reparto solo se usa cuando compensa.
check(engine.default_workers(3, None, 100) == 1, "datos pequeños: sin hilos")
check(engine.default_workers(3, object(), 100000) == 1, "con preprocesador compartido: sin hilos")
modelos_p = [("a", LogisticRegressionLearner()), ("b", NaiveBayesLearner()), ("c", TreeLearner()), ("d", Falla())]
muestreo = engine.Sampling(k=3, seed=42)
uno = engine.evaluate(clasif, None, modelos_p, muestreo, workers=1)
varios = engine.evaluate(clasif, None, modelos_p, muestreo, workers=3)
for clave in "abc":
    check(np.array_equal(uno[clave].results.predicted, varios[clave].results.predicted), f"paralelo = secuencial ({clave})")
check(varios["d"].error is not None and uno["d"].error is not None, "un fallo en paralelo no afecta a los demás")

# Cambios durante el cálculo y cancelación.
grande = sintetico(30000, 2)
banco = Banco(n_folds=4).datos(grande).modelos(RandomForestLearner(n_estimators=30), NaiveBayesLearner())
banco.w.handleNewSignals()
check(banco.w.task is not None and "evaluations_results" not in banco.salida, "evalúa en segundo plano")
time.sleep(0.05)
app.processEvents()
banco.w.set_train_data(clasif)
banco.w.handleNewSignals()
banco.esperar()
check(len(banco.salida["predictions"]) == len(clasif), "un cambio de datos descarta el cálculo anterior")
banco.w.set_train_data(grande)
banco.w.handleNewSignals()
inicio = time.perf_counter()
banco.w.cancel_evaluation()
check(banco.w.task is None and time.perf_counter() - inicio < 1, "cancelar es inmediato")
time.sleep(0.2)
app.processEvents()
check(banco.w.task is None, "tras cancelar no sigue el cálculo")
banco.cerrar()

# Salida de tablas.
banco = Banco(n_folds=2).datos(clasif).modelos(LogisticRegressionLearner(), TreeLearner(), NaiveBayesLearner()).correr().dibujar()
puntos = banco.salida["scores"]
check(all(v.name in [a.name for a in puntos.domain.attributes] for v in [ContinuousVariable("AUC"), ContinuousVariable("Rango medio"), ContinuousVariable("Entrenamiento (s)")]), "scores: columnas")
folds = banco.salida["fold_scores"]
check(folds.domain.class_var.name == "Modelo" and len(folds) == 15, "fold_scores: una fila por partición y modelo")
modelo_0 = folds[folds.Y == 0]
check(np.allclose(modelo_0.get_column("AUC"), banco.w._scoreset.folds["AUC"][0]), "fold_scores: valores por modelo")
comparar = banco.salida["comparisons"]
check("p ajustado" in [a.name for a in comparar.domain.attributes] and "Decisión" in [m.name for m in comparar.domain.metas], "comparisons: columnas")
check(banco.salida["ranks"].get_column("Posición").min() == 1, "ranks: posiciones")

# Informe.
ruta = os.path.join(DIR, "informe.html")
banco.w.export_report(ruta)
banco.esperar()
contenido = open(ruta, encoding="utf-8").read()
for texto in ("Datos y muestreo", "Modelos", "Puntuaciones", "Gráficos", "Comparaciones estadísticas", "Reproducibilidad",
              "Nadeau", "McNemar", "Semilla", "scikit-learn", "baycomp", "concibieron", "Supuesto"):
    check(texto in contenido, f"informe: incluye {texto}")
check(contenido.count("data:image/png;base64,") >= 4, "informe: gráficos incrustados")
check("http://" not in contenido and "https://" not in contenido and "src='/" not in contenido, "informe: autocontenido")
pdf = os.path.join(DIR, "informe.pdf")
banco.w.export_report(pdf)
banco.esperar()
with open(pdf, "rb") as stream:
    check(stream.read(5) == b"%PDF-" and os.path.getsize(pdf) > 100000, "informe en PDF con los gráficos")
check(not list(banco.w.Error.active), "informe: sin errores")
banco.w.export_report("/ruta/que/no/existe/informe.html")
banco.esperar()
check(banco.w.Error.report.is_shown(), "informe: ruta no válida da mensaje")
banco.w.Error.report.clear()
banco.w.send_report()
check(len(banco.w.report_html) > 2000, "informe nativo de Orange")
imagen = os.path.join(DIR, "g.svg")
banco.w.chart_cd.write_image(imagen)
check(os.path.getsize(imagen) > 1000, "exportar imagen")
banco.w.tabs.setCurrentIndex(3)
banco.w.show()
app.processEvents()
time.sleep(0.3)
app.processEvents()
banco.w.grab()
banco.w.scores_proxy.sort(1)
check(banco.w.scores_proxy.rowCount() == 3, "tabla ordenable")

# Estado con el workflow.
banco.w.comparison_metric = "CA"
banco.w.alpha_index = 0
banco.w.resampling = OWTestScorePro.SHUFFLE
banco.w.correction_method = "fdr_bh"
banco.w.rope_auto = False
banco.w.rope_value = 0.05
banco.w.random_seed = 7
ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.cerrar()
nuevo = Banco(ajustes=ajustes).datos(clasif).modelos(LogisticRegressionLearner(), NaiveBayesLearner()).correr()
check(nuevo.w.resampling == OWTestScorePro.SHUFFLE and nuevo.w.random_seed == 7, "se restaura el muestreo")
informe = nuevo.w._analysis
check(informe.metric == "CA" and informe.alpha == 0.01 and informe.correction == "fdr_bh" and informe.rope == 0.05, "se restauran las opciones")
nuevo.cerrar()

check(not excepciones, "sin excepciones sin capturar:\n" + "\n".join(excepciones))
print("\nFALLOS:", fallos)
sys.exit(1 if fallos else 0)
