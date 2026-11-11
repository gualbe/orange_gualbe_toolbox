# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Regenera las capturas de los widgets con el Python de Orange.

Uso (desde cualquier carpeta):
    QT_QPA_PLATFORM=offscreen QT_SCALE_FACTOR=2 \
    /Applications/Orange.app/Contents/MacOS/python assets/generar_imagenes.py
"""
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

from AnyQt.QtWidgets import QApplication  # noqa: E402
import Orange  # noqa: E402
from Orange.data import Table  # noqa: E402

app = QApplication([])

from orangecontrib.orange_gualbe_toolbox.widgets.owfileimport import OWFileImport  # noqa: E402
from orangecontrib.orange_gualbe_toolbox.widgets.owsmote import OWSmote  # noqa: E402
from orangecontrib.orange_gualbe_toolbox.widgets.owtestscorepro import OWTestScorePro  # noqa: E402
from Orange.classification import (  # noqa: E402
    LogisticRegressionLearner, MajorityLearner, NaiveBayesLearner, RandomForestLearner, TreeLearner,
)


def esperar(widget):
    def pendiente(nombre):
        temporizador = getattr(widget, nombre, None)
        return temporizador is not None and temporizador.isActive()

    while widget.task is not None or pendiente("_after_preview") or pendiente("_after_task"):
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def capturar(widget, nombre, ancho, alto):
    # Orange restaura la geometría guardada en show(): resize() va después.
    widget.resize(ancho, alto)
    app.processEvents()
    widget.grab().save(os.path.join(AQUI, nombre))
    widget.onDeleteWidget()


smote = OWSmote()
smote.show()
smote.k_neighbors = 3
smote.set_data(Table("zoo"))
esperar(smote)
capturar(smote, "captura_smote.png", 980, 600)

importar = OWFileImport()
importar.show()
importar.advanced_toggle.setChecked(False)
importar.recent_paths.clear()
importar.open_path(os.path.join(os.path.dirname(Orange.__file__), "datasets", "heart_disease.tab"))
esperar(importar)
capturar(importar, "captura_fileimport.png", 1040, 640)

pro = OWTestScorePro()
pro.show()
pro.set_train_data(Table("heart_disease"))
for posicion, modelo in enumerate((LogisticRegressionLearner(), RandomForestLearner(), NaiveBayesLearner(),
                                   TreeLearner(), MajorityLearner())):
    pro.insert_learner(posicion, modelo)
pro.handleNewSignals()
esperar(pro)
capturar(pro, "captura_testscorepro.png", 1120, 660)


# --------------------------------------------------- ventana deslizante ----
import numpy as np  # noqa: E402
from Orange.data import ContinuousVariable, Domain, TimeVariable  # noqa: E402
from Orange.regression import LinearRegressionLearner, MeanLearner, RandomForestRegressionLearner  # noqa: E402

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_plots as plots  # noqa: E402
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_window as tw  # noqa: E402


def pasajeros_multihorizonte(retardos=3, horizontes=3):
    """Pasajeros aéreos con retardos como atributos y y(t+1)…y(t+h) como objetivos."""
    from orangecontrib.timeseries import Timeseries
    serie = Timeseries.from_file("airpassengers")
    y = serie.Y.ravel()
    filas = range(retardos, len(y) - horizontes)
    X = np.array([[y[t - k] for k in range(retardos)] for t in filas])
    Y = np.array([[y[t + h] for h in range(1, horizontes + 1)] for t in filas])
    fecha = serie.time_variable
    dominio = Domain([ContinuousVariable(f"pasajeros t-{k}" if k else "pasajeros t") for k in range(retardos)],
                     [ContinuousVariable(f"pasajeros t+{h}") for h in range(1, horizontes + 1)], [fecha])
    tabla = Table.from_numpy(dominio, X, Y, metas=serie.get_column(fecha)[list(filas)][:, None])
    tabla.name = "Pasajeros aéreos"
    return tabla


ventana = OWTestScorePro()
ventana.show()
for nombre, valor in dict(resampling=OWTestScorePro.WINDOW, window_start_mode=tw.START_FIRST,
                          window_initial_mode=tw.INITIAL_PERCENT, window_initial_percent=50,
                          window_advance=tw.EXPANDING, window_unit=tw.UNIT_TIME, window_time_unit=5,
                          window_test=12, window_gap=0, window_step_same=True, window_end_mode=tw.END_ALL,
                          window_last=tw.LAST_DISCARD, window_min_train=10, window_max_iterations=200,
                          comparison_metric="RMSE", current_tab=1).items():
    setattr(ventana, nombre, valor)
ventana._sync_window_values()
ventana._sync_window_controls()
ventana.set_train_data(pasajeros_multihorizonte())
for posicion, modelo in enumerate((LinearRegressionLearner(), RandomForestRegressionLearner(random_state=0),
                                   MeanLearner())):
    ventana.insert_learner(posicion, modelo)
ventana.handleNewSignals()
esperar(ventana)
while ventana._window_timer.isActive() or ventana._debounce.isActive() or ventana.task is not None:
    app.processEvents()
    time.sleep(0.01)
ventana.tabs.setCurrentWidget(ventana.windows_page)
capturar_ventana = ventana
capturar_ventana.resize(1240, 900)
app.processEvents()
time.sleep(0.3)
app.processEvents()
ventana.chart_windows.redraw()
ventana.grab().save(os.path.join(AQUI, "captura_ventana.png"))
ventana.tabs.setCurrentWidget(ventana.chart_horizons)
app.processEvents()
time.sleep(0.3)
app.processEvents()
ventana.chart_horizons.redraw()
ventana.grab().save(os.path.join(AQUI, "captura_horizontes.png"))
ventana.onDeleteWidget()

# Dibujo de los dos modos de avance para la ayuda.
filas = Table.from_numpy(Domain([ContinuousVariable("x")], ContinuousVariable("y")),
                         np.arange(60.0)[:, None], np.arange(60.0))
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
modos = (("Creciente: inicio fijo", tw.WindowSpec(initial_mode=tw.INITIAL_ROWS, initial_rows=24, test_size=8,
                                                     advance=tw.EXPANDING, min_train=1)),
         ("Deslizante: tamaño fijo", tw.WindowSpec(initial_mode=tw.INITIAL_ROWS, initial_rows=24, test_size=8,
                                                      advance=tw.ROLLING, min_train=1)),
         ("Con hueco de 3 y paso de 12", tw.WindowSpec(initial_mode=tw.INITIAL_ROWS, initial_rows=24, test_size=8,
                                                          gap=3, step=12, step_same=False, min_train=1)))
for nombre, especificacion in modos:
    figura = plots.windows_figure(tw.plan_windows(filas, especificacion), 4.2, 2.3, plots.LIGHT, title=False)
    figura.suptitle(nombre, fontsize=11, x=0.02, ha="left")
    ruta = os.path.join(AQUI, "ventana_" + nombre.split(":")[0].split(" ")[0].lower() + ".png")
    figura.savefig(ruta, dpi=150, facecolor="white")
