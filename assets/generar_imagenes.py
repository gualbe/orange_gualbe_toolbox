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
