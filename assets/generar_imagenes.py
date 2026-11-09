"""Regenera la captura del widget SMOTE con el Python de Orange.

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
from Orange.data import Table  # noqa: E402

app = QApplication([])

from orangecontrib.orange_gualbe_toolbox.widgets.owsmote import OWSmote  # noqa: E402

w = OWSmote()
w.show()
w.k_neighbors = 3
w.set_data(Table("zoo"))
while w.task is not None:
    app.processEvents()
    time.sleep(0.01)
app.processEvents()
# Orange restaura la geometría guardada en show(): resize() va después.
w.resize(980, 600)
app.processEvents()
w.grab().save(os.path.join(AQUI, "captura_smote.png"))
w.onDeleteWidget()
