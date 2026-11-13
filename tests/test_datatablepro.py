# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Pruebas dirigidas de Data Table Pro.

Script ejecutable con el Python de Orange; sale con 1 si alguna comprobación falla.
"""
import os
import sys
import time
import traceback
import warnings
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QEvent, QItemSelection, QItemSelectionModel, Qt
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, TimeVariable

from orangecontrib.orange_gualbe_toolbox.widgets import _dtp_engine as engine
from orangecontrib.orange_gualbe_toolbox.widgets.owdatatablepro import (MAX_OUTPUTS, OWDataTablePro, SCOPE_ALL,
                                                                         SCOPE_COLUMNS, SCOPE_ROWS)

warnings.simplefilter("ignore")
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []


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


AJUSTES = dict(find_regex=False, find_case=False, find_scope=SCOPE_ALL, affix_scope=SCOPE_COLUMNS,
               select_full_rows=True, show_bars=False, color_categories=False, highlight_edits=True,
               auto_commit=True, stored_changes={})


class Banco:
    def __init__(self, ajustes=None):
        self.w = OWDataTablePro.__new__(OWDataTablePro, signal_manager=None, stored_settings=ajustes)
        self.w.__init__()
        if ajustes is None:  # sin workflow, Orange aplicaría los ajustes guardados del usuario
            for nombre, valor in AJUSTES.items():
                setattr(self.w, nombre, valor)
            self.w._selection_mode()
        self.salida, self.envios = {}, {}
        for i in range(1, MAX_OUTPUTS + 1):
            for nombre in (f"data_{i}", f"selected_data_{i}"):
                getattr(self.w.Outputs, nombre).send = self._capturar(nombre)

    def _capturar(self, nombre):
        def recibir(valor):
            self.salida[nombre] = valor
            self.envios[nombre] = self.envios.get(nombre, 0) + 1
        return recibir

    def titulos(self, *titulos):
        """Simula los enlaces del lienzo: el título de cada nodo de origen, en orden."""
        self.w.__dict__["_OWBaseWidget__process_signal_input"] = [
            SimpleNamespace(channel=SimpleNamespace(name="Data"), link=SimpleNamespace(source_node=SimpleNamespace(title=t)))
            for t in titulos]
        return self

    def meter(self, *tablas):
        for tabla in tablas:
            self.w.insert_data(len(self.w.pages), tabla)
        self.w.handleNewSignals()
        app.processEvents()
        return self

    def pagina(self, i=0):
        self.w.tabs.setCurrentIndex(i)
        return self.w.pages[i]

    def enviar(self):
        self.w.commit.now()
        return self

    def errores(self):
        return " ".join(str(m) for m in self.w.Error.active)

    def avisos(self):
        return " ".join(str(m) for m in self.w.Warning.active)

    def info(self):
        return " ".join(str(m) for m in self.w.Information.active)

    def cerrar(self):
        self.w.onDeleteWidget()
        self.w.hide()
        self.w.deleteLater()
        app.sendPostedEvents(None, QEvent.DeferredDelete)


def mixta(n=8):
    """Numérica, categórica, fecha, texto (meta), clase y desconocidos."""
    dominio = Domain([ContinuousVariable("edad"), DiscreteVariable("color", values=("rojo", "verde", "azul")),
                      TimeVariable("alta", have_date=1)],
                     DiscreteVariable("clase", values=("no", "sí")),
                     [StringVariable("nombre"), ContinuousVariable("peso")])
    X = np.array([[30 + i, i % 3, 1.6e9 + i * 86400] for i in range(n)], dtype=float)
    X[2, 0] = np.nan
    X[5, 1] = np.nan
    Y = np.array([i % 2 for i in range(n)], dtype=float)
    metas = np.array([[f"Persona {i}", 50.0 + i] for i in range(n)], dtype=object)
    metas[3, 0] = ""
    tabla = Table.from_numpy(dominio, X, Y, metas)
    tabla.name = "mixta"
    return tabla


iris, titanic, zoo, housing = Table("iris"), Table("titanic"), Table("zoo"), Table("housing")

# ------------------------------------------------------- entradas y salidas ----

check([o.name for o in OWDataTablePro.get_signals("outputs")] ==
      ["Selected Data 1", "Data 1", "Selected Data 2", "Data 2"] and MAX_OUTPUTS == 2,
      "el widget declara exactamente dos pares de salidas")
banco = Banco().titulos("Flores", "Barco", "Animales").meter(iris, titanic, zoo)
check([banco.w.tabs.tabText(i) for i in range(3)] == ["① Flores", "② Barco", "— Animales"],
      "una pestaña por entrada, con su número de salida; la tercera, sin número")
check([banco.salida[f"data_{i}"].name for i in (1, 2)] == ["Flores", "Barco"],
      "Data N en el orden de las pestañas, con el título del nodo como nombre")
check("3 tablas" in banco.avisos() and "dos primeras tablas tienen salida" in banco.avisos(),
      "más de dos entradas: aviso con el texto ajustado")
check("Sin salida" in banco.w.tabs.tabToolTip(2), "la tercera pestaña dice que no tiene salida")
check(all(banco.salida[f"selected_data_{i}"] is None for i in range(1, MAX_OUTPUTS + 1)), "sin selección, Selected Data N es None")
tercera = banco.pagina(2)
tercera.model.setData(tercera.model.index(0, 0), "x")
check(tercera.state.log, "la tercera pestaña se puede editar")
sin_titulo = Banco().meter(iris)
check(sin_titulo.salida["data_1"].name == "iris" and sin_titulo.w.tabs.tabText(0) == "① iris",
      "sin título de nodo, se usa el nombre de la tabla")
sin_titulo.cerrar()
salida = banco.salida["data_2"]
check(salida.domain == titanic.domain and np.array_equal(salida.X, titanic.X) and np.array_equal(salida.Y, titanic.Y)
      and np.array_equal(salida.ids, titanic.ids), "sin cambios, Data N es igual a la entrada (dominio, valores e ids)")
check("Data 2 y Selected Data 2" in banco.w.tabs.tabToolTip(1), "la pestaña dice qué salidas le corresponden")
banco.w._save_all()
tres_ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.titulos("Barco", "Animales")  # Orange actualiza los enlaces antes de avisar al widget
banco.w.remove_data(0)
banco.w.handleNewSignals()
primera_columna = zoo.domain.attributes[0]
check(banco.salida["data_1"].name == "Barco" and banco.salida["data_2"].name == "Animales" and
      "x" in banco.salida["data_2"].domain[primera_columna.name].values,
      "al quitar un enlace, la tercera pestaña sube a la salida 2 con sus cambios")
check([banco.w.tabs.tabText(i) for i in range(2)] == ["① Barco", "② Animales ✎"] and not banco.avisos(),
      "los distintivos se renumeran y desaparece el aviso")
banco.titulos("Animales")
banco.w.remove_data(0)
banco.w.handleNewSignals()
check(banco.salida["data_1"].name == "Animales" and banco.salida["data_2"] is None, "y la salida que queda libre se vacía")
banco.cerrar()
restaurado = Banco(tres_ajustes).titulos("Flores", "Barco", "Animales").meter(iris, titanic, zoo)
check(not restaurado.errores() and restaurado.w.pages[2].state.log and "3 tablas" in restaurado.avisos(),
      "los ajustes guardados con más de dos pestañas se restauran sin errores")
restaurado.cerrar()

muchas = Banco().meter(*[iris] * 4)
check([muchas.salida[f"data_{i}"].name for i in range(1, MAX_OUTPUTS + 1)] == ["iris", "iris (2)"] and
      [muchas.w.tabs.tabText(i) for i in range(4)] == ["① iris", "② iris (2)", "— iris (3)", "— iris (4)"],
      "identificadores únicos con la misma tabla repetida")
muchas.w.remove_data(3)
muchas.w.remove_data(2)
muchas.w.handleNewSignals()
check(not muchas.avisos(), "con dos entradas desaparece el aviso")
muchas.cerrar()

vacia = Banco().meter(iris[:0], None)
check(vacia.salida["data_1"] is not None and len(vacia.salida["data_1"]) == 0 and vacia.salida["data_2"] is None
      and vacia.w.tabs.count() == 2 and not vacia.errores(), "tabla vacía y entrada sin datos: sin errores")
vacia.cerrar()

# ----------------------------------------------------- columnas y filas ----

banco = Banco().meter(mixta())
pagina = banco.pagina(0)
cabecera = pagina.view.horizontalHeader()
estado = pagina.state
indice = {c.cid: i for i, c in enumerate(estado.columns)}
cabecera.moveSection(cabecera.visualIndex(indice["alta"]), 0)
cabecera.moveSection(cabecera.visualIndex(indice["peso"]), 1)
cabecera.moveSection(cabecera.visualIndex(indice["clase"]), 2)
banco.enviar()
dominio = banco.salida["data_1"].domain
check([v.name for v in dominio.attributes] == ["alta", "edad", "color"] and dominio.class_var.name == "clase" and
      [v.name for v in dominio.metas] == ["peso", "nombre"],
      "reordenar columnas arrastrando: nuevo orden en cada grupo, respetando atributos, clase y metas")

pagina.view.horizontalHeader().sectionClicked.emit(indice["edad"])
banco.enviar()
edades = banco.salida["data_1"].get_column("edad")
check(np.all(np.diff(edades[:-1]) >= 0) and np.isnan(edades[-1]), "pulsar el nombre ordena ascendente; desconocidos al final")
pagina.view.horizontalHeader().sectionClicked.emit(indice["edad"])
banco.enviar()
edades = banco.salida["data_1"].get_column("edad")
check(np.all(np.diff(edades[:-1]) <= 0) and np.isnan(edades[-1]), "pulsar otra vez ordena descendente")
check(banco.salida["data_1"].get_column("nombre")[0] == "Persona 7", "las demás columnas acompañan a su fila")
pagina.view.selectRow(0)
app.processEvents()
banco.enviar()
check(banco.salida["selected_data_1"] is not None and banco.salida["selected_data_1"].get_column("edad")[0] == 37,
      "Selected Data respeta el orden de filas")
banco.w.restore_order()
banco.enviar()
check(np.array_equal(pagina.state.order, np.arange(8)) and banco.salida["data_1"].get_column("edad")[0] == 30 and
      pagina.view.horizontalHeader().sortIndicatorSection() == -1, "orden original restaurado")
estable = engine.TableState(titanic)
estable.sort("survived")
estable.sort("status")
datos = estable.output()
pares = list(zip(datos.get_column("status"), datos.get_column("survived")))
check(pares == sorted(pares), "ordenar es estable: la última columna manda y la anterior desempata")

# ------------------------------------------------------------ edición ----

pagina.view.clearSelection()
modelo = pagina.model
fila_edad = modelo.index(0, indice["edad"])
check(modelo.setData(fila_edad, "41,5") and estado.column("edad").values[0] == 41.5,
      "editar una numérica (acepta coma decimal)")
check(not modelo.setData(fila_edad, "cuarenta") and "no es un número válido" in banco.avisos(),
      "un número no válido se rechaza con un aviso claro")
check(modelo.setData(modelo.index(1, indice["color"]), "morado") and "morado" in estado.column("color").categories,
      "un valor nuevo en una categórica amplía sus valores")
check(modelo.setData(modelo.index(2, indice["alta"]), "2025-01-31") and
      estado.column("alta").text(2) == "2025-01-31", "editar una fecha")
check(not modelo.setData(modelo.index(2, indice["alta"]), "31/31/2025"), "una fecha no válida se rechaza")
check(modelo.setData(modelo.index(4, indice["nombre"]), "Ana") and modelo.setData(modelo.index(0, indice["edad"]), "?")
      and np.isnan(estado.column("edad").values[0]), "editar texto y dejar un valor desconocido")
banco.enviar()
salida = banco.salida["data_1"]
check(salida.get_column("nombre")[4] == "Ana" and "morado" in salida.domain["color"].values,
      "las ediciones llegan a Data N, también los valores nuevos de la categórica")
check(salida.domain["edad"] is mixta().domain["edad"] or salida.domain["edad"].name == "edad",
      "una columna sin cambios de definición conserva su variable")
check(" ✎" in banco.w.tabs.tabText(0), "la pestaña marca que tiene cambios")
pagina.view.setFocus()
banco.w.show()
banco.w.activateWindow()
app.processEvents()
QTest.keyClick(pagina.view, Qt.Key_Z, Qt.ControlModifier)
app.processEvents()
deshecho = estado.column("edad").values[0] == 41.5
if not deshecho:
    banco.w.undo()
check(estado.column("edad").values[0] == 41.5, f"Ctrl+Z deshace la última edición (atajo de teclado: {deshecho})")
QTest.keyClick(pagina.view, Qt.Key_Y, Qt.ControlModifier)
app.processEvents()
if not np.isnan(estado.column("edad").values[0]):
    banco.w.redo()
check(np.isnan(estado.column("edad").values[0]), "Ctrl+Y rehace")
for _ in range(10):
    banco.w.undo()
check(not estado.log and estado.column("color").categories == ["rojo", "verde", "azul"] and
      estado.column("edad").values[0] == 30, "deshacer todo devuelve la tabla original")
banco.w.redo()
check(estado.column("edad").values[0] == 41.5, "rehacer tras deshacerlo todo")
banco.cerrar()

# ------------------------------------------------ buscar y reemplazar ----

banco = Banco().meter(titanic.copy())
pagina = banco.pagina(0)
banco.w.find_edit.setText("CREW")
check(banco.w.find_all() == 885 and pagina.model.current is not None and "de 885" in banco.w.find_status.text(),
      "buscar sin distinguir mayúsculas")
banco.w.find_case = True
check(banco.w.find_all() == 0 and "Ninguna" in banco.w.find_status.text(), "distinguiendo mayúsculas")
banco.w.find_case = False
banco.w.find_all()
primera = pagina.model.current
banco.w.step_match(1)
segunda = pagina.model.current
banco.w.step_match(-1)
check(primera != segunda and pagina.model.current == primera and pagina.model.matches["status"].sum() == 885,
      "resalta las coincidencias y va a la siguiente y a la anterior")
pagina.sort("age", True)
check(pagina.model.matches["status"].sum() == 885 and pagina.model.current == primera,
      "ordenar conserva el resaltado de las coincidencias")
pagina.restore_order()
banco.w.find_regex = True
banco.w.find_edit.setText(r"^(\w)(\w+)$")
banco.w.replace_edit.setText(r"\2-\1")
total, celdas, _ = banco.w.replace_all()
check(total == 2201 * 4 and "8804 reemplazos" in banco.w.find_status.text(),
      "reemplazar con expresión regular y grupos: cuenta los reemplazos")
check(pagina.state.column("status").categories == ["rew-c", "irst-f", "econd-s", "hird-t"] and
      "ale-m" in pagina.state.column("sex").categories, "los grupos se aplican en el reemplazo")
banco.w.undo()
banco.w.find_regex = False
banco.w.find_edit.setText("crew")
banco.w.replace_edit.setText("tripulación")
pagina.select_column(pagina.state.index_of("status"))
banco.w.find_scope = SCOPE_COLUMNS
total, celdas, _ = banco.w.replace_all()
check(celdas == 885 and pagina.state.column("status").categories[0] == "tripulación" and
      "crew" not in pagina.state.column("status").categories, "ámbito: columnas seleccionadas")
banco.w.undo()
pagina.view.clearSelection()
pagina.view.selectRow(0)
pagina.view.selectionModel().select(pagina.model.index(1, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
banco.w.find_scope = SCOPE_ROWS
banco.w.find_edit.setText("adult")
banco.w.replace_edit.setText("mayor")
total, celdas, _ = banco.w.replace_all()
check(celdas == 2 and int((pagina.state.column("age").texts() == "mayor").sum()) == 2, "ámbito: filas seleccionadas")
banco.w.find_scope = SCOPE_ALL
pagina.view.clearSelection()
banco.w.find_edit.setText("[")
banco.w.find_regex = True
check(banco.w.find_all() is None and "expresión regular no es válida" in banco.errores(), "regex no válida: error claro")
banco.cerrar()

numeros = engine.TableState(housing)
total, celdas, rechazadas = numeros.replace("0", "x", rows=None, cids=["CRIM"])
check(celdas == 0 and rechazadas.get("CRIM", 0) > 0, "en una numérica, un reemplazo que no da un número se rechaza")
total, celdas, _ = numeros.replace(r"^(\d+)\.(\d)", r"\1\2", regex=True, cids=["RM"])
check(celdas > 400 and numeros.column("RM").values.max() > 1000, "en una numérica, un reemplazo que da un número se aplica")

# ------------------------------------------------------ prefijo y sufijo ----

banco = Banco().meter(mixta())
pagina = banco.pagina(0)
pagina.select_column(pagina.state.index_of("nombre"))
banco.w.prefix_edit.setText("Sr. ")
banco.w.suffix_edit.setText(" (ES)")
banco.w.affix_scope = SCOPE_COLUMNS
celdas, omitidas = banco.w.apply_affix()
check(celdas == 7 and pagina.state.column("nombre").values[0] == "Sr. Persona 0 (ES)" and
      pagina.state.column("nombre").values[3] == "", "prefijo y sufijo en una columna de texto; los vacíos no se tocan")
banco.w.affix_scope = SCOPE_ALL
celdas, omitidas = banco.w.apply_affix()
check(set(omitidas) == {"edad", "alta", "peso"} and "No se han tocado las columnas numéricas" in banco.info() and
      pagina.state.column("color").categories[0] == "Sr. rojo (ES)", "toda la tabla: categóricas sí, numéricas no, con aviso")
pagina.view.clearSelection()
pagina.view.selectRow(0)
banco.w.affix_scope = SCOPE_ROWS
banco.w.prefix_edit.setText("#")
banco.w.suffix_edit.setText("")
banco.w.apply_affix()
check(pagina.state.column("clase").text(pagina.model.rows[0]).startswith("#"), "filas seleccionadas")
banco.cerrar()

# --------------------------------------------------- nombre, tipo y rol ----

banco = Banco().meter(mixta())
pagina = banco.pagina(0)


def actual(nombre):
    pagina.view.clearSelection()
    pagina.view.setCurrentIndex(pagina.model.index(0, pagina.state.index_of(nombre)))
    banco.w._sync_controls()


actual("edad")
banco.w.name_edit.setText("años")
banco.w.rename_column()
check(pagina.state.column("edad").name == "años", "renombrar")
banco.w.name_edit.setText("color")
banco.w.rename_column()
check("Ya hay una columna" in banco.errores(), "un nombre repetido se rechaza")
actual("nombre")
banco.w.change_kind(banco.w.kind_combo.findData(engine.CATEGORICAL))
check(pagina.state.column("nombre").kind == engine.CATEGORICAL, "texto a categórica")
banco.w.change_role(banco.w.role_combo.findData(engine.TARGET))
actual("color")
banco.w.change_role(banco.w.role_combo.findData(engine.TARGET))
banco.enviar()
check([v.name for v in banco.salida["data_1"].domain.class_vars] == ["color", "clase", "nombre"] or
      {v.name for v in banco.salida["data_1"].domain.class_vars} == {"color", "clase", "nombre"},
      "varios objetivos a la vez")
banco.w.change_role(banco.w.role_combo.findData(engine.META))
banco.enviar()
check("color" in [v.name for v in banco.salida["data_1"].domain.metas], "devolver una columna a meta")
actual("color")
banco.w.change_role(banco.w.role_combo.findData(engine.ATTRIBUTE))
actual("peso")
banco.w.change_kind(banco.w.kind_combo.findData(engine.STRING))
check(pagina.state.column("peso").kind == engine.STRING and pagina.state.column("peso").role == engine.META,
      "a texto: pasa a meta")
banco.w.change_role(banco.w.role_combo.findData(engine.TARGET))
check("solo puede ser meta" in banco.errores(), "una columna de texto no puede ser objetivo")
with_letters = Table.from_numpy(Domain([], metas=[StringVariable("v")]), np.empty((4, 0)),
                                metas=np.array([["1"], ["2,5"], ["tres"], [""]], dtype=object))
banco.w.insert_data(1, with_letters)
banco.w.handleNewSignals()
pagina = banco.pagina(1)
actual("v")
perdidos = banco.w.change_kind(banco.w.kind_combo.findData(engine.NUMERIC))
check(perdidos == {"v": 1} and "v: 1" in banco.info() and list(pagina.state.column("v").values[:2]) == [1.0, 2.5],
      "conversión fallida: dice cuántos valores pasan a desconocidos y en qué columna")
fechas = engine.TableState(Table.from_numpy(Domain([], metas=[StringVariable("f")]), np.empty((3, 0)),
                                            metas=np.array([["2026-11-13"], ["ayer"], ["2026-01-02"]], dtype=object)))
check(fechas.convert("f", engine.DATETIME) == 1 and fechas.column("f").text(0) == "2026-11-13", "texto a fecha, con fallos contados")
check(fechas.convert("f", engine.CATEGORICAL) == 0 and fechas.column("f").categories == ["2026-01-02", "2026-11-13"],
      "fecha a categórica")
banco.cerrar()

# ------------------------------------------------- selección y celdas ----

banco = Banco().meter(iris)
pagina = banco.pagina(0)
banco.w.select_full_rows = False
banco.w._selection_mode()
seleccion = pagina.view.selectionModel()
for fila in (3, 4):
    for columna in (0, 4):
        seleccion.select(pagina.model.index(fila, columna), QItemSelectionModel.Select)
app.processEvents()
banco.enviar()
parcial = banco.salida["selected_data_1"]
check(parcial is not None and len(parcial) == 2 and [v.name for v in parcial.domain.variables] == ["sepal length", "iris"],
      "sin filas completas, Selected Data lleva solo las columnas seleccionadas")
banco.w.copy_selection()
check(QApplication.clipboard().text().splitlines()[0] == "sepal length\tiris", "copiar al portapapeles como texto tabulado")
pagina.view.clearSelection()
pagina.view.selectRow(0)
banco.w.choose_color(0, "#ffcc00")
check(pagina.model.data(pagina.model.index(0, 1), Qt.BackgroundRole).color().name() == "#ffcc00", "color de fondo de una fila")
banco.w.choose_color(None)
check(pagina.model.data(pagina.model.index(0, 1), Qt.BackgroundRole) is None, "quitar colores")
banco.w.show_bars = True
check(0 <= pagina.model.data(pagina.model.index(0, 0), Qt.UserRole + 1) <= 1, "barras en las numéricas")
banco.w.filter_edit.setText("virginica")
banco.w._apply_filter()
check(pagina.model.rowCount() == 50, "filtro rápido de filas")
banco.enviar()
check(len(banco.salida["data_1"]) == 150, "el filtro no cambia Data N")
check("Media" in pagina.model.headerData(0, Qt.Horizontal, Qt.ToolTipRole), "estadísticas de la columna en la cabecera")
banco.cerrar()

# --------------------------------------------------------- persistencia ----

banco = Banco().titulos("Flores").meter(iris.copy())
pagina = banco.pagina(0)
pagina.model.setData(pagina.model.index(0, 0), "9,9")
banco.w.find_edit.setText("Iris-")
banco.w.replace_edit.setText("")
banco.w.replace_all()
pagina.view.horizontalHeader().sectionClicked.emit(1)
cab = pagina.view.horizontalHeader()
cab.moveSection(cab.visualIndex(4), 0)
pagina.view.selectRow(0)
banco.w.choose_color(0, "#aaddff")
banco.w._save_all()
ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.enviar()
antes = banco.salida["data_1"]
banco.cerrar()
nuevo = Banco(ajustes).titulos("Flores").meter(iris.copy())
nuevo.enviar()
despues = nuevo.salida["data_1"]
check(despues.domain["iris"].values == ("setosa", "versicolor", "virginica") and
      np.array_equal(despues.X, antes.X) and [v.name for v in despues.domain.attributes] == [v.name for v in antes.domain.attributes],
      "los cambios se conservan con el workflow guardado (ediciones, reemplazos, orden)")
check(nuevo.w.pages[0].colors["rows"], "y los colores")
check("✎" in nuevo.w.tabs.tabText(0), "la pestaña restaurada aparece como modificada")
nuevo.cerrar()
otra = Banco(ajustes).titulos("Flores").meter(iris[:100].copy())
check("ha cambiado" in otra.avisos() and not otra.w.pages[0].state.log, "con una tabla incompatible, los cambios se descartan con aviso")
otra.cerrar()

banco = Banco().meter(iris.copy())
pagina = banco.pagina(0)
pagina.model.setData(pagina.model.index(0, 0), "1")
banco.w.reset_page()
check(not pagina.state.log and pagina.state.column("sepal length").values[0] == 5.1, "restablecer la pestaña")
banco.cerrar()

# ---------------------------------------------------------- robustez ----

disperso = Table.from_numpy(Domain([ContinuousVariable(f"x{i}") for i in range(4)], ContinuousVariable("y")),
                            sp.csr_matrix(np.eye(6, 4)), np.arange(6.0))
banco = Banco().meter(disperso)
banco.enviar()
check(banco.salida["data_1"] is not None and np.array_equal(banco.salida["data_1"].X, np.eye(6, 4)) and
      "dispersa" in banco.info(), "tabla dispersa: se muestra y se emite densa, con aviso")
banco.cerrar()
pesada = Table.from_numpy(Domain([ContinuousVariable("a")]), np.arange(5.0)[:, None], W=np.arange(5.0) + 1)
check(np.array_equal(engine.TableState(pesada).output().W, np.arange(5.0) + 1), "los pesos se conservan")


# ------------------------------------------------- correcciones de la 0.5.1 ----

# 1. Precisión: abrir y confirmar el editor no redondea; reemplazar no pierde decimales.
exacta = Table.from_numpy(Domain([ContinuousVariable("x")]), np.array([[3.14159265358], [0.000123456789], [123456789.123]]))
banco = Banco().meter(exacta)
pagina = banco.pagina(0)
editores = [pagina.model.data(pagina.model.index(i, 0), Qt.EditRole) for i in range(3)]
check(editores == ["3.14159265358", "0.000123456789", "123456789.123"],
      f"el editor recibe el valor con toda la precisión ({editores})")
for i in range(3):
    pagina.model.setData(pagina.model.index(i, 0), pagina.model.data(pagina.model.index(i, 0), Qt.EditRole))
    pagina.model.setData(pagina.model.index(i, 0), pagina.model.data(pagina.model.index(i, 0), Qt.DisplayRole))
check(list(pagina.state.column("x").values) == [3.14159265358, 0.000123456789, 123456789.123] and not pagina.state.log
      and pagina.state.column("x").edited is None, "confirmar sin cambios no redondea ni registra ninguna edición")
banco.w.find_regex = True
banco.w.find_edit.setText(r"(\d)")
banco.w.replace_edit.setText(r"\1")
check(banco.w.replace_all()[1] == 0 and not pagina.state.log, "un reemplazo que no cambia el número no toca la celda")
banco.w.find_regex = False
banco.w.find_edit.setText("123456789")
banco.w.replace_edit.setText("223456789")
check(banco.w.replace_all()[1] == 0 and pagina.state.column("x").values[2] == 123456789.123,
      "los dígitos que no se ven («1.23457e+08») no se buscan ni se reemplazan")
banco.w.find_edit.setText("1.23457e+08")
banco.w.replace_edit.setText("5")
banco.w.replace_all()
check(pagina.state.column("x").values[2] == 5.0 and pagina.state.column("x").values[0] == 3.14159265358,
      "reemplazar el texto visible entero; las celdas sin coincidencia conservan todos sus decimales")
banco.w.find_edit.setText("5.0")
check(engine.TableState(iris).find("5.0", cids=["sepal length"]) and
      len(engine.TableState(iris).find("5.0", cids=["sepal length"])) == 10,
      "se sigue buscando en el texto visible cuando es exacto («5.0»)")
banco.cerrar()

# 2 y 5. Restaurar: avisos acumulados, huella de contenido, nodo renombrado y cambios sin dueño.
banco = Banco().titulos("A", "B").meter(iris.copy(), titanic.copy())
for i in range(2):
    pagina = banco.pagina(i)
    pagina.model.setData(pagina.model.index(0, 0), "?")
banco.w._save_all()
ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.cerrar()
otra = Banco(ajustes).titulos("A", "B").meter(iris[:100].copy(), titanic.copy())
check("«A» ha cambiado" in otra.avisos() and otra.w.pages[1].state.log,
      "el aviso de los cambios descartados sobrevive a las demás entradas, que sí se restauran")
otra.cerrar()
cambiada = iris.copy()
with cambiada.unlocked():
    cambiada.X[7, 2] += 1
otra = Banco(ajustes).titulos("A", "B").meter(cambiada, titanic.copy())
check("misma forma, otros valores" in otra.avisos() and not otra.w.pages[0].state.log,
      "misma forma pero otros datos: los cambios se descartan con aviso")
otra.cerrar()
otra = Banco(ajustes).titulos("Flores", "B").meter(iris.copy(), titanic.copy())
check(otra.w.pages[0].state.log and "otro nombre" in otra.avisos() and "Flores" in otra.w.stored_changes,
      "nodo renombrado: los cambios se reconocen por el contenido y se guardan con el nombre nuevo")
otra.cerrar()
otra = Banco(ajustes).titulos("B").meter(titanic.copy())
check("no corresponden a ninguna entrada" in otra.avisos() and "«A»" in otra.avisos(),
      "cambios guardados sin entrada que los reciba: se avisa y se conservan")
otra.cerrar()

# 9. Tras restaurar se puede deshacer.
otra = Banco(ajustes).titulos("A", "B").meter(iris.copy(), titanic.copy())
otra.pagina(0)
check(otra.w.pages[0].state.can_undo(), "tras restaurar el workflow se pueden deshacer los cambios repetidos")
otra.w.undo()
check(otra.w.pages[0].state.column("sepal length").values[0] == 5.1, "y deshacer devuelve el valor original")
otra.cerrar()

# 3. Solo se envían las salidas que cambian.
banco = Banco().meter(iris, titanic)
banco.envios.clear()
pagina = banco.pagina(0)
pagina.view.selectRow(2)
app.processEvents()
banco.enviar()
check(set(banco.envios) == {"selected_data_1"}, f"seleccionar solo reenvía Selected Data de esa pestaña ({banco.envios})")
banco.envios.clear()
segunda = banco.pagina(1)
segunda.model.setData(segunda.model.index(0, 0), "second")
banco.enviar()
check(set(banco.envios) == {"data_2"}, f"editar solo reenvía las salidas de esa pestaña ({banco.envios})")
banco.cerrar()

# 4. Reemplazar una categoría por «» o «?» da un valor desconocido.
for vacio in ("", "?"):
    estado = engine.TableState(iris)
    estado.replace("Iris-setosa", vacio, cids=["iris"])
    columna = estado.column("iris")
    check(columna.categories == ["Iris-versicolor", "Iris-virginica"] and int(np.isnan(columna.values).sum()) == 50
          and vacio not in estado.output().domain.class_var.values,
          f"reemplazar una categoría por «{vacio}» la deja desconocida")
texto = Table.from_numpy(Domain([], metas=[StringVariable("t")]), np.empty((3, 0)),
                         metas=np.array([["a"], ["?"], ["b"]], dtype=object))
estado = engine.TableState(texto)
estado.convert("t", engine.CATEGORICAL)
check(estado.column("t").categories == ["a", "b"] and np.isnan(estado.column("t").values[1]),
      "al pasar texto a categórica, «?» es desconocido")

# 6. Tipo y rol de varias columnas a la vez: todo o nada.
banco = Banco().meter(zoo)
pagina = banco.pagina(0)
for nombre in ("hair", "name"):
    pagina.view.selectionModel().select(
        QItemSelection(pagina.model.index(0, pagina.state.index_of(nombre)),
                       pagina.model.index(pagina.model.rowCount() - 1, pagina.state.index_of(nombre))),
        QItemSelectionModel.Select)
app.processEvents()
banco.w.change_role(banco.w.role_combo.findData(engine.TARGET))
banco.enviar()
check("No se ha cambiado ningún rol" in banco.errores() and pagina.state.column("hair").role == engine.ATTRIBUTE
      and "hair" in [v.name for v in banco.salida["data_1"].domain.attributes],
      "rol en varias columnas con una imposible: no cambia ninguna y el estado coincide con la salida")
banco.cerrar()
muchas_cat = Table.from_numpy(Domain([], metas=[StringVariable("a"), StringVariable("b")]), np.empty((10001, 0)),
                              metas=np.array([[f"v{i}", "x"] for i in range(10001)], dtype=object))
estado = engine.TableState(muchas_cat)
try:
    estado.convert(["b", "a"], engine.CATEGORICAL)
    atomico = False
except engine.OperationError:
    atomico = estado.column("b").kind == engine.STRING and not estado.log
check(atomico, "cambio de tipo en varias columnas que falla en una: no cambia ninguna")

# 7. Fechas: la hora se ve en una columna de solo fechas y se respeta la zona de la columna.
estado = engine.TableState(Table.from_numpy(Domain([TimeVariable("d", have_date=1)]), np.array([[1.6e9]])))
estado.edit(0, "d", "2026-11-13 10:30")
check("10:30" in estado.column("d").text(0) and estado.output().domain["d"].have_time, "escribir una hora en una columna de fechas la muestra")
zona = TimeVariable("t")
valores = [zona.parse("2020-01-01 10:00:00+02:00"), zona.parse("2020-01-02 11:00:00+02:00")]
estado = engine.TableState(Table.from_numpy(Domain([zona]), np.array(valores)[:, None]))
estado.edit(1, "t", "2020-01-03 11:00")
check(estado.column("t").text(1).startswith("2020-01-03 11:00:00") and "+0200" in estado.column("t").text(1),
      f"una hora sin zona se toma en la zona de la columna ({estado.column('t').text(1)})")
reemplazo = engine.TableState(Table.from_numpy(Domain([zona]), np.array(valores)[:, None]))
reemplazo.replace("2020-01-01", "2021-01-01", cids=["t"])
check(reemplazo.column("t").text(0).startswith("2021-01-01 10:00:00"), "reemplazar en fechas con zona conserva la hora local")

# 8. Un prefijo que superaría el máximo de valores no deja la columna a medias.
grande_cat = DiscreteVariable("c", values=tuple(f"v{i}" for i in range(9999)))
estado = engine.TableState(Table.from_numpy(Domain([grande_cat]), np.arange(9999.0)[:, None]))
try:
    estado.affix(prefix="p_", cids=["c"])
except engine.OperationError:
    pass
check(len(estado.column("c").categories) == 9999 and estado.column("c").var is grande_cat and not estado.log,
      "prefijo rechazado por el máximo de valores: la columna queda intacta")

# Observación: con el filtro activo, los ámbitos actúan solo sobre las filas visibles.
banco = Banco().meter(iris.copy())
pagina = banco.pagina(0)
banco.w.filter_edit.setText("virginica")
banco.w._apply_filter()
check(banco.w.find_scope_combo.itemText(0) == "Filas visibles (filtro)", "con filtro, el ámbito dice «Filas visibles»")
banco.w.find_edit.setText("Iris-")
banco.w.replace_edit.setText("")
banco.w.find_scope = SCOPE_ALL
total, celdas, _ = banco.w.replace_all()
check(celdas == 50 and pagina.state.column("iris").categories == ["Iris-setosa", "Iris-versicolor", "virginica"],
      "con filtro, reemplazar solo cambia las filas visibles")
banco.w.filter_edit.setText("")
banco.w._apply_filter()
check(banco.w.find_scope_combo.itemText(0) == "Toda la tabla", "sin filtro, vuelve a «Toda la tabla»")
banco.cerrar()

# ------------------------------------------------- correcciones de la 0.5.2 ----

# 1. Buscar y reemplazar trabajan sobre el texto visible.
normales = np.random.RandomState(0).normal(size=20000)
estado = engine.TableState(Table.from_numpy(Domain([ContinuousVariable("n")]), normales[:, None]))
visibles = int(sum("99" in t for t in estado.column("n").texts()))
check(len(estado.find("99")) == visibles, f"buscar resalta solo lo que se ve ({len(estado.find('99'))} = {visibles})")
pi = Table.from_numpy(Domain([ContinuousVariable("x")]), np.array([[3.14159265358], [2.71828182846], [5.5]]))
estado = engine.TableState(pi)
estado.replace("3.14159", "2", cids=["x"])
check(estado.column("x").values[0] == 2.0 and estado.column("x").values[1] == 2.71828182846,
      "reemplazar el texto visible entero da el número del reemplazo; las demás celdas no se tocan")
estado = engine.TableState(pi)
estado.replace("3.14", "9.99", cids=["x"])
check(estado.column("x").values[0] == 9.99159 and estado.last_rounded == {"x": 1},
      "una coincidencia parcial se aplica al texto visible y se cuenta la celda redondeada reescrita")
banco = Banco().meter(pi)
banco.w.find_edit.setText("3.14")
banco.w.replace_edit.setText("9.99")
banco.w.replace_all()
check("se veían redondeadas" in banco.w.find_status.text(), "el widget avisa de las celdas redondeadas reescritas")
banco.cerrar()

# 2. Dos pestañas con la misma tabla: al renombrar una, cada una recupera lo suyo.
banco = Banco().titulos("A", "B").meter(iris.copy(), iris.copy())
banco.pagina(0).model.setData(banco.pagina(0).model.index(0, 0), "1")
banco.pagina(1).model.setData(banco.pagina(1).model.index(0, 0), "2")
banco.w._save_all()
ajustes = banco.w.settingsHandler.pack_data(banco.w)
banco.cerrar()
invertido = dict(ajustes)
invertido["stored_changes"] = dict(reversed(list(ajustes["stored_changes"].items())))
for nombre, guardado in (("en su orden", ajustes), ("con las claves al revés", invertido)):
    otra = Banco(guardado).titulos("A2", "B").meter(iris.copy(), iris.copy())
    otra.enviar()
    check((otra.salida["data_1"].X[0, 0], otra.salida["data_2"].X[0, 0]) == (1.0, 2.0),
          f"nodo renombrado {nombre}: A2 recibe lo de A y B lo suyo")
    otra.cerrar()
otra = Banco(invertido)
otra.titulos("A2")
otra.w.insert_data(0, iris.copy())
otra.w.handleNewSignals()
check(not otra.w.pages[0].state.log and "varios cambios guardados" in otra.avisos(),
      "con la entrada sola y dos candidatos iguales, no se adivina")
otra.titulos("A2", "B")
otra.w.insert_data(1, iris.copy())
otra.w.handleNewSignals()
otra.enviar()
check((otra.salida["data_1"].X[0, 0], otra.salida["data_2"].X[0, 0]) == (1.0, 2.0),
      "al llegar la otra entrada, cada una recibe lo suyo")
otra.cerrar()

# 3. Un solo envío por operación y nada de Data N al filtrar.
banco = Banco().meter(iris.copy())
pagina = banco.pagina(0)
pagina.view.selectRow(3)
app.processEvents()
banco.enviar()
for nombre, accion in (("ordenar", lambda: pagina.sort("sepal width", False)),
                       ("orden original", pagina.restore_order),
                       ("restablecer", banco.w.reset_page)):
    banco.envios.clear()
    accion()
    app.processEvents()
    banco.enviar()
    check(banco.envios.get("data_1", 0) <= 1 and banco.envios.get("selected_data_1", 0) <= 1 and banco.envios.get("data_1", 0) == 1,
          f"{nombre}: Data 1 se envía una sola vez ({banco.envios})")
banco.envios.clear()
banco.w.filter_edit.setText("setosa")
banco.w._apply_filter()
banco.w.filter_edit.setText("")
banco.w._apply_filter()
banco.enviar()
check("data_1" not in banco.envios, f"cambiar o quitar el filtro no reenvía Data 1 ({banco.envios})")
banco.cerrar()

# 4. Huella de la tabla completa.
cinco = Table.from_numpy(Domain([ContinuousVariable(f"n{i}") for i in range(5)]), np.random.RandomState(1).normal(size=(300_000, 5)))
inicio = time.time()
huella = engine.fingerprint(cinco)
coste_huella = time.time() - inicio
otra_tabla = cinco.copy()
with otra_tabla.unlocked():
    otra_tabla.X[[11, 777, 299_001], 3] += 1e-9
check(engine.fingerprint(otra_tabla) != huella and engine.fingerprint(Table("housing")) == engine.fingerprint(Table("housing")),
      f"la huella cubre toda la tabla: un cambio en cualquier fila se nota ({coste_huella * 1000:.0f} ms con 300.000 × 5)")
disperso_grande = Table.from_numpy(Domain([ContinuousVariable(f"w{i}") for i in range(60000)]),
                                   sp.random(20000, 60000, density=0.0005, format="csr", random_state=0))
inicio = time.time()
engine.fingerprint(disperso_grande)
check(time.time() - inicio < 2, "la huella de una tabla dispersa enorme no la densifica")

# 5. Filas sueltas compactas en el registro.
estado = engine.TableState(Table.from_numpy(Domain([], metas=[StringVariable("s")]), np.empty((300_000, 0)),
                                            metas=np.array([f"f{i}" for i in range(300_000)], dtype=object)[:, None]))
inicio = time.time()
estado.replace("f", "g", rows=np.arange(0, 300_000, 2))
por_filas = time.time() - inicio
import pickle
tamano = len(pickle.dumps(estado.log))
repetido = engine.TableState(estado.source)
repetido.replay(estado.log)
check(tamano < 10_000 and np.array_equal(repetido.column("s").values, estado.column("s").values) and por_filas < 1,
      f"150.000 filas alternas: {tamano} bytes en el registro, {por_filas:.2f} s, y se repite igual")

# 6. Formato idéntico al de Orange, fechas sin pérdida y filtro que no se pierde.
for variable in (ContinuousVariable("a"), ContinuousVariable("b", number_of_decimals=2),
                 ContinuousVariable("c", number_of_decimals=5)):
    valores = np.array([0.0, -0.0, 123456789012.34567, -98765432109.9, 1e16, 2.675, 1.005])
    estado = engine.TableState(Table.from_numpy(Domain([variable]), valores[:, None]))
    check(list(estado.column(variable.name).texts()) == [variable.str_val(v) for v in valores],
          f"formato igual que str_val con ceros con signo y números grandes ({variable.name})")
textos = np.array(["2020-09-13 12:26:40.123456", "1999-12-31 23:59:59.999999", "2024-02-29 00:00:00.5"], dtype=object)
mios = engine._parse_times(textos, np.ones(3, dtype=bool), None)[0]
check(all(mios[i] == TimeVariable("t").parse(textos[i]) for i in range(3)), "las fechas con pandas dan los mismos segundos que Orange")
banco = Banco().meter(iris.copy(), titanic.copy())
banco.pagina(0)
banco.w.filter_edit.setText("virginica")
banco.w.tabs.setCurrentIndex(1)
app.processEvents()
check(banco.w.pages[0].filter_text == "virginica" and banco.w.pages[0].model.rowCount() == 50 and
      banco.w.filter_edit.text() == "", "lo escrito en el filtro no se pierde al cambiar de pestaña enseguida")
banco.cerrar()

# ------------------------------------------------- correcciones de la 0.5.3 ----

# 1. Cambios guardados con otra versión de la huella: se restauran y no se borran.
banco = Banco().titulos("A").meter(iris.copy())
banco.pagina(0).model.setData(banco.pagina(0).model.index(0, 0), "1")
banco.w._save_all()
antiguo = banco.w.settingsHandler.pack_data(banco.w)
banco.cerrar()
entrada = antiguo["stored_changes"]["A"]
entrada["fingerprint"] = "ac6f51f8264ba76e"  # sha1 de 16 caracteres, como en la 0.5.1
entrada.pop("fingerprint_version", None)
otra = Banco(antiguo).titulos("A").meter(iris.copy())
otra.enviar()
check(otra.salida["data_1"].X[0, 0] == 1.0 and "A" in otra.w.stored_changes and not otra.avisos(),
      "un workflow guardado con la huella de la 0.5.1 se restaura y no se borra")
otra.w._save_all()
check(otra.w.stored_changes["A"].get("fingerprint_version") == engine.FINGERPRINT_VERSION,
      "al volver a guardar, la huella pasa al formato actual")
otra.cerrar()

# 2. Quitar el enlace de una pestaña nunca traslada sus cambios a otra.
banco = Banco().titulos("A", "B").meter(iris.copy(), iris.copy())
banco.pagina(0).model.setData(banco.pagina(0).model.index(0, 0), "1")
banco.w._save_all()
dos = banco.w.settingsHandler.pack_data(banco.w)
banco.titulos("B")
banco.w.remove_data(0)
banco.w.handleNewSignals()
banco.enviar()
check(banco.salida["data_1"].X[0, 0] == iris.X[0, 0] and "se han aplicado" not in banco.avisos(),
      "en la sesión: al quitar A, B no hereda su edición")
banco.cerrar()
otra = Banco(dos).titulos("A", "B").meter(iris.copy(), iris.copy())
otra.pagina(1).sort("petal width", True)
orden_b = otra.w.pages[1].state.order.copy()
otra.titulos("B")
otra.w.remove_data(0)
otra.w.handleNewSignals()
otra.enviar()
check(otra.salida["data_1"].X[0, 0] == iris.X[orden_b[0], 0] and np.array_equal(otra.w.pages[0].state.order, orden_b)
      and not otra.w.pages[0].state.log, "al reabrir: quitar A no pasa su edición a B, que conserva su propio orden")
otra.cerrar()
otra = Banco(dos).titulos("A", "B").meter(iris.copy(), iris.copy())
otra.titulos("B")
otra.w.remove_data(0)
otra.w.handleNewSignals()
otra.enviar()
check(otra.salida["data_1"].X[0, 0] == iris.X[0, 0], "al reabrir sin tocar B: quitar A tampoco le pasa la edición")
otra.cerrar()

# 3. La huella se calcula solo cuando hace falta, y es barata también con texto.
largos = Table.from_numpy(Domain([ContinuousVariable("a")], metas=[StringVariable(f"s{j}") for j in range(3)]),
                          np.zeros((300_000, 1)),
                          metas=np.array([[("x" * 500) + str(i * 3 + j) for j in range(3)] for i in range(300_000)],
                                         dtype=object))
inicio = time.time()
estado = engine.TableState(largos)
abrir_texto = time.time() - inicio
check(estado._fingerprint is None, "abrir una tabla no calcula la huella")
inicio = time.time()
_ = estado.fingerprint
huella_texto = time.time() - inicio
otra_largos = largos.copy()
with otra_largos.unlocked():
    otra_largos.metas[123_456, 1] = "cambiado"
check(huella_texto < 1 and engine.fingerprint(otra_largos) != estado.fingerprint,
      f"huella de 300.000 × 3 textos de 500 caracteres en {huella_texto:.2f} s (abrir: {abrir_texto:.2f} s), y detecta un cambio")

# --------------------------------------------------- tablas grandes ----

n = 300_000
r = np.random.RandomState(0)
grande = Table.from_numpy(
    Domain([ContinuousVariable(f"n{i}") for i in range(6)] + [DiscreteVariable("c", values=tuple("abcdefgh"))],
           ContinuousVariable("y"), [StringVariable("s")]),
    np.column_stack([r.normal(size=(n, 6)), r.randint(0, 8, n)]), r.normal(size=n),
    metas=np.array([f"fila {i}" for i in range(n)], dtype=object)[:, None])
inicio = time.time()
banco = Banco().meter(grande)
carga = time.time() - inicio
pagina = banco.pagina(0)
inicio = time.time()
pagina.sort("n0", False)
ordenar = time.time() - inicio
inicio = time.time()
for fila in range(0, 2000, 7):
    pagina.model.data(pagina.model.index(fila, 3))
lectura = time.time() - inicio
inicio = time.time()
banco.w.find_edit.setText("e")
banco.w.find_regex = False
pagina.select_column(pagina.state.index_of("c"))
banco.w.find_scope = SCOPE_COLUMNS
total, celdas, _ = banco.w.replace_all()
reemplazo = time.time() - inicio
inicio = time.time()
banco.enviar()
salida_t = time.time() - inicio
fechas_g = engine.TableState(Table.from_numpy(Domain([TimeVariable("f", have_date=1, have_time=1)]),
                                              (1.6e9 + r.randint(0, 10 ** 7, n)).astype(float)[:, None]))
inicio = time.time()
fechas_g.replace("2020", "2021", cids=["f"])
fecha_reemplazo = time.time() - inicio
inicio = time.time()
fechas_g.convert("f", engine.STRING)
fecha_texto = time.time() - inicio
inicio = time.time()
fechas_g.convert("f", engine.DATETIME)
texto_fecha = time.time() - inicio
inicio = time.time()
pagina.state.replace("fila", "row", rows=np.arange(0, n, 2), cids=["s"])
por_filas = time.time() - inicio
print(f"300.000 filas: carga {carga:.2f} s, ordenar {ordenar:.2f} s, 286 celdas {lectura * 1000:.0f} ms, "
      f"reemplazar en una columna {reemplazo:.2f} s, salidas {salida_t:.2f} s, reemplazar fechas {fecha_reemplazo:.2f} s, "
      f"fecha a texto {fecha_texto:.2f} s, texto a fecha {texto_fecha:.2f} s, 150.000 filas alternas {por_filas:.2f} s")
check(fecha_reemplazo < 2 and fecha_texto < 2 and texto_fecha < 2 and por_filas < 2,
      "fechas y ámbito por filas, vectorizados")
check(carga < 20 and ordenar < 5 and lectura < 1 and reemplazo < 10 and salida_t < 10 and celdas == int((grande.X[:, 6] == 4).sum()),
      "tablas grandes: carga, orden, lectura perezosa, reemplazo y salidas en tiempos razonables")
banco.cerrar()

check(not excepciones, "sin excepciones sin capturar:\n" + "\n".join(excepciones))
print("\nFALLOS:", fallos)
# Cierre ordenado: borrar los objetos de Qt antes que la aplicación evita fallos al salir.
app.sendPostedEvents(None, QEvent.DeferredDelete)
app.processEvents()
sys.stdout.flush()
sys.exit(1 if fallos else 0)
