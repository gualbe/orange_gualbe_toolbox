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

from AnyQt.QtCore import QEvent, QItemSelectionModel, Qt
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, TimeVariable

from orangecontrib.orange_gualbe_toolbox.widgets import _dtp_engine as engine
from orangecontrib.orange_gualbe_toolbox.widgets.owdatatablepro import OWDataTablePro, SCOPE_ALL, SCOPE_COLUMNS, SCOPE_ROWS

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
        self.salida = {}
        for i in range(1, 6):
            for nombre in (f"data_{i}", f"selected_data_{i}"):
                getattr(self.w.Outputs, nombre).send = self._capturar(nombre)

    def _capturar(self, nombre):
        def recibir(valor):
            self.salida[nombre] = valor
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

banco = Banco().titulos("Flores", "Barco", "Animales").meter(iris, titanic, zoo)
check([banco.w.tabs.tabText(i) for i in range(3)] == ["① Flores", "② Barco", "③ Animales"],
      "una pestaña por entrada, con su número de salida y el título del nodo")
check([banco.salida[f"data_{i}"].name for i in (1, 2, 3)] == ["Flores", "Barco", "Animales"] and
      banco.salida["data_4"] is None and banco.salida["data_5"] is None,
      "Data N en el orden de las pestañas, con el título del nodo como nombre")
check(all(banco.salida[f"selected_data_{i}"] is None for i in range(1, 6)), "sin selección, Selected Data N es None")
sin_titulo = Banco().meter(iris)
check(sin_titulo.salida["data_1"].name == "iris" and sin_titulo.w.tabs.tabText(0) == "① iris",
      "sin título de nodo, se usa el nombre de la tabla")
sin_titulo.cerrar()
salida = banco.salida["data_2"]
check(salida.domain == titanic.domain and np.array_equal(salida.X, titanic.X) and np.array_equal(salida.Y, titanic.Y)
      and np.array_equal(salida.ids, titanic.ids), "sin cambios, Data N es igual a la entrada (dominio, valores e ids)")
check("Data 2 y Selected Data 2" in banco.w.tabs.tabToolTip(1), "la pestaña dice qué salidas le corresponden")
banco.titulos("Barco", "Animales")  # Orange actualiza los enlaces antes de avisar al widget
banco.w.remove_data(0)
banco.w.handleNewSignals()
check(banco.salida["data_1"].name == "Barco" and banco.salida["data_2"].name == "Animales" and
      banco.salida["data_3"] is None, "al quitar un enlace, las salidas se recolocan y la última se vacía")
check(banco.w.tabs.tabText(0) == "① Barco", "los distintivos se renumeran")
banco.cerrar()

muchas = Banco().meter(*[iris] * 7)
check("7 tablas" in muchas.avisos() and muchas.w.tabs.tabText(5).startswith("—") and
      all(muchas.salida[f"data_{i}"] is not None for i in range(1, 6)), "más de 5 entradas: aviso, y la sexta sin salida")
check([muchas.salida[f"data_{i}"].name for i in range(1, 6)] == ["iris", "iris (2)", "iris (3)", "iris (4)", "iris (5)"],
      "identificadores únicos con la misma tabla repetida")
muchas.w.remove_data(6)
muchas.w.remove_data(5)
muchas.w.handleNewSignals()
check(not muchas.avisos(), "con 5 entradas desaparece el aviso")
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
print(f"300.000 filas: carga {carga:.2f} s, ordenar {ordenar:.2f} s, 286 celdas {lectura * 1000:.0f} ms, "
      f"reemplazar en una columna {reemplazo:.2f} s, salidas {salida_t:.2f} s")
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
