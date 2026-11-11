# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

"""Prueba sin interfaz gráfica del widget File Import.

Se ejecuta con el Python de Orange y sale con 1 si alguna comprobación falla.
Recorre la matriz de formatos × anomalías × opciones con el widget real, incluida
la descarga desde un servidor HTTP local.
"""
import bz2
import gzip
import http.server
import io
import lzma
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
import warnings
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import openpyxl

from AnyQt.QtCore import QModelIndex, Qt
from AnyQt.QtWidgets import QApplication
import Orange
from Orange.data import Table

from orangecontrib.orange_gualbe_toolbox.widgets import _fileimport as engine
from orangecontrib.orange_gualbe_toolbox.widgets import owfileimport as module
from orangecontrib.orange_gualbe_toolbox.widgets.owfileimport import ColumnsModel, OWFileImport

warnings.simplefilter("ignore", DeprecationWarning)
app = QApplication.instance() or QApplication([])
fallos = 0
excepciones = []


def _anotar(*error):
    texto = "".join(traceback.format_exception(*error))
    excepciones.append(texto)
    print("EXCEPCIÓN\n" + texto)


sys.excepthook = _anotar
DIR = tempfile.mkdtemp(prefix="test-file-import-")


def check(ok, msg):
    global fallos
    if not ok:
        print("FAIL " + msg)
    fallos += not ok
    return ok


def ruta(nombre):
    return os.path.join(DIR, nombre)


def escribir(nombre, contenido, encoding="utf-8"):
    path = ruta(nombre)
    data = contenido.encode(encoding) if isinstance(contenido, str) else contenido
    with open(path, "wb") as stream:
        stream.write(data)
    return path


class Banco:
    def __init__(self, ajustes=None):
        self.w = OWFileImport.__new__(OWFileImport, signal_manager=None, stored_settings=ajustes)
        self.w.__init__()
        self.salida = "sin enviar"
        self.w.Outputs.data.send = self._recibir
        self.w.recent_paths.clear()
        self.w.set_file_list()
        self.esperar()
        self.salida = "sin enviar"

    def _recibir(self, tabla):
        self.salida = tabla

    def esperar(self):
        app.processEvents()  # por si queda un QTimer de arranque pendiente
        inicio = time.time()
        while (self.w.task is not None or self.w._debounce.isActive()
               or self.w._after_preview.isActive()) and time.time() - inicio < 120:
            app.processEvents()
            time.sleep(0.001)
        app.processEvents()
        if not check(self.w.task is None, "el cálculo termina"):
            self.w.cancel()

    def abrir(self, path, **opciones):
        """Abre el archivo y aplica opciones como si se hubieran elegido."""
        self.salida = "sin enviar"
        config_clave = os.path.basename(path)
        self.w.source_configs.pop(config_clave, None)
        if opciones:
            base = engine.Options().to_dict()
            base.update(opciones)
            self.w.source_configs[config_clave] = {"options": base}
        self.w.open_path(path)
        self.esperar()
        self.w.grab()
        return self.salida

    def url(self, direccion):
        self.salida = "sin enviar"
        self.w.source_configs.pop(direccion, None)
        self.w.open_url(direccion)
        self.esperar()
        return self.salida

    def error(self):
        if self.w.Error.failed.is_shown():
            return str(self.w.Error.failed)
        if self.w.Error.unexpected.is_shown():
            return "INESPERADO " + str(self.w.Error.unexpected)
        return ""

    def avisos(self):
        return str(self.w.Warning.notes) if self.w.Warning.notes.is_shown() else ""

    def cerrar(self):
        self.w.onDeleteWidget()


banco = Banco()


def tipos(tabla):
    out = {}
    for var in tabla.domain.variables + tabla.domain.metas:
        out[var.name] = ("fecha" if var.is_time else "num" if var.is_continuous
                         else "cat" if var.is_discrete else "texto")
    return out


def correcto(tabla, etiqueta, filas=None, columnas=None):
    ok = check(isinstance(tabla, Table), f"{etiqueta}: hay tabla ({banco.error() or tabla!r:.200})")
    if ok and filas is not None:
        check(len(tabla) == filas, f"{etiqueta}: {len(tabla)} filas != {filas}")
    if ok and columnas is not None:
        total = len(tabla.domain.variables) + len(tabla.domain.metas)
        check(total == columnas, f"{etiqueta}: {total} columnas != {columnas}")
    check(not banco.w.Error.unexpected.is_shown(),
          f"{etiqueta}: error inesperado {banco.error()}")
    return ok


def falla(tabla, etiqueta, *textos):
    check(tabla is None, f"{etiqueta}: no debe haber salida ({tabla!r:.100})")
    mensaje = banco.error()
    check(mensaje and not mensaje.startswith("INESPERADO"),
          f"{etiqueta}: mensaje claro, no {mensaje!r}")
    for texto in textos:
        check(texto.lower() in mensaje.lower(), f"{etiqueta}: {texto!r} no está en {mensaje!r}")


# ------------------------------------------------------------------ formatos ----

CABECERA = ["num", "entero", "color", "nombre", "fecha"]
FILAS = [[f"{i}.5", str(i * 3), ["rojo", "verde", "azul"][i % 3],
          f"Persona número {i} con un texto largo", f"2024-01-{i + 1:02d}"]
         for i in range(30)]
FILAS[4][0] = ""
FILAS[7][2] = "?"


def delimitado(sep, decimal="."):
    filas = [sep.join(CABECERA)]
    for fila in FILAS:
        copia = list(fila)
        copia[0] = copia[0].replace(".", decimal)
        filas.append(sep.join(copia))
    return "\n".join(filas) + "\n"


TEXTO_COMA = delimitado(",")
ESPERADO = {"num": "num", "entero": "num", "color": "cat", "nombre": "texto", "fecha": "fecha"}


def comprobar_base(tabla, etiqueta):
    if not correcto(tabla, etiqueta, 30, 5):
        return
    check(tipos(tabla) == ESPERADO, f"{etiqueta}: tipos {tipos(tabla)}")
    columna = tabla.get_column("num") if hasattr(tabla, "get_column") else None
    check(columna is not None and np.isnan(columna[4]) and columna[5] == 5.5,
          f"{etiqueta}: valores numéricos y perdidos")
    fecha = tabla.get_column("fecha")
    check(fecha[0] == 1704067200.0, f"{etiqueta}: fecha {fecha[0]}")
    color = tabla.domain["color"]
    check(np.isnan(tabla.get_column("color")[7]), f"{etiqueta}: «?» es perdido")
    check(set(color.values) == {"rojo", "verde", "azul"}, f"{etiqueta}: categorías")
    check(tabla.metas[3, 0] == "Persona número 3 con un texto largo", f"{etiqueta}: texto")


for nombre, contenido, delim in (
        ("base.csv", TEXTO_COMA, ","),
        ("punto_coma.csv", delimitado(";"), ";"),
        ("tabulado.tsv", delimitado("\t"), "\t"),
        ("barra.txt", delimitado("|"), "|")):
    t = banco.abrir(escribir(nombre, contenido))
    comprobar_base(t, nombre)
    check(banco.w._loaded is not None and banco.w._loaded.delimiter == delim,
          f"{nombre}: separador detectado")

t = banco.abrir(escribir("decimal_coma.csv", delimitado(";", ",")))
comprobar_base(t, "decimal con coma")
check(banco.w._loaded.decimal == ",", "decimal con coma: detectado")

# Codificaciones × finales de línea × comillas
for encoding, bom in (("utf-8", b""), ("utf-8", b"\xef\xbb\xbf"), ("utf-16", b""),
                      ("utf-16-le", b""), ("cp1252", b""), ("latin-1", b"")):
    for salto in ("\n", "\r\n"):
        for comillas in (False, True):
            texto = TEXTO_COMA
            if comillas:
                texto = "\n".join(",".join(f'"{c}"' for c in linea.split(","))
                                  for linea in texto.splitlines()) + "\n"
            texto = texto.replace("\n", salto)
            etiqueta = f"{encoding} bom={bool(bom)} {salto!r} comillas={comillas}"
            path = escribir("cod.csv", bom + texto.encode(encoding))
            comprobar_base(banco.abrir(path), etiqueta)

for compresor, extension in ((gzip.compress, ".gz"), (bz2.compress, ".bz2"), (lzma.compress, ".xz")):
    path = escribir("comprimido.csv" + extension, compresor(TEXTO_COMA.encode()))
    comprobar_base(banco.abrir(path), "compresión " + extension)
path = ruta("comprimido.csv.zip")
with zipfile.ZipFile(path, "w") as z:
    z.writestr("datos.csv", TEXTO_COMA)
comprobar_base(banco.abrir(path), "compresión zip")
path = ruta("dos.zip")
with zipfile.ZipFile(path, "w") as z:
    z.writestr("a.csv", TEXTO_COMA)
    z.writestr("b.csv", TEXTO_COMA)
falla(banco.abrir(path), "zip con dos archivos", "exactamente un archivo")

# TAB de Orange: tres filas, roles y tipos declarados
tab = ("edad\tgrupo\tid\tnota\tignorada\tobjetivo\n"
       "c\tA B\ts\tcontinuous\td\td\n"
       "\t\tmeta\t\ti\tclass\n"
       "30\tA\t001\t7.5\tx\tsi\n"
       "40\tB\t002\t?\ty\tno\n"
       "50\tA\t003\t9\tz\tsi\n")
t = banco.abrir(escribir("orange.tab", tab))
if correcto(t, "TAB tres filas", 3, 5):
    check([v.name for v in t.domain.attributes] == ["edad", "grupo", "nota"], "TAB: atributos")
    check([v.name for v in t.domain.class_vars] == ["objetivo"], "TAB: clase")
    check([v.name for v in t.domain.metas] == ["id"], "TAB: meta")
    check(t.domain["grupo"].values == ("A", "B"), "TAB: valores declarados")
    check(t.metas[0, 0] == "001", "TAB: el texto no se convierte en número")
t = banco.abrir(escribir("prefijos.tab", "C#x\tD#y\tcD#z\tm#n\tS#t\n1\ta\tb\tq\tw\n2\tc\td\tr\tu\n"))
if correcto(t, "TAB prefijos", 2):
    check([v.name for v in t.domain.class_vars] == ["z"], "TAB prefijos: clase")
    check(t.domain["y"].is_discrete and t.domain["x"].is_continuous, "TAB prefijos: tipos")

iris_tab = os.path.join(os.path.dirname(Orange.__file__), "datasets", "iris.tab")
t = banco.abrir(iris_tab)
referencia = Table("iris")
if correcto(t, "iris.tab", 150, 5):
    check(np.allclose(t.X, referencia.X), "iris.tab: mismos atributos que File")
    check(t.domain.class_var.values == referencia.domain.class_var.values, "iris.tab: clases")
    check(np.array_equal(t.Y, referencia.Y), "iris.tab: clase igual que File")

# Excel
def libro(nombre, hojas):
    book = openpyxl.Workbook()
    book.remove(book.active)
    for titulo, filas in hojas.items():
        hoja = book.create_sheet(titulo)
        for fila in filas:
            hoja.append(fila)
    path = ruta(nombre)
    book.save(path)
    return path


import datetime as dt  # noqa: E402

FILAS_XL = [[CABECERA[0], CABECERA[1], CABECERA[2], CABECERA[3], CABECERA[4]]]
for i, fila in enumerate(FILAS):
    FILAS_XL.append([None if i == 4 else i + 0.5, i * 3, fila[2] if i != 7 else "?",
                     fila[3], dt.datetime(2024, 1, i + 1)])
comprobar_base(banco.abrir(libro("base.xlsx", {"Datos": FILAS_XL})), "xlsx")
check(banco.w._loaded.format == "xlsx", "xlsx: formato detectado")

path = libro("hojas.xlsx", {"Vacía": [], "Datos": FILAS_XL, "Otra": [["a", "b"], [1, 2], [3, 4]]})
comprobar_base(banco.abrir(path), "xlsx con primera hoja vacía")
check("está vacía" in banco.avisos(), "xlsx: avisa de la hoja vacía")
check(banco.w.sheet_combo.count() == 3 and banco.w.sheet_combo.isVisibleTo(banco.w), "xlsx: lista las hojas")
t = banco.abrir(path, sheet="Otra")
correcto(t, "xlsx: hoja elegida", 2, 2)
falla(banco.abrir(path, sheet="Vacía"), "xlsx: hoja vacía elegida", "vacía", "Datos")
falla(banco.abrir(libro("vacio.xlsx", {"A": [], "B": []})), "xlsx sin datos", "vacías")
falla(banco.abrir(libro("solo_cabecera.xlsx", {"A": [["a", "b"]]})), "xlsx solo cabecera", "ninguna fila")

book = openpyxl.Workbook()
hoja = book.active
hoja.append(["grupo", "valor"])
hoja.append(["A", 1])
hoja.append([None, 2])
hoja.append(["B", 3])
hoja.merge_cells("A2:A3")
book.save(ruta("combinadas.xlsx"))
t = banco.abrir(ruta("combinadas.xlsx"))
if correcto(t, "celdas combinadas", 3, 2):
    grupo = t.domain["grupo"]
    check([grupo.values[int(v)] for v in t.get_column("grupo")] == ["A", "A", "B"],
          "celdas combinadas: el valor se propaga")
    check("combinada" in banco.avisos(), "celdas combinadas: avisa")

t = banco.abrir(libro("errores.xlsx", {"A": [["x", "y"], [1, "#N/A"], [2, "#DIV/0!"], [3, "ok"], [4, "ok"]]}))
if correcto(t, "errores de Excel", 4, 2):
    check(np.isnan(t.get_column("y")[:2]).all(), "errores de Excel: son perdidos")

mixtas = [["id", "mixta"]] + [[i, i if i < 9 else "texto"] for i in range(10)]
correcto(banco.abrir(libro("mixta.xlsx", {"A": mixtas})), "xlsx con columna mezclada", 10, 2)

xls_dir = os.path.join(os.path.dirname(Orange.__file__), "tests", "xlsx_files")
for nombre in ("header_0.xls", "header_1_flags.xls", "header_3.xls", "missing.xls"):
    origen = os.path.join(xls_dir, nombre)
    if check(os.path.exists(origen), f"existe {nombre}"):
        t = banco.abrir(origen)
        correcto(t, nombre)
        check(banco.w._loaded.format == "xls", f"{nombre}: formato")

cifrado = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 500 + "EncryptedPackage".encode("utf-16-le") + b"\0" * 500
falla(banco.abrir(escribir("cifrado.xlsx", cifrado)), "Excel cifrado", "contraseña")
falla(banco.abrir(escribir("dañado.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1basura" + b"\xff" * 300)),
      "XLS dañado")
falla(banco.abrir(escribir("zip_falso.xlsx", b"PK\x03\x04basura")), "XLSX dañado")

# ARFF
ARFF = """% comentario inicial
@relation prueba
@attribute edad numeric
@attribute 'nombre completo' string
@attribute cuando date "yyyy-MM-dd HH:mm"
@attribute color {rojo,'azul claro',verde}
@attribute sano {si,no}

@data
30,'Ana P',"2024-01-02 10:30",rojo,si
% comentario en medio
?,'Luis \\'el\\' G',"2024-02-03 00:00",'azul claro',no
45,?,?,?,si
"""
t = banco.abrir(escribir("denso.arff", ARFF.replace('"2024-01-02 10:30"', "'2024-01-02 10:30'")
                         .replace('"2024-02-03 00:00"', "'2024-02-03 00:00'")))
if correcto(t, "ARFF denso", 3, 5):
    check(tipos(t) == {"edad": "num", "nombre completo": "texto", "cuando": "fecha",
                       "color": "cat", "sano": "cat"}, f"ARFF: tipos {tipos(t)}")
    check(t.domain["color"].values == ("rojo", "azul claro", "verde"), "ARFF: nominales declarados")
    check([v.name for v in t.domain.class_vars] == ["sano"], "ARFF: última nominal es la clase")
    check(np.isnan(t.get_column("edad")[1]), "ARFF: ? es perdido")
    check(t.metas[1, 0] == "Luis 'el' G", f"ARFF: comillas escapadas {t.metas[1, 0]!r}")
    check(t.get_column("cuando")[0] == 1704191400.0, "ARFF: fecha")

SPARSE = """@relation s
@attribute a numeric
@attribute b numeric
@attribute c {x,y}
@data
{0 1.5,2 y}
{1 2}
{0 ?, 1 3}
{}
"""
t = banco.abrir(escribir("disperso.arff", SPARSE))
if correcto(t, "ARFF disperso", 4, 3):
    check(np.isnan(t.get_column("a")[2]) and t.get_column("a")[1] == 0, "ARFF disperso: numéricos")
    check([t.domain["c"].values[int(v)] for v in t.get_column("c")] == ["y", "x", "x", "x"],
          "ARFF disperso: el omitido es el primer valor")

for etiqueta, texto, esperado in (
        ("sin @data", "@relation a\n@attribute x numeric\n", "@data"),
        ("sin atributos", "@relation a\n@data\n1\n", "@attribute"),
        ("tipo raro", "@relation a\n@attribute x cosa\n@data\n1\n", "tipo desconocido"),
        ("nominal sin cerrar", "@relation a\n@attribute x {a,b\n@data\na\n", "llave"),
        ("línea rara", "@relation a\nhola\n@attribute x numeric\n@data\n1\n", "línea 2"),
        ("relacional", "@relation a\n@attribute x relational\n@data\n1\n", "relacional"),
        ("sin filas", "@relation a\n@attribute x numeric\n@data\n", "filas"),
        ("disperso mal", "@relation a\n@attribute x numeric\n@data\n{0 1\n", "disperso")):
    falla(banco.abrir(escribir("mal.arff", texto)), f"ARFF {etiqueta}", esperado)

t = banco.abrir(escribir("campos.arff", "@relation a\n@attribute x numeric\n@attribute y numeric\n"
                         "@data\n1,2\n3\n4,5,6\n7,8\n"))
correcto(t, "ARFF con filas irregulares", 4, 2)
check("más de" in banco.avisos() and "menos de" in banco.avisos(), "ARFF irregular: informa")
t = banco.abrir(escribir("no_declarado.arff", "@relation a\n@attribute x {a,b}\n@attribute y numeric\n"
                         "@data\na,1\nc,2\nb,xx\n"))
if correcto(t, "ARFF con valor no declarado", 3, 2):
    check(t.domain["x"].values == ("a", "b", "c"), "ARFF: valor no declarado se añade")
    check("declarad" in banco.avisos() and "tipo de su columna" in banco.avisos(), "ARFF: avisa")

# ------------------------------------------------------------------ anomalías ----

falla(banco.abrir(ruta("no_existe.csv")), "archivo inexistente", "No existe")
falla(banco.abrir(escribir("vacio.csv", "")), "archivo vacío", "vacío")
falla(banco.abrir(escribir("blancos.csv", "\n\n   \n")), "archivo en blanco", "vacío")
falla(banco.abrir(escribir("solo_cab.csv", "a,b,c\n")), "solo cabecera", "ninguna fila")
os.makedirs(ruta("carpeta"), exist_ok=True)
falla(banco.abrir(ruta("carpeta")), "carpeta", "carpeta")
privado = escribir("privado.csv", TEXTO_COMA)
os.chmod(privado, 0)
if os.access(privado, os.R_OK):
    print("(se omite la prueba de permisos: se ejecuta como superusuario)")
else:
    falla(banco.abrir(privado), "sin permisos", "permiso")
os.chmod(privado, 0o644)

falla(banco.abrir(escribir("binario.csv", b"\x00\x01\x02" * 2000 + b"\xff")), "binario", "binario")
falla(banco.abrir(escribir("web.csv", "<!DOCTYPE html><html><body>hola</body></html>")),
      "HTML con extensión csv", "HTML")

# Codificación incorrecta elegida a mano; UTF-8 solo en el principio
latin = TEXTO_COMA.replace("Persona número", "Señor Ñandú").encode("cp1252")
falla(banco.abrir(escribir("latin.csv", latin), encoding="utf-8"),
      "codificación incorrecta elegida", "codificación")
t = banco.abrir(ruta("latin.csv"))
if correcto(t, "no UTF-8 detectado", 30, 5):
    check(t.metas[0, 0].startswith("Señor Ñandú"), "no UTF-8: texto correcto")
falla(banco.abrir(ruta("latin.csv"), encoding="no-existe"), "codificación inexistente", "no existe")
tarde = (b"a,b\n" + b"1,hola\n" * 400000 + b"2,caf\xe9\n3,x\n")
t = banco.abrir(escribir("tarde.csv", tarde))
if correcto(t, "byte no UTF-8 fuera del bloque inicial", 400002, 2):
    check("Windows-1252" in banco.avisos(), "byte tardío: avisa de la codificación")

ambiguo = "a;b,c\n1;2,3\n4;5,6\n7;8,9\n"
t = banco.abrir(escribir("ambiguo.csv", ambiguo))
correcto(t, "separador ambiguo")
check("ambiguo" in banco.avisos(), "separador ambiguo: avisa")
t = banco.abrir(ruta("ambiguo.csv"), delimiter=";")
correcto(t, "separador elegido a mano", 3, 2)

irregular = "a,b,c\n1,2,3\n4,5\n6,7,8,9\n10,11,12\n13\n"
t = banco.abrir(escribir("irregular.csv", irregular))
if correcto(t, "campos de más y de menos", 5, 3):
    check("1 fila tiene más" in banco.avisos(), f"irregular: cuenta las largas: {banco.avisos()}")
    check("2 filas tienen menos" in banco.avisos(), f"irregular: cuenta las cortas: {banco.avisos()}")
    check(np.isnan(t.get_column("c")[1]) and t.get_column("c")[2] == 8, "irregular: valores")
t = banco.abrir(escribir("irregular_sin_salto.csv", "a,b,c\n1,2,3\n4,5"))
check("1 fila tiene menos" in banco.avisos(), "irregular: última línea sin salto")
t = banco.abrir(escribir("comillas_multilinea.csv", 'a,b\n"uno\ndos, tres",2\n"x",3\n'))
if correcto(t, "campo con salto de línea entre comillas", 2, 2):
    check(not banco.avisos(), f"las comillas no se cuentan como filas irregulares: {banco.avisos()}")
falla(banco.abrir(escribir("sin_cerrar.csv", 'a,b\n1,2\n"3,4\n5,6\n')),
      "comillas sin cerrar", "comillas sin cerrar")
t = banco.abrir(ruta("sin_cerrar.csv"), quotechar="")
correcto(t, "comillas desactivadas", 3, 2)

t = banco.abrir(escribir("nombres.csv", "a,a,,  ,a\n1,2,3,4,5\n6,7,8,9,10\n"))
if correcto(t, "nombres repetidos y vacíos", 2, 5):
    nombres = [v.name for v in t.domain.attributes]
    check(len(set(nombres)) == 5 and "a (2)" in nombres and "Columna 3" in nombres,
          f"nombres únicos: {nombres}")
    check("repetido" in banco.avisos() and "nombre" in banco.avisos(), "nombres: avisa")

t = banco.abrir(escribir("cols_vacias.csv", "a,b,c\n1,,x\n2,,y\n3,,z\n"))
if correcto(t, "columna entera vacía", 3, 2):
    check("vacía" in banco.avisos(), "columna vacía: avisa")
    check("b" not in tipos(t), "columna vacía: se omite")
t = banco.abrir(escribir("coma_final.csv", "a,b,\n1,2,\n3,4,\n"))
correcto(t, "coma al final de cada línea", 2, 2)

filas_mix = "\n".join(f"{i},{i if i != 50 else 'n/d'},{i if i % 2 else 'txt' + str(i % 4)}" for i in range(100))
t = banco.abrir(escribir("mezcla.csv", "id,casi,mitad\n" + filas_mix + "\n"))
if correcto(t, "tipos mezclados", 100, 3):
    check(tipos(t)["casi"] == "num", "mezcla: casi todo numérico sigue siendo numérico")
    check(np.isnan(t.get_column("casi")[50]), "mezcla: el texto queda perdido")
    check("tipo de su columna" in banco.avisos(), "mezcla: informa")
    check(tipos(t)["mitad"] in ("cat", "texto"), "mezcla: mitad y mitad no es numérico")

fechas = ("d,e,f,g\n"
          "31/12/2023,2023-12-31T10:00:00Z,1500-01-01,no es fecha\n"
          "01/02/2024,2024-01-01T10:00:00+02:00,2024-05-01,12\n"
          "15/03/2024,2024-01-02T10:00:00Z,2024-05-02,x\n")
fechas += "".join(f"16/03/2024,2024-01-02T10:00:00Z,2024-05-{d:02d},y\n" for d in range(3, 28))
t = banco.abrir(escribir("fechas.csv", fechas))
if correcto(t, "fechas raras", 28, 4):
    check(tipos(t)["d"] == "fecha" and t.get_column("d")[0] == 1703980800.0, "fechas: día primero")
    check(t.get_column("e")[1] == 1704096000.0, "fechas: zona horaria")
    check(tipos(t)["f"] == "fecha" and np.isnan(t.get_column("f")[0]), "fechas: fuera de rango")
    check("tipo de su columna" in banco.avisos(), "fechas: informa de las no válidas")
t = banco.abrir(escribir("mdy.csv", "d\n12/31/2023\n01/02/2024\n03/15/2024\n"))
if correcto(t, "fechas mes primero", 3, 1):
    check(t.get_column("d")[0] == 1703980800.0, "fechas: mes primero deducido")
t = banco.abrir(escribir("orden.csv", "d\n01/02/2024\n03/04/2024\n"), date_order="mdy")
if correcto(t, "orden de fecha elegido", 2, 1):
    check(t.get_column("d")[0] == 1704153600.0, "fechas: orden mes/día forzado")

enormes = ("id,flo,infi,normal\n"
           "12345678901234567890123,1e308,1e999,1\n"
           "99999999999999999999999,2.5,-1e999,2\n"
           "9007199254740993,3.5,5,3\n")
t = banco.abrir(escribir("enormes.csv", enormes))
if correcto(t, "números enormes", 3, 4):
    check(np.isnan(t.get_column("infi")[:2]).all(), "enormes: infinitos a perdidos")
    check("infinito" in banco.avisos(), "enormes: avisa de los infinitos")
    check("enormes" in banco.avisos(), "enormes: avisa de la pérdida de precisión")
    check(t.get_column("flo")[0] == 1e308, "enormes: flotante grande")

# Extensión que no coincide con el contenido
t = banco.abrir(escribir("texto.xlsx", TEXTO_COMA))
comprobar_base(t, "texto con extensión xlsx")
check("extensión" in banco.avisos(), "extensión falsa: avisa")
shutil.copy(ruta("base.xlsx"), ruta("excel.csv"))
comprobar_base(banco.abrir(ruta("excel.csv")), "xlsx con extensión csv")
check("extensión" in banco.avisos(), "xlsx como csv: avisa")
t = banco.abrir(escribir("arff.txt", SPARSE))
correcto(t, "ARFF con extensión txt", 4, 3)
falla(banco.abrir(ruta("base.csv"), format="xlsx"), "formato incorrecto elegido", "Excel")

# ------------------------------------------------------------------- opciones ----

texto = "# comentario\nbasura\n\nx;y\n1,5;a\n2,5;b\nN.D.;c\n3,5;\n"
t = banco.abrir(escribir("opciones.csv", texto), skip_rows=2, delimiter=";", decimal=",",
                na_values="N.D.", comment="#")
if correcto(t, "saltar filas, separador, decimal, perdidos y comentarios", 4, 2):
    check(np.isnan(t.get_column("x")[2]) and t.get_column("x")[1] == 2.5, "opciones: números")

t = banco.abrir(escribir("sin_cab.csv", "1,2,a\n3,4,b\n"), header=engine.HEADER_NONE)
if correcto(t, "sin cabecera", 2, 3):
    check([v.name for v in t.domain.attributes][:2] == ["Columna 1", "Columna 2"], "sin cabecera: nombres")
t = banco.abrir(ruta("sin_cab.csv"))
correcto(t, "sin cabecera detectado", 2, 3)
t = banco.abrir(escribir("cab3.csv", "titulo\nsubtitulo\nh1,h2\n1,2\n3,4\n"), header=3)
if correcto(t, "cabecera en otra fila", 2, 2):
    check("h1" in tipos(t), "cabecera en otra fila: nombres")
falla(banco.abrir(ruta("sin_cab.csv"), header=50), "cabecera inexistente", "no existe")

t = banco.abrir(ruta("base.csv"), max_rows=7)
correcto(t, "máximo de filas", 7, 5)
check("primeras" in banco.avisos(), "máximo de filas: avisa")
muchas_saltadas = "".join(f"linea de prologo {i}\n" for i in range(20000)) + "x,y\n1,2\n3,4\n"
t = banco.abrir(escribir("prologo.csv", muchas_saltadas), skip_rows=20000)
if correcto(t, "saltar muchas filas", 2, 2):
    check("x" in tipos(t), "saltar muchas filas: encuentra la cabecera")
t = banco.abrir(libro("max.xlsx", {"A": FILAS_XL}), max_rows=7)
correcto(t, "máximo de filas en Excel", 7, 5)
t = banco.abrir(ruta("denso.arff"), max_rows=2)
correcto(t, "máximo de filas en ARFF", 2, 5)

t = banco.abrir(escribir("miles.csv", 'x;y\n1.234,5;a\n2.000,0;b\n1.000.000,25;c\n'))
if correcto(t, "miles europeos detectados", 3, 2):
    check(t.get_column("x")[2] == 1000000.25, "miles europeos: valores")
t = banco.abrir(escribir("miles2.csv", 'x|y\n1,234.5|a\n2,000.0|b\n'), thousands=",", decimal=".")
if correcto(t, "miles elegidos", 2, 2):
    check(t.get_column("x")[0] == 1234.5, "miles elegidos: valores")
falla(banco.abrir(ruta("miles.csv"), decimal=".", thousands="."), "decimal igual a miles", "iguales")
falla(banco.abrir(ruta("base.csv"), delimiter=",", decimal=","), "separador igual a decimal", "separador")
falla(banco.abrir(ruta("base.csv"), delimiter="ab"), "separador largo", "único carácter")
t = banco.abrir(ruta("base.csv"), format="tsv")
correcto(t, "formato corregido a TSV: una sola columna")
t = banco.abrir(escribir("clase.csv", "a,clase\n1,x\n2,y\n3,x\n"))
if correcto(t, "clase por nombre", 3, 2):
    check([v.name for v in t.domain.class_vars] == ["clase"], "clase por nombre")
t = banco.abrir(escribir("binaria.csv", "b,n\n0,1.5\n1,2.5\n1,3.5\n0,4.5\n"))
if correcto(t, "enteros binarios", 4, 2):
    check(tipos(t) == {"b": "cat", "n": "num"}, f"enteros binarios como categóricos: {tipos(t)}")

# Vista previa: solo el principio del archivo
grande = escribir("grande.csv", b"a,b,c\n" + b"1.5,hola,2\n" * 2000000)
tamano = os.path.getsize(grande)
leidos = {"n": 0}
original_read = engine._Feed.read


def contar(self, count=-1):
    data = original_read(self, count)
    leidos["n"] += len(data)
    return data


engine._Feed.read = contar
previa = engine.load(grande, engine.Options(), "grande.csv", limit=engine.INFERENCE_ROWS, head_only=True)
engine._Feed.read = original_read
check(leidos["n"] <= engine.HEAD_BYTES, f"la vista previa lee {leidos['n']} de {tamano} bytes")
check(previa.n_rows == engine.INFERENCE_ROWS and not previa.complete, "la vista previa limita las filas")

# Cancelación y segundo plano: la interfaz sigue viva y no llega resultado
banco.salida = "sin enviar"
banco.w.source_configs.pop("grande.csv", None)
banco.w.open_path(grande)
check(banco.w.task is not None and banco.salida == "sin enviar", "la lectura va en segundo plano")
inicio = time.time()
while banco.w._preview is None and time.time() - inicio < 30:
    app.processEvents()
    time.sleep(0.001)
check(banco.w.task is not None, "tras la vista previa empieza la carga completa")
banco.w.cancel_import()
check(banco.w.task is None and not banco.w.cancel_button.isVisibleTo(banco.w), "cancelar detiene la tarea")
time.sleep(0.3)
app.processEvents()
check(banco.salida == "sin enviar", "tras cancelar no llega resultado")
check(banco.w.Information.details.is_shown(), "cancelar lo indica")
check(engine._Compressed is not None, "motor disponible")

cancelado = False
try:
    engine.load(grande, engine.Options(), "grande.csv",
                progress=engine.Progress(is_cancelled=lambda: True))
except engine.ImportCancelled:
    cancelado = True
check(cancelado, "el motor se cancela durante la lectura")

# ------------------------------------------------------------------- columnas ----

t = banco.abrir(ruta("base.csv"))
modelo = banco.w.columns_model
cargas = {"n": 0}
original_load = module.engine.load


def contar_cargas(*args, **kwargs):
    if not kwargs.get("head_only"):
        cargas["n"] += 1
    return original_load(*args, **kwargs)


module.engine.load = contar_cargas
fila = [c.name for c in modelo.columns].index("entero")
modelo.setData(modelo.index(fila, ColumnsModel.TYPE), engine.CATEGORICAL)
banco.esperar()
check(tipos(banco.salida)["entero"] == "cat", "cambiar tipo: numérico a categórico")
check(banco.salida.domain["entero"].values[:2] == ("0", "3"), "cambiar tipo: etiquetas enteras")
check(cargas["n"] == 0, "cambiar el tipo no vuelve a leer el archivo")
modelo.setData(modelo.index(fila, ColumnsModel.ROLE), engine.CLASS)
banco.esperar()
check([v.name for v in banco.salida.domain.class_vars] == ["entero"], "cambiar rol a clase")
modelo.setData(modelo.index(fila, ColumnsModel.NAME), "renombrada")
banco.esperar()
check("renombrada" in tipos(banco.salida), "renombrar")
fila_c = [c.name for c in modelo.columns].index("color")
modelo.setData(modelo.index(fila_c, ColumnsModel.ROLE), engine.SKIP)
banco.esperar()
check("color" not in tipos(banco.salida), "omitir columna")
check(banco.w.preview_model.rowCount() == 30, "la vista previa sigue ahí")
fila_n = [c.name for c in modelo.columns].index("num")
modelo.setData(modelo.index(fila_n, ColumnsModel.TYPE), engine.TEXT)
banco.esperar()
check(tipos(banco.salida)["num"] == "texto" and modelo.choices[fila_n].role == engine.META,
      "el texto siempre es meta")
modelo.setData(modelo.index(fila_n, ColumnsModel.ROLE), engine.ATTRIBUTE)
check(modelo.choices[fila_n].role == engine.META, "el texto no puede ser atributo")
fila_f = [c.name for c in modelo.columns].index("fecha")
modelo.setData(modelo.index(fila_f, ColumnsModel.TYPE), engine.NUMERIC)
banco.esperar()
check(tipos(banco.salida)["fecha"] == "num" and "tipo de su columna" in banco.avisos(),
      "tipo imposible: valores perdidos y aviso")
check(cargas["n"] == 0, "editar columnas nunca vuelve a leer el archivo")
module.engine.load = original_load
banco.w.columns_view.selectAll()
banco.w.columns_model.apply_to(banco.w._selected_rows(), "role", engine.SKIP)
banco.esperar()
check(banco.salida is None and "omitidas" in banco.error(), "todas omitidas: mensaje claro")
banco.w.reset_columns()
banco.esperar()
comprobar_base(banco.salida, "restaurar lo detectado")

# Aplicación manual
banco.w.auto_apply = False
banco.salida = "sin enviar"
modelo.setData(modelo.index(0, ColumnsModel.ROLE), engine.META)
banco.esperar()
check(banco.salida == "sin enviar", "sin aplicación automática no envía")
banco.w.commit.now()
banco.esperar()
check(isinstance(banco.salida, Table), "al aplicar envía")
banco.w.auto_apply = True
banco.w.reset_columns()
banco.esperar()

# ---------------------------------------------------- estado con el workflow ----

t = banco.abrir(escribir("estado.csv", "a;b;c\n1;x;9\n2;y;8\n3;z;7\n"), delimiter=";")
modelo = banco.w.columns_model
modelo.setData(modelo.index(2, ColumnsModel.ROLE), engine.CLASS)
banco.esperar()
modelo.setData(modelo.index(1, ColumnsModel.NAME), "letra")
banco.esperar()
ajustes = banco.w.settingsHandler.pack_data(banco.w)

nuevo = Banco.__new__(Banco)
nuevo.salida = "sin enviar"
nuevo.w = OWFileImport.__new__(OWFileImport, signal_manager=None, stored_settings=dict(ajustes))
nuevo.w.__init__()
nuevo.w.Outputs.data.send = nuevo._recibir
nuevo.esperar()
if check(isinstance(nuevo.salida, Table), f"se restaura desde el workflow: {nuevo.error()}"):
    check([v.name for v in nuevo.salida.domain.class_vars] == ["c"], "se restaura el rol")
    check("letra" in tipos(nuevo.salida), "se restaura el nombre")
    check(nuevo.w.delimiter_combo.currentData() == ";", "se restaura el separador")
nuevo.cerrar()

escribir("estado.csv", "a;nueva;c;otra\n1;x;9;1\n2;y;8;2\n")
cambiado = Banco.__new__(Banco)
cambiado.salida = "sin enviar"
cambiado.w = OWFileImport.__new__(OWFileImport, signal_manager=None, stored_settings=dict(ajustes))
cambiado.w.__init__()
cambiado.w.Outputs.data.send = cambiado._recibir
cambiado.esperar()
if check(isinstance(cambiado.salida, Table), "columnas cambiadas: sigue cargando"):
    check(cambiado.w.Warning.columns_changed.is_shown()
          and "b" in str(cambiado.w.Warning.columns_changed)
          and "nueva" in str(cambiado.w.Warning.columns_changed), "columnas cambiadas: avisa")
cambiado.cerrar()

os.remove(ruta("estado.csv"))
perdido = Banco.__new__(Banco)
perdido.salida = "sin enviar"
perdido.w = OWFileImport.__new__(OWFileImport, signal_manager=None, stored_settings=dict(ajustes))
perdido.w.__init__()
perdido.w.Outputs.data.send = perdido._recibir
perdido.esperar()
check(perdido.salida is None and "No existe" in perdido.error(), "archivo desaparecido: mensaje claro")
check(perdido.w._config().get("columns"), "archivo desaparecido: se conserva la configuración")
perdido.cerrar()
check(not any(v is not None and hasattr(v, "__reduce__") and isinstance(v, np.ndarray)
              for v in ajustes.values()), "los ajustes no guardan datos")
Options_vacias = engine.Options.from_dict({"format": "raro", "skip_rows": "x", "desconocida": 1})
check(Options_vacias.format == engine.AUTO and Options_vacias.skip_rows == 0, "ajustes corruptos: valores por defecto")

# ------------------------------------------------------------------------ URL ----

CSV_URL = TEXTO_COMA.encode()
XLSX_URL = open(ruta("base.xlsx"), "rb").read()
GRANDE_URL = b"a,b\n" + b"1,2\n" * 2000000
CUERPO_BOMBA = b"1,2\n" * (10 * 1024 * 1024)  # 40 MB que comprimen a unos pocos KB


def comprimir_zip(cuerpo, miembros=1, metodo=zipfile.ZIP_DEFLATED):
    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w", metodo) as z:
        for numero in range(miembros):
            z.writestr(f"datos{numero}.csv", cuerpo)
    return memoria.getvalue()


def declarar_tamano(datos, tamano):
    """Falsea el tamaño descomprimido que declara el directorio central del ZIP."""
    posicion = datos.index(b"PK\x01\x02")
    return datos[:posicion + 24] + struct.pack("<I", tamano) + datos[posicion + 28:]


def xlsx_bomba():
    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
        z.writestr("xl/worksheets/sheet1.xml", b" " * (40 * 1024 * 1024))
    return memoria.getvalue()


BOMBAS = {
    "b.csv.gz": gzip.compress(CUERPO_BOMBA),
    "b.csv.bz2": bz2.compress(CUERPO_BOMBA),
    "b.csv.xz": lzma.compress(CUERPO_BOMBA, preset=1),
    "b.csv.zip": comprimir_zip(CUERPO_BOMBA),
    "b.xlsx": xlsx_bomba(),
}
LEGITIMOS = {
    "l.csv.gz": gzip.compress(CSV_URL),
    "l.csv.bz2": bz2.compress(CSV_URL),
    "l.csv.xz": lzma.compress(CSV_URL),
    "l.csv.zip": comprimir_zip(CSV_URL),
    "dos.zip": comprimir_zip(CSV_URL, miembros=2),
    "corto.csv.zip": declarar_tamano(comprimir_zip(CSV_URL), 10),
    "largo.csv.zip": declarar_tamano(comprimir_zip(CSV_URL), 2 ** 31),
}


class Manejador(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def enviar(self, cuerpo, tipo="text/csv", extra=None):
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        for clave, valor in (extra or {}).items():
            self.send_header(clave, valor)
        self.end_headers()
        self.wfile.write(cuerpo)

    def redirigir(self, destino):
        self.send_response(302)
        self.send_header("Location", destino)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        ruta_url = self.path.split("?")[0]
        if ruta_url == "/datos.csv":
            self.enviar(CSV_URL)
        elif ruta_url == "/libro.xlsx":
            self.enviar(XLSX_URL, "application/octet-stream")
        elif ruta_url == "/descarga":
            self.enviar(CSV_URL, "application/octet-stream",
                        {"Content-Disposition": 'attachment; filename="otra cosa.csv"'})
        elif ruta_url == "/404":
            self.send_error(404, "No encontrado")
        elif ruta_url == "/500":
            self.send_error(500, "Fallo")
        elif ruta_url == "/redirige":
            self.redirigir("/datos.csv")
        elif ruta_url == "/bucle":
            self.redirigir("/bucle")
        elif ruta_url == "/afile":
            self.redirigir("file:///etc/passwd")
        elif ruta_url == "/web":
            self.enviar(b"<html><body>Portada</body></html>", "text/html; charset=utf-8")
        elif ruta_url == "/web_disfrazada":
            self.enviar(b"<!doctype html><html><body>x</body></html>", "application/octet-stream")
        elif ruta_url == "/enorme":
            self.enviar(GRANDE_URL)
        elif ruta_url == "/sin_longitud":
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            for _ in range(40):
                self.wfile.write(b"1,2\n" * 200000)
        elif ruta_url == "/lento":
            time.sleep(4)
            self.enviar(CSV_URL)
        elif ruta_url == "/cortada":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"a,b\n1,2\n")
        elif ruta_url == "/vacia":
            self.enviar(b"")
        elif ruta_url.startswith("/bomba/"):
            self.enviar(BOMBAS[ruta_url[7:]], "application/octet-stream")
        elif ruta_url.startswith("/legit/"):
            self.enviar(LEGITIMOS[ruta_url[7:]], "application/octet-stream")
        elif ruta_url == "/goteo":
            self.send_response(200)
            self.send_header("Content-Length", "45")
            self.end_headers()
            for _ in range(45):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.2)
        elif ruta_url == "/goteo_html":
            cuerpo = b"<html><body>hola</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            for posicion in range(len(cuerpo)):
                self.wfile.write(cuerpo[posicion:posicion + 1])
                self.wfile.flush()
                time.sleep(0.01)
        elif ruta_url == "/mudo":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"a,b\n1")
            self.wfile.flush()
            time.sleep(8)
        elif ruta_url == "/sin_cabeceras":
            time.sleep(8)
            self.enviar(CSV_URL)
        else:
            self.send_error(404)


class Servidor(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


servidor = Servidor(("127.0.0.1", 0), Manejador)
threading.Thread(target=servidor.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{servidor.server_address[1]}"

comprobar_base(banco.url(BASE + "/datos.csv"), "URL con CSV")
temporal = banco.w._download.path
check(os.path.dirname(temporal) == banco.w._temp_dir, "URL: se descarga a la carpeta temporal del widget")
comprobar_base(banco.url(BASE + "/libro.xlsx"), "URL con Excel")
comprobar_base(banco.url(BASE + "/descarga"), "URL con Content-Disposition")
check(banco.salida.name == "otra cosa", "URL: el nombre viene del servidor")
comprobar_base(banco.url(BASE + "/redirige"), "URL con redirección")
check(banco.w.Information.details.is_shown(), "URL con redirección: se indica")
falla(banco.url(BASE + "/404"), "URL 404", "404")
falla(banco.url(BASE + "/500"), "URL 500", "500")
falla(banco.url(BASE + "/bucle"), "URL en bucle", "redirige")
falla(banco.url(BASE + "/afile"), "URL que redirige a file", "no permitido")
falla(banco.url(BASE + "/web"), "URL que devuelve HTML", "HTML")
falla(banco.url(BASE + "/web_disfrazada"), "URL con HTML disfrazado", "HTML")
falla(banco.url(BASE + "/cortada"), "URL cortada")
falla(banco.url(BASE + "/vacia"), "URL vacía", "dato")
falla(banco.url("file:///etc/passwd"), "esquema file", "http")
falla(banco.url("ftp://127.0.0.1/x"), "esquema ftp", "http")
falla(banco.url("http://127.0.0.1:9/x.csv"), "URL sin servidor", "conectar")

error_tiempo = None
try:
    engine.download(BASE + "/lento", DIR, timeout=1)
except engine.ImportFailure as error:
    error_tiempo = str(error)
check(error_tiempo and "tiempo" in error_tiempo, f"URL lenta: tiempo agotado ({error_tiempo})")
for destino in ("/enorme", "/sin_longitud"):
    error_tamano = None
    try:
        engine.download(BASE + destino, DIR, max_megabytes=1)
    except engine.ImportFailure as error:
        error_tamano = str(error)
    check(error_tamano and "límite" in error_tamano, f"URL {destino}: límite de tamaño ({error_tamano})")
restos = [n for n in os.listdir(DIR) if n.startswith("url-")]
check(not restos, f"las descargas fallidas no dejan temporales: {restos}")

# Descargas lentas: el límite total y la cancelación no dependen de que llegue un bloque.
def cronometrar(url, directorio=None, **opciones):
    inicio = time.perf_counter()
    error = None
    try:
        engine.download(url, directorio or DIR, **opciones)
    except (engine.ImportFailure, engine.ImportCancelled) as excepcion:
        error = excepcion
    return time.perf_counter() - inicio, error


duracion, error = cronometrar(BASE + "/goteo", deadline=2)
check(isinstance(error, engine.ImportFailure) and "tiempo máximo de 2 s" in str(error),
      f"goteo: salta el límite total ({error})")
check(1.8 <= duracion < 3, f"goteo: límite total a los {duracion:.2f} s")


def cancelar_a(segundos):
    inicio = time.perf_counter()
    return engine.Progress(is_cancelled=lambda: time.perf_counter() - inicio > segundos)


for ruta_lenta in ("/goteo", "/mudo", "/sin_cabeceras"):
    carpeta_lenta = tempfile.mkdtemp(dir=DIR)
    duracion, error = cronometrar(BASE + ruta_lenta, carpeta_lenta, progress=cancelar_a(1))
    check(isinstance(error, engine.ImportCancelled), f"{ruta_lenta}: se cancela ({error!r})")
    check(duracion < 2, f"{ruta_lenta}: cancelada a 1 s y atendida a {duracion:.2f} s")
duracion, error = cronometrar(BASE + "/goteo_html")
check(isinstance(error, engine.ImportFailure) and "HTML" in str(error),
      f"HTML que llega byte a byte: se detecta ({error})")

banco.salida = "sin enviar"
banco.w.open_url(BASE + "/goteo")
inicio = time.time()
while time.time() - inicio < 1:
    app.processEvents()
    time.sleep(0.005)
check(banco.w.task is not None, "goteo en el widget: sigue descargando")
inicio = time.perf_counter()
banco.w.cancel_import()
check(banco.w.task is None and time.perf_counter() - inicio < 1, "goteo en el widget: cancelar es inmediato")
check(banco.salida == "sin enviar", "goteo en el widget: no llega resultado")

# Bombas de compresión: se cortan con su mensaje y sin dejar temporales.
check(engine.decompression_limit(40 * 1024, 500) == 16 * 1024 * 1024, "límite: mínimo para archivos pequeños")
check(engine.decompression_limit(10 * 1024 * 1024, 500) == 1000 * 1024 * 1024, "límite: 100 veces lo descargado")
check(engine.decompression_limit(400 * 1024 * 1024, 500) == 5000 * 1024 * 1024, "límite: 10 veces la descarga máxima")
check(engine.decompression_limit(40 * 1024, 1) == 10 * 1024 * 1024, "límite: respeta una descarga máxima pequeña")
for nombre in BOMBAS:
    falla(banco.url(f"{BASE}/bomba/{nombre}"), f"bomba {nombre}", "descomprimirlos", "archivo local")
    check(banco.w._download is None, f"bomba {nombre}: olvida la descarga")
    check(not os.listdir(banco.w._temp_dir), f"bomba {nombre}: sin temporales {os.listdir(banco.w._temp_dir)}")

for nombre in ("b.csv.gz", "b.csv.bz2", "b.csv.xz", "b.csv.zip"):
    path = escribir(nombre, BOMBAS[nombre])
    limite = engine.decompression_limit(len(BOMBAS[nombre]), 500)
    for texto, kwargs in (("vista previa", dict(limit=engine.INFERENCE_ROWS, head_only=True, max_output=1000)),
                          ("carga completa", dict(max_output=limite))):
        cortada = False
        try:
            engine.load(path, engine.Options(), nombre, **kwargs)
        except engine.DecompressionLimit:
            cortada = True
        check(cortada, f"bomba {nombre}: se corta en la {texto}")

cabecera_arff = "@relation b\n@attribute a numeric\n@attribute b numeric\n@data\n"
for etiqueta, fila in (("densa", "1,2\n"), ("dispersa", "{0 1,1 2}\n")):
    comprimido = gzip.compress(cabecera_arff.encode() + fila.encode() * 400000)
    path = escribir(f"bomba_{etiqueta}.arff.gz", comprimido)
    cortada = False
    try:
        engine.load(path, engine.Options(), f"bomba_{etiqueta}.arff.gz", max_output=100000)
    except engine.DecompressionLimit:
        cortada = True
    check(cortada, f"bomba ARFF {etiqueta}: se corta")
    t2 = engine.load(path, engine.Options(), f"bomba_{etiqueta}.arff.gz")
    check(t2.n_rows == 400000, f"ARFF {etiqueta} local: no se limita")

# Lo legítimo sigue funcionando, por URL y en local.
for nombre in ("l.csv.gz", "l.csv.bz2", "l.csv.xz", "l.csv.zip"):
    comprobar_base(banco.url(f"{BASE}/legit/{nombre}"), f"legítimo por URL {nombre}")
    comprobar_base(banco.abrir(escribir(nombre, LEGITIMOS[nombre])), f"legítimo local {nombre}")
falla(banco.url(f"{BASE}/legit/dos.zip"), "ZIP con varios miembros por URL", "exactamente un archivo")
falla(banco.url(f"{BASE}/legit/corto.csv.zip"), "ZIP que declara menos de lo que tiene", "dañado")
comprobar_base(banco.url(f"{BASE}/legit/largo.csv.zip"), "ZIP que declara más de lo que tiene")

local = escribir("grande_local.csv.gz", gzip.compress(CUERPO_BOMBA[:20 * 1024 * 1024]))
t = banco.abrir(local)
correcto(t, "un archivo local muy comprimido no se limita", 5 * 1024 * 1024)
cancelado = False
try:
    engine.load(local, engine.Options(), "grande_local.csv.gz",
                progress=engine.Progress(is_cancelled=lambda: True))
except engine.ImportCancelled:
    cancelado = True
check(cancelado, "un archivo local comprimido se puede cancelar")

# Arrastrar y soltar
from AnyQt.QtCore import QUrl  # noqa: E402

check(banco.w.canDropUrl(QUrl.fromLocalFile(ruta("base.csv"))), "arrastrar: admite archivos")
check(banco.w.canDropUrl(QUrl(BASE + "/datos.csv")), "arrastrar: admite http")
check(not banco.w.canDropUrl(QUrl("ftp://x/y")), "arrastrar: rechaza otros esquemas")
banco.salida = "sin enviar"
banco.w.handleDroppedUrl(QUrl.fromLocalFile(ruta("base.csv")))
banco.esperar()
comprobar_base(banco.salida, "arrastrar un archivo")
banco.salida = "sin enviar"
banco.w.handleDroppedUrl(QUrl(BASE + "/datos.csv"))
banco.esperar()
comprobar_base(banco.salida, "arrastrar una URL")

# Mientras se descarga, la interfaz sigue respondiendo.
banco.salida = "sin enviar"
banco.w.open_url(BASE + "/lento")
inicio = time.time()
vueltas = 0
while banco.w.task is not None and time.time() - inicio < 20:
    app.processEvents()
    vueltas += 1
    time.sleep(0.001)
banco.esperar()
check(vueltas > 200, f"la red no bloquea la interfaz ({vueltas} vueltas de eventos)")
comprobar_base(banco.salida, "URL lenta con tiempo suficiente")

servidor.shutdown()

carpeta = banco.w._temp_dir
banco.cerrar()
check(not os.path.exists(carpeta), "al cerrar se borra la carpeta temporal")
check(not excepciones, "sin excepciones sin capturar:\n" + "\n".join(excepciones))
shutil.rmtree(DIR, ignore_errors=True)

print("\nFALLOS:", fallos)
sys.exit(1 if fallos else 0)
