"""Motor de lectura del widget File Import (sin Qt).

Todos los formatos terminan en el mismo sitio: un DataFrame con una columna por
cada columna del archivo y una lista de ColumnMeta con lo inferido o declarado.
Convertir a Table es un paso aparte y barato: cambiar el tipo o el rol de una
columna no vuelve a leer el archivo.

Los formatos de texto pasan siempre por el motor C de pandas. Se le da el archivo
a través de _Feed, un objeto con read() que transcodifica a UTF-8, cuenta las
filas con un número de campos incorrecto (con numpy, por bloques), informa del
progreso y permite cancelar, todo sin una segunda lectura.
"""
import bz2
import codecs
import csv
import gzip
import http.client
import io
import lzma
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Orange.data import (
    ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table,
    TimeVariable,
)
from Orange.misc.collections import natural_sorted

NUMERIC, CATEGORICAL, TEXT, DATE = "numeric", "categorical", "text", "date"
TYPES = (NUMERIC, CATEGORICAL, TEXT, DATE)
ATTRIBUTE, CLASS, META, SKIP = "attribute", "class", "meta", "skip"
ROLES = (ATTRIBUTE, CLASS, META, SKIP)

AUTO = "auto"
FORMATS = (AUTO, "csv", "tsv", "tab", "xlsx", "xls", "arff")
DELIMITED = ("csv", "tsv", "tab")

HEADER_AUTO, HEADER_NONE = -1, 0
DEFAULT_NA = "?, ., NA, N/A, NaN, nan, null, NULL, None, #N/A, ~"
EXCEL_ERRORS = frozenset(("#N/A", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?",
                          "#NUM!", "#NULL!", "#SPILL!", "#CALC!"))

HEAD_BYTES = 2 * 1024 * 1024
PREVIEW_ROWS = 100
INFERENCE_ROWS = 5000
MAX_HEAD_RECORDS = 6000
# Igual que Orange: como mucho 100 categorías deducidas automáticamente.
MAX_AUTO_CATEGORIES = 100
SAFE_INTEGER = 2 ** 53
CANCEL_EVERY = 2000
CLASS_NAMES = frozenset(("class", "clase", "target", "objetivo"))

DOWNLOAD_TIMEOUT = 20
DOWNLOAD_DEADLINE = 900
DEFAULT_MAX_DOWNLOAD_MB = 500
MAX_REDIRECTS = 5
USER_AGENT = "orange-gualbe-toolbox (File Import)"
DOWNLOAD_BLOCK = 256 * 1024


# Un campo de texto muy largo no debe impedir analizar la cabecera.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


class ImportFailure(Exception):
    """Fallo esperado: su texto se muestra tal cual en el widget."""


class ImportCancelled(Exception):
    """El usuario ha cancelado o ha empezado otra lectura."""


@dataclass
class Options:
    format: str = AUTO
    encoding: str = AUTO
    delimiter: str = AUTO
    quotechar: str = '"'
    decimal: str = AUTO
    thousands: str = AUTO
    skip_rows: int = 0
    header: int = HEADER_AUTO
    na_values: str = DEFAULT_NA
    comment: str = ""
    max_rows: int = 0
    sheet: str = ""
    date_order: str = AUTO
    max_download_mb: int = DEFAULT_MAX_DOWNLOAD_MB

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Reconstruye las opciones de un workflow, ignorando lo que no encaje."""
        options = cls()
        for item in fields(cls):
            value = (values or {}).get(item.name, getattr(options, item.name))
            if isinstance(value, type(getattr(options, item.name))):
                setattr(options, item.name, value)
        if options.format not in FORMATS:
            options.format = AUTO
        return options


@dataclass
class Progress:
    """Conecta el motor con quien lo ejecuta (hilo del widget o una prueba)."""
    report: Optional[Callable[[float], None]] = None
    is_cancelled: Optional[Callable[[], bool]] = None
    status: Optional[Callable[[str], None]] = None

    def check(self):
        if self.is_cancelled is not None and self.is_cancelled():
            raise ImportCancelled()

    def update(self, percent):
        if self.report is not None:
            self.report(min(percent, 100.0))

    def say(self, text):
        if self.status is not None:
            self.status(text)


@dataclass
class ColumnMeta:
    index: int
    source: str
    name: str
    type: str
    role: str
    n_missing: int = 0
    n_unique: int = 0
    samples: List[str] = field(default_factory=list)
    declared_values: Optional[tuple] = None
    date_format: Optional[str] = None
    n_invalid: int = 0


@dataclass
class Layout:
    """Lo que se sabe de un archivo de texto antes de leerlo entero."""
    encoding: str
    delimiter: str
    quotechar: Optional[str]
    decimal: str
    thousands: Optional[str]
    skip_records: int
    n_columns: int
    names: List[str]
    declared: Dict[int, dict]
    notes: List[str]
    header_row: int = HEADER_NONE
    body_offset: int = 0
    comment: Optional[str] = None
    escapechar: Optional[str] = None
    skip_initial_space: bool = False


@dataclass
class Loaded:
    frame: pd.DataFrame
    columns: List[ColumnMeta]
    format: str
    notes: List[str]
    name: str
    encoding: str = ""
    delimiter: str = ""
    decimal: str = "."
    thousands: str = ""
    header_row: int = HEADER_NONE
    sheets: List[str] = field(default_factory=list)
    sheet: str = ""
    complete: bool = True
    truncated_by_limit: bool = False

    @property
    def n_rows(self):
        return len(self.frame)


def parse_na_values(text):
    values = {part.strip() for part in text.split(",")}
    values.add("")
    return sorted(values)


# ---------------------------------------------------------------- apertura ----

_MAGIC = (
    (b"\x1f\x8b", lambda raw: gzip.GzipFile(fileobj=raw)),
    (b"BZh", bz2.BZ2File),
    (b"\xfd7zXZ\x00", lzma.LZMAFile),
)


def _raise_open_error(path, error):
    name = os.path.basename(path) or path
    if isinstance(error, FileNotFoundError):
        raise ImportFailure(f"No existe el archivo «{path}».") from error
    if isinstance(error, IsADirectoryError):
        raise ImportFailure(f"«{name}» es una carpeta, no un archivo.") from error
    if isinstance(error, PermissionError):
        raise ImportFailure(
            f"No hay permiso para leer «{name}». Revisa los permisos del "
            "archivo o cópialo a otra carpeta.") from error
    raise ImportFailure(f"No se puede abrir «{name}»: {error}") from error


class _Compressed:
    """Abre el archivo y descomprime si hace falta, según su contenido."""

    def __init__(self, path):
        self.size = 0
        try:
            self.size = os.path.getsize(path)
            self.raw = open(path, "rb")  # pylint: disable=consider-using-with
        except OSError as error:
            _raise_open_error(path, error)
        self.stream = self.raw
        self._member = None
        magic = self.raw.read(8)
        self.raw.seek(0)
        for prefix, opener in _MAGIC:
            if magic.startswith(prefix):
                self.stream = opener(self.raw)
                break
        else:
            if magic.startswith(b"PK\x03\x04") and zipfile.is_zipfile(self.raw):
                self.raw.seek(0)
                archive = zipfile.ZipFile(self.raw)  # pylint: disable=consider-using-with
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) != 1:
                    self.close()
                    raise ImportFailure(
                        "El archivo ZIP debe contener exactamente un archivo de "
                        f"datos y contiene {len(members)}.")
                self._member = archive
                self.stream = archive.open(members[0])

    def read(self, count):
        try:
            return self.stream.read(count)
        except (OSError, EOFError, lzma.LZMAError, zipfile.BadZipFile) as error:
            raise ImportFailure(f"El archivo comprimido está dañado: {error}") from error

    def tell(self):
        return self.raw.tell()

    def close(self):
        for item in (self.stream, self._member, self.raw):
            if item is not None:
                try:
                    item.close()
                except OSError:
                    pass


def read_head(path, limit=HEAD_BYTES):
    """Primeros bytes (ya descomprimidos) y si queda más por leer."""
    source = _Compressed(path)
    try:
        data = source.read(limit + 1)
    finally:
        source.close()
    return data[:limit], len(data) > limit


# ------------------------------------------------------ formato y codificación ----

def _looks_like_html(head):
    start = head[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return start.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def _sniff_excel_kind(path):
    """Devuelve 'xlsx', 'xls', 'encrypted' o None mirando el contenido."""
    try:
        with open(path, "rb") as stream:
            magic = stream.read(8)
            if magic.startswith(b"PK\x03\x04"):
                try:
                    with zipfile.ZipFile(stream) as archive:
                        names = set(archive.namelist())
                except zipfile.BadZipFile:
                    return None
                return "xlsx" if "xl/workbook.xml" in names else None
            if magic.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
                stream.seek(0)
                body = stream.read(4 * 1024 * 1024)
                marker = "EncryptedPackage".encode("utf-16-le")
                return "encrypted" if marker in body else "xls"
    except OSError as error:
        _raise_open_error(path, error)
    return None


def _extension(name):
    name = name.lower()
    for suffix in (".gz", ".bz2", ".xz", ".zip"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return os.path.splitext(name)[1]


_EXTENSION_FORMAT = {
    ".csv": "csv", ".txt": "csv", ".tsv": "tsv", ".tab": "tab", ".arff": "arff",
    ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xls",
}


def detect_format(path, display_name, requested=AUTO):
    """Devuelve (formato, avisos). El contenido manda sobre la extensión."""
    notes = []
    kind = _sniff_excel_kind(path)
    if kind == "encrypted":
        raise ImportFailure(
            "El libro de Excel está protegido con contraseña. Ábrelo en Excel, "
            "quita la contraseña y guárdalo de nuevo.")
    by_name = _EXTENSION_FORMAT.get(_extension(display_name))
    if kind is None:
        head, _ = read_head(path, 64 * 1024)
        if not head.strip():
            raise ImportFailure(f"El archivo «{os.path.basename(display_name)}» está vacío.")
        if _looks_like_html(head):
            raise ImportFailure(
                "El contenido es una página web (HTML), no una tabla de datos. "
                "Si viene de un enlace, comprueba que apunte directamente al archivo.")
        decoded = head[:4096].lstrip(b"\xef\xbb\xbf").lstrip().lower()
        if decoded.startswith((b"%", b"@relation")) and b"@relation" in head[:65536].lower():
            kind = "arff"
    if requested != AUTO:
        if requested in ("xlsx", "xls") and kind not in ("xlsx", "xls"):
            raise ImportFailure(
                f"Has elegido {requested.upper()} pero el contenido no es un libro "
                "de Excel. Cambia el formato a Automático.")
        if requested in ("xlsx", "xls") and kind != requested:
            requested = kind
        return requested, notes
    detected = kind or by_name or "csv"
    if kind is not None and by_name is not None and by_name != kind:
        extension = _extension(display_name)
        notes.append(f"La extensión es «{extension}» pero el contenido es "
                     f"{kind.upper()}: se lee como {kind.upper()}.")
    if kind is None and by_name in ("xlsx", "xls"):
        notes.append(f"La extensión es «{_extension(display_name)}» pero el "
                     "contenido es texto: se lee como texto delimitado.")
        detected = "csv"
    return detected, notes


def detect_encoding(head):
    """Devuelve el nombre de la codificación deducida de los primeros bytes."""
    for bom, name in ((codecs.BOM_UTF8, "utf-8-sig"),
                      (codecs.BOM_UTF32_LE, "utf-32"),
                      (codecs.BOM_UTF32_BE, "utf-32"),
                      (codecs.BOM_UTF16_LE, "utf-16"),
                      (codecs.BOM_UTF16_BE, "utf-16")):
        if head.startswith(bom):
            return name
    sample = head[:4096]
    if len(sample) >= 4:
        odd = sample[1::2].count(0) / max(len(sample) // 2, 1)
        even = sample[0::2].count(0) / max(len(sample) // 2, 1)
        if odd > 0.4 and even < 0.05:
            return "utf-16-le"
        if even > 0.4 and odd < 0.05:
            return "utf-16-be"
    try:
        codecs.getincrementaldecoder("utf-8")("strict").decode(head, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        head.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        return "latin-1"


def _check_encoding(name):
    try:
        codecs.lookup(name)
    except (LookupError, TypeError) as error:
        raise ImportFailure(f"La codificación «{name}» no existe.") from error


def _decode_head(head, encoding):
    decoder = codecs.getincrementaldecoder(encoding)("replace")
    return decoder.decode(head, final=False)


# ----------------------------------------------------- análisis de la cabecera ----

_TYPE_TOKENS = {
    "c": NUMERIC, "continuous": NUMERIC, "n": NUMERIC, "numeric": NUMERIC,
    "d": CATEGORICAL, "discrete": CATEGORICAL, "categorical": CATEGORICAL,
    "s": TEXT, "string": TEXT, "text": TEXT,
    "t": DATE, "time": DATE,
}
_FLAG_TOKENS = {"c": CLASS, "class": CLASS, "m": META, "meta": META,
                "i": SKIP, "ignore": SKIP, "w": "weight", "weight": "weight"}
_PREFIX = re.compile(r"^([cmiwCDST]{1,2})#(.*)$")
_NUMBER = re.compile(r"^\s*[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?\s*$")


def _sniff_delimiter(text, options):
    """Devuelve (separador, alternativas igual de plausibles)."""
    lines = [line for line in text.splitlines()[options.skip_rows:options.skip_rows + 200]
             if line.strip()]
    if options.comment:
        lines = [line for line in lines if not line.startswith(options.comment)]
    if len(lines) > 1:
        lines = lines[:-1] if len(lines) == 200 else lines
    scores = []
    for order, candidate in enumerate((",", ";", "\t", "|", " ", ":")):
        if candidate not in text:
            continue
        quote = options.quotechar or None
        reader = csv.reader(lines, delimiter=candidate, quotechar=quote,
                            quoting=csv.QUOTE_MINIMAL if quote else csv.QUOTE_NONE)
        try:
            widths = Counter(len(row) for row in reader)
        except csv.Error:
            continue
        width, hits = widths.most_common(1)[0]
        if width < 2:
            continue
        consistency = hits / sum(widths.values())
        if candidate in " :":
            consistency -= 0.1
        scores.append((consistency, -order, candidate, width))
    if not scores:
        return ",", []
    scores.sort(reverse=True)
    best = scores[0]
    rivals = [item[2] for item in scores[1:]
              if item[0] >= best[0] - 0.02 and item[2] in ",;\t|" and best[2] in ",;\t|"]
    return best[2], rivals


def _detect_number_format(rows, delimiter):
    """Deduce (decimal, miles) a partir de las celdas de las primeras filas."""
    cells = [cell.strip() for row in rows[:500] for cell in row if cell.strip()]
    if not cells:
        return ".", None
    european = sum(bool(re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", c)) for c in cells)
    european_plain = sum(bool(re.fullmatch(r"[-+]?\d+,\d+", c)) for c in cells)
    anglo = sum(bool(re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", c)) for c in cells)
    point = sum(bool(re.fullmatch(r"[-+]?\d+\.\d+", c)) for c in cells)
    if delimiter != "," and european_plain and european_plain >= point:
        multi = sum(bool(re.search(r"\d\.\d{3}(\.\d{3}|,\d)", c)) for c in cells)
        return ",", "." if multi or european else None
    if european and delimiter != ".":
        multi = sum(bool(re.search(r"\.\d{3}\.\d{3}|\.\d{3},\d", c)) for c in cells)
        if multi:
            return ",", "."
    if anglo and delimiter != ",":
        return ".", ","
    return ".", None


def _parse_orange_three_rows(rows):
    """Reconoce la cabecera de tres filas de Orange y devuelve lo declarado."""
    if len(rows) < 3:
        return None
    names, types, flags = rows[0], rows[1], rows[2]
    width = max(len(names), len(types), len(flags))
    if not any(cell.strip() for cell in types):
        return None
    declared = {}
    recognised = 0
    for index in range(width):
        type_cell = types[index].strip() if index < len(types) else ""
        flag_cell = flags[index].strip() if index < len(flags) else ""
        item = {}
        token = type_cell.lower()
        if token in _TYPE_TOKENS:
            item["type"] = _TYPE_TOKENS[token]
            recognised += 1
        elif " " in type_cell:
            item["type"] = CATEGORICAL
            item["values"] = tuple(type_cell.replace("\\ ", "\0").split())
            item["values"] = tuple(v.replace("\0", " ") for v in item["values"])
            recognised += 1
        elif type_cell:
            return None
        for part in flag_cell.split():
            if "=" in part:
                continue
            role = _FLAG_TOKENS.get(part.lower())
            if role is None:
                return None
            item["role"] = role
        declared[index] = item
    return declared if recognised else None


def _is_header_row(first, rest):
    """Decide si la primera fila es de nombres y no de datos."""
    filled = [cell for cell in first if cell.strip()]
    if not filled:
        return False
    if any(_NUMBER.match(cell) for cell in filled):
        return False
    if not rest:
        return True
    width = len(first)
    numeric_below = 0
    for column in range(width):
        cells = [row[column] for row in rest[:50]
                 if column < len(row) and row[column].strip()]
        if cells and sum(bool(_NUMBER.match(cell)) for cell in cells) > len(cells) / 2:
            numeric_below += 1
    if numeric_below:
        return True
    if len(set(filled)) < len(filled):
        return False
    return len(filled) >= len([cell for cell in first]) * 0.5


def unique_names(names):
    """Da nombre a las columnas vacías y desambigua las repetidas."""
    empty = duplicated = 0
    used = set()
    result = []
    for index, raw in enumerate(names):
        name = re.sub(r"[\x00-\x1f]+", " ", str(raw)).strip()
        if not name:
            empty += 1
            name = f"Columna {index + 1}"
        if name in used:
            duplicated += 1
            base, suffix = name, 2
            while f"{base} ({suffix})" in used:
                suffix += 1
            name = f"{base} ({suffix})"
        used.add(name)
        result.append(name)
    return result, empty, duplicated


def _resolve_header(records, options, notes):
    """Localiza la cabecera. Devuelve (nº de registros previos a los datos,
    nombres, lo declarado, fila de cabecera 1-based o 0)."""
    candidates = []
    for position, row in enumerate(records):
        if position < options.skip_rows or not row or not any(c.strip() for c in row):
            continue
        if options.comment and row[0].startswith(options.comment):
            continue
        candidates.append((position, row))
    if not candidates:
        return None

    width = Counter(len(row) for _, row in candidates[:500]).most_common(1)[0][0]
    data_rows = [row for _, row in candidates]
    declared = {}
    if options.header == HEADER_NONE:
        names = [""] * width
        return candidates[0][0], names, declared, HEADER_NONE

    header_at = 0
    if options.header > 0:
        header_at = options.header - 1
        if header_at >= len(candidates):
            raise ImportFailure(
                f"La fila de cabecera {options.header} no existe: solo hay "
                f"{len(candidates)} filas en las opciones actuales.")
    if options.header == HEADER_AUTO or options.header == 1:
        three = _parse_orange_three_rows(data_rows[:3])
        if three is not None:
            names = list(data_rows[0])
            return (candidates[2][0] + 1, names, three, 1)
    if options.header == HEADER_AUTO and not _is_header_row(data_rows[0], data_rows[1:]):
        names = [""] * width
        return candidates[0][0], names, declared, HEADER_NONE

    names = list(data_rows[header_at])
    if any(_PREFIX.match(cell.strip()) for cell in names):
        stripped = []
        for index, cell in enumerate(names):
            match = _PREFIX.match(cell.strip())
            if not match:
                stripped.append(cell)
                continue
            item = {}
            for letter in match.group(1):
                if letter in "CDST":
                    item["type"] = {"C": NUMERIC, "D": CATEGORICAL,
                                    "S": TEXT, "T": DATE}[letter]
                elif letter in "cmiw":
                    item["role"] = _FLAG_TOKENS[letter]
            declared[index] = item
            stripped.append(match.group(2))
        names = stripped
    return candidates[header_at][0] + 1, names, declared, header_at + 1


def _head_limit(options):
    """Bytes iniciales necesarios: las filas saltadas no deben comerse la muestra."""
    return min(HEAD_BYTES + options.skip_rows * 4096, 512 * 1024 * 1024)


def analyse_delimited(path, options, fmt):
    """Mira solo el principio del archivo y deduce cómo leerlo."""
    head, truncated = read_head(path, _head_limit(options))
    notes = []
    if not head.strip():
        raise ImportFailure("El archivo está vacío.")
    if options.encoding == AUTO:
        encoding = detect_encoding(head)
    else:
        encoding = options.encoding
        _check_encoding(encoding)
    if b"\0" in head[:65536] and not encoding.startswith(("utf-16", "utf-32")):
        raise ImportFailure(
            "El archivo parece binario y no un texto delimitado. Comprueba el "
            "formato o la codificación.")
    text = _decode_head(head, encoding).lstrip("﻿")
    if not text.strip():
        raise ImportFailure("El archivo está vacío.")

    if options.delimiter != AUTO:
        delimiter = "\t" if options.delimiter in ("\\t", "tab") else options.delimiter
    elif fmt in ("tsv", "tab"):
        delimiter = "\t"
    else:
        delimiter, rivals = _sniff_delimiter(text, options)
        if rivals:
            shown = ", ".join("tabulador" if r == "\t" else f"«{r}»" for r in rivals)
            sep = "tabulador" if delimiter == "\t" else f"«{delimiter}»"
            notes.append(f"El separador es ambiguo: se usa {sep} pero {shown} "
                         "también encaja. Compruébalo en la vista previa.")
    if len(delimiter) != 1 or delimiter in "\r\n":
        raise ImportFailure("El separador debe ser un único carácter.")

    quote = options.quotechar or None
    if quote and len(quote) != 1:
        raise ImportFailure("El carácter de comillas debe ser uno solo.")
    if quote == delimiter:
        raise ImportFailure("Las comillas y el separador no pueden ser el mismo carácter.")
    if options.comment and len(options.comment) != 1:
        raise ImportFailure("El marcador de comentario debe ser un único carácter.")

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter,
                        quotechar=quote,
                        quoting=csv.QUOTE_MINIMAL if quote else csv.QUOTE_NONE)
    records = []
    try:
        for row in reader:
            records.append(row)
            if len(records) >= MAX_HEAD_RECORDS + options.skip_rows:
                break
    except csv.Error as error:
        raise ImportFailure(f"No se puede interpretar el archivo: {error}") from error
    if truncated and len(records) > 1:
        records.pop()

    found = _resolve_header(records, options, notes)
    if found is None:
        raise ImportFailure("El archivo no tiene filas con datos con las opciones actuales.")
    skip_records, names, declared, header_row = found

    body = [row for row in records[skip_records:] if any(cell.strip() for cell in row)]
    if options.comment:
        body = [row for row in body if not row[0].startswith(options.comment)]
    width = Counter(len(row) for row in body[:500]).most_common(1)[0][0] if body else len(names)
    n_columns = max(width, len(names))
    names = list(names) + [""] * (n_columns - len(names))

    decimal, thousands = _detect_number_format(body, delimiter)
    if options.decimal != AUTO:
        decimal = options.decimal
    if options.thousands != AUTO:
        thousands = options.thousands or None
    if decimal == thousands:
        raise ImportFailure("Los separadores decimal y de miles no pueden ser iguales.")
    if decimal == delimiter or thousands == delimiter:
        raise ImportFailure(
            "El separador de campos no puede ser también el separador decimal o "
            "el de miles. Cambia uno de los dos.")
    return Layout(encoding, delimiter, quote, decimal, thousands, skip_records,
                  n_columns, names, declared, notes, header_row=header_row,
                  comment=options.comment or None)


# ------------------------------------------------------------- lectura rápida ----

class _FieldCounter:
    """Cuenta, por bloques y con numpy, las filas con campos de más o de menos."""

    def __init__(self, delimiter, quote, expected, skip_records, comment, max_records):
        self.delimiter = ord(delimiter)
        self.quote = ord(quote) if quote else None
        self.expected = expected
        self.skip = skip_records
        self.comment = ord(comment) if comment else None
        self.max_records = max_records or None
        self.in_quote = False
        self.pending_fields = 0
        self.pending_length = 0
        self.first_byte = None
        self.records = 0
        self.data_records = 0
        self.long = self.short = 0
        self.first_bad_record = None
        self.saw_quote = False

    def feed(self, data):
        if not data or (self.max_records and self.data_records >= self.max_records):
            return
        array = np.frombuffer(data, np.uint8)
        delimiters = array == self.delimiter
        newlines = array == 10
        if self.quote is not None:
            quotes = array == self.quote
            if quotes.any():
                self.saw_quote = True
                inside = np.bitwise_xor.accumulate(quotes)
                if self.in_quote:
                    inside = ~inside
                self.in_quote = bool(inside[-1])
                delimiters &= ~inside
                newlines &= ~inside
            elif self.in_quote:
                return
        ends = np.flatnonzero(newlines)
        positions = np.flatnonzero(delimiters)
        if self.pending_length == 0 and self.first_byte is None:
            self.first_byte = int(array[0])
        if not len(ends):
            self.pending_fields += len(positions)
            self.pending_length += len(array)
            return
        before = np.searchsorted(positions, ends)
        counts = np.diff(before, prepend=0) + 1
        counts[0] += self.pending_fields
        starts = np.concatenate(([0], ends[:-1] + 1))
        lengths = ends - starts
        lengths[0] += self.pending_length
        firsts = array[np.minimum(starts, len(array) - 1)]
        if self.first_byte is not None:
            firsts[0] = self.first_byte
        tail = array[ends - 1] if self.pending_length == 0 else None
        self._account(counts, lengths, firsts, array, ends, tail)
        self.pending_fields = len(positions) - int(before[-1])
        self.pending_length = len(array) - int(ends[-1]) - 1
        self.first_byte = int(array[ends[-1] + 1]) if self.pending_length else None

    def _account(self, counts, lengths, firsts, array, ends, _tail):
        index = np.arange(self.records, self.records + len(counts))
        self.records += len(counts)
        carriage = (array[np.maximum(ends - 1, 0)] == 13) & (lengths <= 1)
        keep = (index >= self.skip) & (lengths > 0) & ~carriage
        if self.comment is not None:
            keep &= firsts != self.comment
        self._tally(counts[keep], index[keep])

    def _tally(self, counts, index):
        if self.max_records:
            room = self.max_records - self.data_records
            counts, index = counts[:room], index[:room]
        self.data_records += len(counts)
        long_mask = counts > self.expected
        short_mask = counts < self.expected
        self.long += int(long_mask.sum())
        self.short += int(short_mask.sum())
        if self.first_bad_record is None and (long_mask.any() or short_mask.any()):
            self.first_bad_record = int(index[np.argmax(long_mask | short_mask)])

    def finish(self):
        """Tiene en cuenta una última línea sin salto de línea."""
        if self.pending_length and self.records >= self.skip and not self.in_quote:
            if self.comment is None or self.first_byte != self.comment:
                self._tally(np.array([self.pending_fields + 1]), np.array([self.records]))
        self.pending_length = 0

    @property
    def reliable(self):
        if self.in_quote:
            return False
        bad = self.long + self.short
        return not (self.saw_quote and self.data_records and bad > self.data_records / 2)


class _Feed:
    """Objeto de solo lectura que consume el motor C de pandas.

    pandas envuelve en TextIOWrapper lo que detecta como binario; este objeto no
    tiene `mode` y entrega UTF-8 ya validado, así que el parser lo lee directo.
    """

    def __init__(self, source, encoding, counter, progress, size, skip_bytes=0,
                 cap_bytes=None):
        self.source = source
        self.counter = counter
        self.progress = progress
        self.size = max(size, 1)
        self.cap = cap_bytes
        self.delivered = 0
        self.encoding = codecs.lookup(encoding).name
        self.strip_bom = self.encoding == "utf-8-sig"
        self.pass_through = self.encoding in ("utf-8", "utf-8-sig", "ascii")
        decoder_name = "utf-8" if self.pass_through else self.encoding
        self.decoder = codecs.getincrementaldecoder(decoder_name)("strict")
        if skip_bytes:
            self.source.read(skip_bytes)

    def __iter__(self):  # pandas exige que el objeto sea iterable
        return self

    def __next__(self):
        raise StopIteration

    def read(self, count=-1):
        self.progress.check()
        if count is None or count < 0:
            count = 1024 * 1024
        if self.cap is not None:
            count = min(count, self.cap - self.delivered)
            if count <= 0:
                return b""
        data = self.source.read(count)
        self.delivered += len(data)
        final = len(data) == 0
        if self.pass_through:
            if self.strip_bom:
                self.strip_bom = False
                if data.startswith(codecs.BOM_UTF8):
                    data = data[3:]
            if not data.isascii():
                self.decoder.decode(data, final=False)
            elif final:
                self.decoder.decode(b"", final=True)
        else:
            data = self.decoder.decode(data, final=final).encode("utf-8")
        if self.counter is not None:
            self.counter.feed(data)
        if self.source.size:
            self.progress.update(55.0 * self.source.tell() / self.size)
        return data


def _parser_failure(error, layout):
    text = str(error)
    match = re.search(r"EOF inside string starting (?:at|with) (?:row|line) (\d+)", text)
    if match or "EOF inside string" in text:
        where = f" (empieza en la fila {match.group(1)})" if match else ""
        return ImportFailure(
            f"Hay unas comillas sin cerrar{where}. Corrige el archivo o desactiva "
            "las comillas en la importación avanzada.")
    return ImportFailure(f"No se puede interpretar el archivo: {text}")


def _read_with_c_engine(path, options, layout, limit, progress, head_only):
    """Lee las filas de datos con el motor C. Devuelve (frame, contador)."""
    names = [f"c{i}" for i in range(layout.n_columns)]
    dtype = {}
    for index, item in layout.declared.items():
        if item.get("type") in (CATEGORICAL, TEXT) and index < layout.n_columns:
            dtype[names[index]] = str
    if head_only:
        # El bloque inicial puede cortar un campo entre comillas: se amplía antes
        # de concluir que las comillas están realmente sin cerrar.
        first = _head_limit(options)
        for cap in (first, 8 * first, 32 * first):
            try:
                return _parse_block(path, options, layout, limit, progress, names,
                                    dtype, cap)
            except _BlockTooSmall:
                continue
        raise ImportFailure("Hay unas comillas sin cerrar al principio del archivo. "
                            "Corrige el archivo o desactiva las comillas en la "
                            "importación avanzada.")
    return _parse_block(path, options, layout, limit, progress, names, dtype, None)


class _BlockTooSmall(Exception):
    pass


def _parse_block(path, options, layout, limit, progress, names, dtype, cap):
    head_only = cap is not None
    source = _Compressed(path)
    counter = _FieldCounter(layout.delimiter, layout.quotechar, layout.n_columns,
                            layout.skip_records, layout.comment, limit)
    feed = _Feed(source, layout.encoding, counter, progress, source.size,
                 skip_bytes=layout.body_offset, cap_bytes=cap)
    keywords = {}
    if layout.thousands:
        keywords["thousands"] = layout.thousands
    if layout.escapechar:
        keywords["escapechar"] = layout.escapechar
    try:
        frame = pd.read_csv(
            feed, engine="c", sep=layout.delimiter, header=None, names=names,
            usecols=list(range(layout.n_columns)), index_col=False,
            skiprows=layout.skip_records, nrows=limit or None,
            quotechar=layout.quotechar or '"',
            quoting=csv.QUOTE_MINIMAL if layout.quotechar else csv.QUOTE_NONE,
            decimal=layout.decimal, comment=layout.comment,
            na_values=parse_na_values(options.na_values), keep_default_na=False,
            skip_blank_lines=True, skipinitialspace=layout.skip_initial_space,
            low_memory=False, dtype=dtype or None, encoding="utf-8", **keywords)
    except ImportCancelled:
        raise
    except UnicodeDecodeError:
        raise
    except pd.errors.EmptyDataError as error:
        raise ImportFailure("El archivo no tiene filas de datos con las opciones "
                            "actuales.") from error
    except pd.errors.ParserError as error:
        if head_only and "EOF inside string" in str(error) and feed.delivered >= cap:
            raise _BlockTooSmall() from error
        raise _parser_failure(error, layout) from error
    except (ValueError, OverflowError) as error:
        raise ImportFailure(f"No se puede interpretar el archivo: {error}") from error
    finally:
        source.close()
    if head_only:
        if feed.delivered >= cap and len(frame) > 1 and (not limit or len(frame) < limit):
            frame = frame.iloc[:-1]  # la última fila puede haber quedado cortada
    elif limit is None:
        counter.finish()
    return frame, counter


def _row_warnings(counter, n_columns, skip_records, n_rows):
    if counter is None or not counter.reliable:
        return []
    notes = []
    first = (counter.first_bad_record or 0) + 1
    if counter.long:
        notes.append(f"{_count(counter.long, 'fila tiene', 'filas tienen')} más de "
                     f"{n_columns} campos: se ignoran los sobrantes (primera en la "
                     f"línea {first}).")
    if counter.short:
        notes.append(f"{_count(counter.short, 'fila tiene', 'filas tienen')} menos de "
                     f"{n_columns} campos: lo que falta queda como valor perdido "
                     f"(primera en la línea {first}).")
    return notes


def _count(number, singular, plural):
    return f"{format_int(number)} {singular if number == 1 else plural}"


def format_int(number):
    return f"{int(number):,}".replace(",", ".")


# ------------------------------------------------------------------- ARFF ----

_ARFF_ATTRIBUTE = re.compile(r"""^('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)\s+(.*)$""", re.S)
_ARFF_DATA = re.compile(rb"^[ \t]*@data\b[^\n]*\n?", re.I | re.M)
_ARFF_SPARSE_ITEM = re.compile(
    r"""\s*(\d+)\s+('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,}\s]+)\s*(?:,|$)""")


def _arff_unquote(token):
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1].replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    return token


def _arff_date_format(java):
    if not java:
        return "ISO8601"
    java = java.strip().strip("'\"")
    result, position = "", 0
    table = (("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"), ("HH", "%H"),
             ("mm", "%M"), ("ss", "%S"), ("Z", "%z"))
    while position < len(java):
        if java[position] == "'":
            end = java.find("'", position + 1)
            if end < 0:
                break
            result += java[position + 1:end]
            position = end + 1
            continue
        for pattern, code in table:
            if java.startswith(pattern, position):
                result += code
                position += len(pattern)
                break
        else:
            result += java[position]
            position += 1
    return result


def _arff_header(path, encoding_option):
    """Devuelve (atributos, desplazamiento de los datos, codificación, avisos)."""
    source = _Compressed(path)
    try:
        chunk = 1024 * 1024
        head = b""
        while True:
            more = source.read(chunk)
            head += more
            match = _ARFF_DATA.search(head)
            if match or not more:
                break
            if len(head) > 256 * 1024 * 1024:
                break
    finally:
        source.close()
    if not head.strip():
        raise ImportFailure("El archivo ARFF está vacío.")
    if encoding_option == AUTO:
        encoding = detect_encoding(head[:HEAD_BYTES])
    else:
        encoding = encoding_option
        _check_encoding(encoding)
    if match is None:
        raise ImportFailure("ARFF mal formado: falta la sección @data.")
    text = head[:match.start()].decode(encoding, "replace")
    attributes = []
    relation_seen = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if lower.startswith("@relation"):
            relation_seen = True
        elif lower.startswith("@attribute"):
            attributes.append(_arff_attribute(line, number))
        elif lower.startswith("@end"):
            raise ImportFailure(f"ARFF mal formado en la línea {number}: "
                                "los atributos relacionales no están admitidos.")
        else:
            raise ImportFailure(f"ARFF mal formado en la línea {number}: "
                                f"no se entiende «{line[:60]}».")
    if not attributes:
        raise ImportFailure("ARFF mal formado: no declara ningún @attribute.")
    notes = [] if relation_seen else ["El ARFF no declara @relation."]
    return attributes, match.end(), encoding, notes


def _arff_attribute(line, number):
    rest = line[len("@attribute"):].strip()
    match = _ARFF_ATTRIBUTE.match(rest)
    if not match:
        raise ImportFailure(f"ARFF mal formado en la línea {number}: "
                            f"atributo incompleto «{line[:60]}».")
    name = _arff_unquote(match.group(1))
    spec = match.group(2).strip()
    if spec.startswith("{"):
        if not spec.endswith("}"):
            raise ImportFailure(f"ARFF mal formado en la línea {number}: "
                                "falta la llave de cierre de los valores nominales.")
        inner = spec[1:-1]
        quote = "'" if "'" in inner or '"' not in inner else '"'
        values = next(csv.reader([inner], quotechar=quote, skipinitialspace=True,
                                 escapechar="\\"), [])
        values = tuple(value.strip() for value in values)
        return {"name": name, "type": CATEGORICAL, "values": values}
    kind = spec.split(None, 1)[0].lower()
    if kind in ("numeric", "real", "integer"):
        return {"name": name, "type": NUMERIC}
    if kind == "string":
        return {"name": name, "type": TEXT}
    if kind == "date":
        extra = spec.split(None, 1)[1] if len(spec.split(None, 1)) > 1 else ""
        return {"name": name, "type": DATE, "format": _arff_date_format(extra)}
    if kind == "relational":
        raise ImportFailure(f"El atributo «{name}» es relacional y ese tipo de ARFF "
                            "no está admitido.")
    raise ImportFailure(f"ARFF mal formado en la línea {number}: "
                        f"tipo desconocido «{kind}».")


def _arff_first_data_line(path, offset, encoding):
    head, _ = read_head(path, offset + 65536)
    text = head[offset:].decode(encoding, "replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            return stripped
    return ""


def _read_arff_sparse(path, offset, encoding, attributes, limit, progress):
    source = _Compressed(path)
    try:
        source.read(offset)
        body = io.TextIOWrapper(source.stream, encoding=encoding, errors="replace")
        rows = []
        for number, line in enumerate(body, 1):
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not (line.startswith("{") and line.endswith("}")):
                raise ImportFailure(
                    f"ARFF mal formado: la fila {len(rows) + 1} de datos no sigue "
                    "el formato disperso {índice valor, ...}.")
            rows.append(line[1:-1])
            if limit and len(rows) >= limit:
                break
            if number % CANCEL_EVERY == 0:
                progress.check()
                progress.update(30.0 * source.tell() / max(source.size, 1))
        body.detach()
    finally:
        source.close()
    n_rows = len(rows)
    n_columns = len(attributes)
    columns = []
    for attribute in attributes:
        if attribute["type"] == NUMERIC:
            columns.append(np.zeros(n_rows))
        else:
            default = (attribute["values"][0] if attribute.get("values") else None)
            columns.append(np.full(n_rows, default, dtype=object))
    invalid = 0
    for row, text in enumerate(rows):
        if row % CANCEL_EVERY == 0:
            progress.check()
        for match in _ARFF_SPARSE_ITEM.finditer(text):
            index = int(match.group(1))
            if index >= n_columns:
                invalid += 1
                continue
            token = _arff_unquote(match.group(2))
            if attributes[index]["type"] == NUMERIC:
                try:
                    columns[index][row] = np.nan if token == "?" else float(token)
                except ValueError:
                    columns[index][row] = np.nan
                    invalid += 1
            else:
                columns[index][row] = None if token == "?" else token
    frame = pd.DataFrame({f"c{i}": column for i, column in enumerate(columns)})
    notes = []
    if invalid:
        notes.append(f"{_count(invalid, 'valor del formato disperso no es válido', 'valores del formato disperso no son válidos')}: "
                     "quedan como perdidos o se ignoran.")
    return frame, notes


def _read_arff(path, options, display_name, limit, progress, head_only):
    attributes, offset, encoding, notes = _arff_header(path, options.encoding)
    names = [attribute["name"] for attribute in attributes]
    declared = {}
    for index, attribute in enumerate(attributes):
        item = {"type": attribute["type"]}
        if attribute.get("values") is not None:
            item["values"] = attribute["values"]
        if attribute.get("format"):
            item["date_format"] = attribute["format"]
        declared[index] = item
    last_nominal = attributes[-1]["type"] == CATEGORICAL
    named = [i for i, name in enumerate(names) if name.strip().lower() in CLASS_NAMES]
    target = named[0] if named else (len(attributes) - 1 if last_nominal else None)
    if target is not None and attributes[target]["type"] != TEXT:
        declared[target]["role"] = CLASS

    first = _arff_first_data_line(path, offset, encoding)
    if not first:
        raise ImportFailure("El ARFF no tiene filas en la sección @data.")
    counter = None
    if first.startswith("{"):
        frame, more = _read_arff_sparse(path, offset, encoding, attributes, limit, progress)
        notes += more
    else:
        quote = '"' if '"' in first and "'" not in first else "'"
        layout = Layout(encoding, ",", quote, ".", None, 0, len(attributes), names,
                        declared, notes, body_offset=offset, comment="%",
                        escapechar="\\", skip_initial_space=True)
        arff_options = Options(na_values="?")
        frame, counter = _read_with_c_engine(path, arff_options, layout, limit,
                                             progress, head_only)
        notes += _row_warnings(counter, len(attributes), 0, len(frame))
    if not len(frame):
        raise ImportFailure("El ARFF no tiene filas en la sección @data.")
    return frame, names, declared, notes, encoding


# ------------------------------------------------------------------ Excel ----

def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_MERGE = re.compile(rb'<mergeCell\s+ref="([A-Z]+)(\d+):([A-Z]+)(\d+)"')


def _column_number(letters):
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - 64
    return number - 1


def _fill_merged(rows, ranges):
    """Copia el valor de cada celda combinada a todas las que abarca."""
    for first_row, last_row, first_col, last_col in ranges:
        if first_row >= len(rows) or first_col >= len(rows[first_row]):
            continue
        value = rows[first_row][first_col]
        for row_index in range(first_row, min(last_row + 1, len(rows))):
            row = rows[row_index]
            if len(row) <= last_col:
                row.extend([None] * (last_col + 1 - len(row)))
            for col_index in range(first_col, last_col + 1):
                row[col_index] = value


def _xlsx_merged_ranges(path, sheet_index):
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = sorted(name for name in archive.namelist()
                                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
            names = [f"xl/worksheets/sheet{sheet_index + 1}.xml"]
            names = [name for name in names if name in candidates] or candidates[sheet_index:1 + sheet_index]
            if not names:
                return []
            with archive.open(names[0]) as stream:
                content = stream.read()
    except (OSError, zipfile.BadZipFile, IndexError, KeyError):
        return []
    return [(int(top) - 1, int(bottom) - 1, _column_number(left), _column_number(right))
            for left, top, right, bottom in
            ((m.group(1).decode(), m.group(2).decode(), m.group(3).decode(), m.group(4).decode())
             for m in _MERGE.finditer(content))]


def _read_xlsx_rows(path, sheet, limit, progress):
    import openpyxl  # lazy: tarda en importarse y solo se usa con Excel
    try:
        # Se le pasa el archivo abierto: openpyxl rechaza por la extensión del nombre.
        handle = open(path, "rb")  # pylint: disable=consider-using-with
    except OSError as error:
        _raise_open_error(path, error)
    try:
        book = openpyxl.load_workbook(handle, read_only=True, data_only=True)
    except Exception as error:  # pylint: disable=broad-except
        # openpyxl lanza de todo con archivos dañados: es un dato externo.
        handle.close()
        raise ImportFailure(f"No se puede abrir el libro de Excel: {error}") from error
    try:
        names = list(book.sheetnames)
        chosen, index = _choose_sheet(sheet, names)
        worksheet = book[chosen]
        worksheet.reset_dimensions()
        rows = []
        for count, row in enumerate(worksheet.iter_rows(values_only=True), 1):
            rows.append(list(row))
            if limit and count >= limit:
                break
            if count % CANCEL_EVERY == 0:
                progress.check()
                progress.update(min(40.0, 40.0 * count / 1048576))
    except (ImportCancelled, ImportFailure):
        raise
    except Exception as error:  # pylint: disable=broad-except
        raise ImportFailure(f"El libro de Excel está dañado: {error}") from error
    finally:
        book.close()
        handle.close()
    merged = _xlsx_merged_ranges(path, index)
    if merged:
        _fill_merged(rows, merged)
    return rows, names, chosen, len(merged)


def _choose_sheet(sheet, names):
    if not names:
        raise ImportFailure("El libro de Excel no tiene hojas.")
    if sheet and sheet in names:
        return sheet, names.index(sheet)
    return names[0], 0


def _read_xls_rows(path, sheet, limit, progress):
    import xlrd  # lazy: solo se usa con .xls
    try:
        try:
            book = xlrd.open_workbook(path, on_demand=True, formatting_info=True)
        except NotImplementedError:
            book = xlrd.open_workbook(path, on_demand=True)
    except xlrd.biffh.XLRDError as error:
        text = str(error).lower()
        if "encrypted" in text or "workbook in ole2" in text:
            raise ImportFailure("El libro de Excel está protegido con contraseña o "
                                "no es un Excel válido. Quita la contraseña y "
                                "guárdalo de nuevo.") from error
        raise ImportFailure(f"No se puede abrir el libro de Excel: {error}") from error
    except Exception as error:  # pylint: disable=broad-except
        # xlrd lanza CompDocError, AssertionError, etc. con archivos dañados.
        raise ImportFailure(f"El libro de Excel está dañado: {error}") from error
    try:
        names = list(book.sheet_names())
        chosen, index = _choose_sheet(sheet, names)
        worksheet = book.sheet_by_index(index)
        n_rows = min(worksheet.nrows, limit) if limit else worksheet.nrows
        columns = []
        for col in range(worksheet.ncols):
            progress.check()
            kinds = worksheet.col_types(col, 0, n_rows)
            values = worksheet.col_values(col, 0, n_rows)
            cleaned = []
            for kind, value in zip(kinds, values):
                if kind in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR):
                    cleaned.append(None)
                elif kind == xlrd.XL_CELL_DATE:
                    try:
                        cleaned.append(xlrd.xldate_as_datetime(value, book.datemode))
                    except (ValueError, OverflowError, xlrd.xldate.XLDateError):
                        cleaned.append(None)
                elif kind == xlrd.XL_CELL_BOOLEAN:
                    cleaned.append(bool(value))
                else:
                    cleaned.append(value)
            columns.append(cleaned)
        rows = [list(row) for row in zip(*columns)] if columns else []
        merged = []
        for top, bottom, left, right in getattr(worksheet, "merged_cells", []):
            merged.append((top, bottom - 1, left, right - 1))
    finally:
        book.release_resources()
    if merged:
        _fill_merged(rows, merged)
    return rows, names, chosen, len(merged)


def _read_excel(path, fmt, options, limit, progress):
    reader = _read_xlsx_rows if fmt == "xlsx" else _read_xls_rows
    # Se pide de más para que la cabecera y las filas saltadas no resten filas.
    want = (limit + options.skip_rows + 16 + max(options.header, 0)) if limit else None
    rows, sheets, chosen, merged = reader(path, options.sheet, want, progress)
    notes = []
    has_data = any(any(cell not in (None, "") for cell in row) for row in rows)
    if not has_data:
        if options.sheet:
            others = ", ".join(f"«{name}»" for name in sheets if name != chosen)
            hint = f" Elige otra: {others}." if others else ""
            raise ImportFailure(f"La hoja «{chosen}» está vacía.{hint}")
        for sheet in sheets:
            if sheet == chosen:
                continue
            rows, sheets, other, merged = reader(path, sheet, want, progress)
            if any(any(cell not in (None, "") for cell in row) for row in rows):
                notes.append(f"La hoja «{chosen}» está vacía: se lee «{other}».")
                chosen = other
                has_data = True
                break
        if not has_data:
            raise ImportFailure("Todas las hojas del libro de Excel están vacías.")
    if merged:
        notes.append(f"La hoja tiene {_count(merged, 'celda combinada', 'celdas combinadas')}: "
                     "su valor se ha copiado a todas las celdas que abarca.")
    records = [[_cell_text(cell) for cell in row] for row in rows[:MAX_HEAD_RECORDS]]
    excel_options = Options(skip_rows=options.skip_rows, header=options.header,
                            comment=options.comment)
    found = _resolve_header(records, excel_options, notes)
    if found is None:
        raise ImportFailure("La hoja no tiene filas con datos con las opciones actuales.")
    skip_records, names, declared, header_row = found
    body = rows[skip_records:]
    if limit:
        body = body[:limit]
    if not body:
        raise ImportFailure("La hoja tiene cabecera pero ninguna fila de datos.")
    width = max(len(row) for row in body)
    n_columns = max(width, len(names))
    frame = pd.DataFrame(body, columns=[f"c{i}" for i in range(width)], dtype=object)
    for index in range(width, n_columns):
        frame[f"c{index}"] = None
    frame = frame.astype(object)
    markers = list(set(parse_na_values(options.na_values)) | EXCEL_ERRORS)
    for column in frame.columns:
        series = frame[column]
        frame[column] = series.where(~series.isin(markers))
    frame = frame.dropna(how="all").reset_index(drop=True)
    if not len(frame):
        raise ImportFailure("La hoja tiene cabecera pero ninguna fila de datos.")
    frame = frame.infer_objects()
    names = list(names) + [""] * (n_columns - len(names))
    return frame, names, declared, notes, sheets, chosen, header_row


# --------------------------------------------------------------- inferencia ----

_DATE_SHAPE = re.compile(r"\d{1,4}[-/.]\d{1,2}|\d{1,2}[-/.][A-Za-z]{3}")
_TIME_SUFFIXES = ("", " %H:%M", " %H:%M:%S")
_DAY_FIRST = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y")
_MONTH_FIRST = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y")
_YEAR_FIRST = ("%Y/%m/%d", "%Y.%m.%d")


def _date_candidates(order):
    bases = {AUTO: _DAY_FIRST + _MONTH_FIRST, "dmy": _DAY_FIRST, "mdy": _MONTH_FIRST}
    candidates = ["ISO8601"]
    for base in _YEAR_FIRST + bases.get(order, bases[AUTO]):
        candidates.extend(base + suffix for suffix in _TIME_SUFFIXES)
    return candidates


def _parse_dates(series, date_format):
    return pd.to_datetime(series, format=date_format, errors="coerce", utc=True)


def _guess_date_format(sample, order):
    """Devuelve el formato que mejor encaja con la muestra o None."""
    if not len(sample):
        return None
    texts = sample.astype(str)
    if texts.str.contains(_DATE_SHAPE).mean() < 0.9:
        return None
    best, best_score = None, 0.0
    for candidate in _date_candidates(order):
        score = _parse_dates(texts, candidate).notna().mean()
        if score > best_score + 1e-9:
            best, best_score = candidate, score
        if score == 1.0:
            break
    return best if best_score >= 0.9 else None


def _normalise_numbers(series, decimal, thousands):
    texts = series.astype(str).str.strip()
    if thousands:
        texts = texts.str.replace(thousands, "", regex=False)
    if decimal != ".":
        texts = texts.str.replace(decimal, ".", regex=False)
    return texts


_HUGE_EXPONENT = r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+"


def _coerce_numbers(texts):
    """to_numeric que no pierde los números que desbordan (1e999 es infinito)."""
    parsed = pd.to_numeric(texts, errors="coerce")
    failed = parsed.isna() & texts.notna()
    if failed.any():
        pending = texts[failed].astype(str).str.strip()
        huge = pending[pending.str.fullmatch(_HUGE_EXPONENT)]
        if len(huge):
            parsed = parsed.astype(float)
            parsed.loc[huge.index] = huge.map(float)
    return parsed


def _sample(series, size=2000):
    values = series.dropna()
    if len(values) <= size:
        return values
    step = len(values) / (size / 2)
    spaced = values.iloc[(np.arange(size // 2) * step).astype(int)]
    return pd.concat((values.iloc[:size // 2], spaced))


def _infer_object_column(series, loaded_context):
    """Devuelve (tipo, formato de fecha) de una columna de texto o mezclada."""
    decimal, thousands, order = loaded_context
    sample = _sample(series)
    if len(sample) == 0:
        return NUMERIC, None
    if sample.map(lambda value: isinstance(value, (pd.Timestamp, np.datetime64)) or
                  hasattr(value, "isoformat")).mean() >= 0.9:
        return DATE, "ISO8601"
    numeric = _coerce_numbers(_normalise_numbers(sample, decimal, thousands))
    if numeric.notna().mean() >= 0.95:
        return NUMERIC, None
    found = _guess_date_format(sample, order)
    if found is not None:
        return DATE, found
    uniques = series.nunique(dropna=True)
    limit = min(MAX_AUTO_CATEGORIES, int(round(max(len(series), 1) ** 0.7)))
    long_text = sample.astype(str).str.len().mean() > 50
    return (CATEGORICAL if uniques <= limit and not long_text else TEXT), None


def build_columns(frame, names, declared, decimal, thousands, order):
    """Crea un ColumnMeta por columna a partir de los datos y lo declarado."""
    final, _, _ = unique_names(names)
    context = (decimal, thousands or None, order)
    columns = []
    for index, column in enumerate(frame.columns):
        series = frame[column]
        declared_item = declared.get(index, {})
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        date_format = declared_item.get("date_format")
        if declared_item.get("type"):
            kind = declared_item["type"]
            if kind == DATE and date_format is None:
                _, date_format = _infer_object_column(series, context)
                date_format = date_format or "ISO8601"
        elif pd.api.types.is_bool_dtype(series.dtype):
            kind = CATEGORICAL
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            kind, date_format = DATE, "ISO8601"
        elif pd.api.types.is_numeric_dtype(series.dtype):
            integral = n_unique <= 2 and n_unique > 0 and _is_integral(series)
            kind = CATEGORICAL if integral else NUMERIC
        else:
            kind, date_format = _infer_object_column(series, context)
        role = declared_item.get("role", ATTRIBUTE)
        if role == "weight":
            role = META
        if n_missing == len(series) and "role" not in declared_item:
            role = SKIP
        elif kind == TEXT and role in (ATTRIBUTE, CLASS):
            role = META
        samples = [_display(value) for value in series.dropna().iloc[:3]]
        columns.append(ColumnMeta(
            index=index, source=str(names[index]), name=final[index], type=kind,
            role=role, n_missing=n_missing, n_unique=n_unique, samples=samples,
            declared_values=declared_item.get("values"), date_format=date_format))
    if not any(column.role == CLASS for column in columns) and not declared:
        for column in columns:
            if column.name.strip().lower() in CLASS_NAMES and column.type != TEXT \
                    and column.role == ATTRIBUTE:
                column.role = CLASS
                break
    return columns


def _is_integral(series):
    values = series.dropna().to_numpy(dtype=float)
    return bool(len(values)) and bool(np.all(np.isfinite(values) & (values == np.floor(values))))


def _display(value):
    if isinstance(value, float):
        return format_number(value)
    return str(value)


def format_number(value):
    if value != value:  # NaN
        return ""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


def _structure_notes(names, columns):
    notes = []
    _, empty, duplicated = unique_names(names)
    if empty and any(str(name).strip() for name in names):
        notes.append(f"{_count(empty, 'columna no tiene', 'columnas no tienen')} nombre: "
                     "se les ha asignado uno.")
    if duplicated:
        notes.append(f"{_count(duplicated, 'nombre de columna está repetido', 'nombres de columna están repetidos')}: "
                     "se han renombrado.")
    blank = [column for column in columns if column.n_unique == 0 and column.role == SKIP]
    if blank:
        notes.append(f"{_count(len(blank), 'columna está vacía y se omite', 'columnas están vacías y se omiten')}: "
                     "puedes recuperarlas en la tabla de columnas.")
    return notes


# ----------------------------------------------------------------- carga ----

def _load_delimited(path, options, fmt, limit, progress, head_only):
    layout = analyse_delimited(path, options, fmt)
    automatic_encoding = options.encoding == AUTO
    try:
        frame, counter = _read_with_c_engine(path, options, layout, limit, progress, head_only)
    except UnicodeDecodeError as error:
        if not automatic_encoding or layout.encoding not in ("utf-8", "utf-8-sig"):
            raise ImportFailure(
                f"El archivo no está en {layout.encoding}: hay un byte no válido en "
                f"la posición {format_int(error.start)} del bloque leído. Elige otra "
                "codificación en la importación avanzada.") from error
        layout.encoding = "cp1252"
        layout.notes.append("El archivo no es UTF-8 de principio a fin: se ha leído "
                            "como Windows-1252. Si ves caracteres raros, elige la "
                            "codificación correcta.")
        frame, counter = _read_with_c_engine(path, options, layout, limit, progress, head_only)
    return frame, layout, counter


def load(path, options, display_name, limit=None, head_only=False, progress=None):
    """Lee un archivo local. `limit` y `head_only` son para la vista previa."""
    progress = progress or Progress()
    options = Options.from_dict(options.to_dict())
    fmt, notes = detect_format(path, display_name, options.format)
    effective_limit = limit or None
    if options.max_rows:
        effective_limit = min(effective_limit or options.max_rows, options.max_rows)
    name = os.path.splitext(os.path.basename(display_name))[0]
    for suffix in (".gz", ".bz2", ".xz", ".zip"):
        name = name[:-len(suffix)] if name.lower().endswith(suffix) else name
    extra = {}
    try:
        if fmt in DELIMITED:
            frame, layout, counter = _load_delimited(
                path, options, fmt, effective_limit, progress, head_only)
            names, declared = layout.names, layout.declared
            notes += layout.notes
            if not head_only:
                notes += _row_warnings(counter, layout.n_columns, layout.skip_records,
                                       len(frame))
            decimal, thousands = layout.decimal, layout.thousands
            header_row = layout.header_row
            extra = dict(encoding=layout.encoding, delimiter=layout.delimiter)
            if any(item.get("role") == "weight" for item in declared.values()):
                notes.append("La columna de pesos de la cabecera de Orange se importa "
                             "como meta.")
        elif fmt == "arff":
            frame, names, declared, more, encoding = _read_arff(
                path, options, display_name, effective_limit, progress, head_only)
            notes += more
            decimal, thousands, header_row = ".", None, 1
            extra = dict(encoding=encoding)
        else:
            frame, names, declared, more, sheets, sheet, header_row = _read_excel(
                path, fmt, options, effective_limit, progress)
            notes += more
            decimal, thousands = ".", None
            extra = dict(sheets=sheets, sheet=sheet)
    except MemoryError as error:
        raise ImportFailure("No hay memoria suficiente para leer el archivo. Limita "
                            "el número de filas en la importación avanzada.") from error
    progress.check()
    if not len(frame):
        if head_only:
            frame = frame.iloc[:0]
        raise ImportFailure("El archivo tiene cabecera pero ninguna fila de datos.")
    progress.update(65.0)
    columns = build_columns(frame, names, declared, decimal, thousands, options.date_order)
    notes += _structure_notes(names, columns)
    limited = bool(options.max_rows) and len(frame) >= options.max_rows and not head_only
    if limited:
        notes.append(f"Se han leído solo las primeras {format_int(options.max_rows)} "
                     "filas, como indica la opción de máximo de filas.")
    return Loaded(frame=frame, columns=columns, format=fmt, notes=_dedupe(notes), name=name,
                  decimal=decimal, thousands=thousands or "", header_row=header_row,
                  complete=not head_only, truncated_by_limit=limited, **extra)


def _dedupe(notes):
    return list(dict.fromkeys(notes))


# ------------------------------------------------------------ a tabla Orange ----

@dataclass
class ColumnChoice:
    name: str
    type: str
    role: str


def default_choices(columns):
    return [ColumnChoice(column.name, column.type, column.role) for column in columns]


def _to_numeric(series, decimal, thousands):
    """Devuelve (array de float, nº de valores no convertibles, nº de infinitos)."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(float).to_numpy(), 0, 0
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = series.to_numpy(dtype=float, na_value=np.nan)
        bad = 0
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        values = _seconds(series)
        bad = 0
    else:
        known = series.notna()
        if _sample(series).map(type).eq(str).any():
            parsed = _coerce_numbers(_normalise_numbers(series.where(known, ""), decimal,
                                                        thousands))
        else:
            parsed = pd.to_numeric(series, errors="coerce")
        values = parsed.to_numpy(dtype=float, na_value=np.nan)
        bad = int((known.to_numpy() & np.isnan(values)).sum())
    infinite = np.isinf(values)
    n_infinite = int(infinite.sum())
    if n_infinite:
        values = values.copy()
        values[infinite] = np.nan
    return values, bad, n_infinite


def _seconds(parsed):
    parsed = pd.to_datetime(parsed, errors="coerce", utc=True)
    delta = parsed - pd.Timestamp("1970-01-01", tz="UTC")
    return (delta / pd.Timedelta(seconds=1)).to_numpy(dtype=float, na_value=np.nan)


def _to_dates(series, meta):
    """Devuelve (segundos POSIX, nº no convertibles, tiene fecha, tiene hora)."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    elif _sample(series).map(lambda v: hasattr(v, "isoformat")).any():
        parsed = pd.to_datetime(series.map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v),
                                format="ISO8601", errors="coerce", utc=True)
    else:
        date_format = meta.date_format
        if date_format is None:
            date_format = _guess_date_format(_sample(series), AUTO) or "ISO8601"
        parsed = _parse_dates(series.astype(object).where(series.notna()), date_format)
    values = _seconds(parsed)
    bad = int((series.notna().to_numpy() & np.isnan(values)).sum())
    known = values[~np.isnan(values)]
    has_time = bool(len(known)) and bool(np.any(np.mod(known, 86400) != 0))
    return values, bad, True, has_time


def _to_categorical(series, meta):
    """Devuelve (códigos en float con NaN, valores)."""
    if pd.api.types.is_bool_dtype(series.dtype):
        codes, uniques = pd.factorize(series.astype(object), use_na_sentinel=True)
    else:
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
    labels = [format_number(u) if isinstance(u, (float, np.floating)) else
              (str(int(u)) if isinstance(u, (int, np.integer)) and not isinstance(u, bool) else str(u))
              for u in uniques]
    declared = list(meta.declared_values or ())
    extra = [label for label in natural_sorted(set(labels)) if label not in set(declared)]
    values = declared + extra
    position = {label: index for index, label in enumerate(values)}
    mapping = np.array([position[label] for label in labels], dtype=float)
    result = np.full(len(series), np.nan)
    present = codes >= 0
    if len(mapping):
        result[present] = mapping[codes[present]]
    return result, tuple(values), len(extra) if declared else 0


def _to_text(series):
    if pd.api.types.is_numeric_dtype(series.dtype):
        out = series.map(format_number, na_action="ignore")
    else:
        out = series
    return out.astype(object).where(series.notna(), "").map(str).to_numpy(dtype=object)


def frame_to_table(loaded, choices, progress=None):
    """Convierte lo leído en una Table. Devuelve (tabla, avisos)."""
    progress = progress or Progress()
    frame = loaded.frame
    n_rows = len(frame)
    used = [(meta, choice) for meta, choice in zip(loaded.columns, choices)
            if choice.role != SKIP]
    if not used:
        raise ImportFailure("Todas las columnas están omitidas: indica al menos una "
                            "con otro rol.")
    names, _, _ = unique_names([choice.name for _, choice in used])
    attributes, classes, metas = [], [], []
    x_data, y_data, m_data = [], [], []
    notes = []
    n_invalid = n_infinite = 0
    invalid_columns = []
    huge_columns = []
    for position, ((meta, choice), name) in enumerate(zip(used, names)):
        progress.check()
        progress.update(70.0 + 30.0 * position / len(used))
        series = frame.iloc[:, meta.index]
        role = choice.role
        if choice.type == NUMERIC:
            values, bad, infinite = _to_numeric(series, loaded.decimal or ".",
                                                loaded.thousands or None)
            n_infinite += infinite
            if bad:
                n_invalid += bad
                invalid_columns.append(name)
            finite = values[np.isfinite(values)]
            if pd.api.types.is_integer_dtype(series.dtype) and len(finite) \
                    and np.abs(finite).max() > SAFE_INTEGER:
                huge_columns.append(name)
            elif series.dtype == object and len(finite) and np.abs(finite).max() > SAFE_INTEGER:
                huge_columns.append(name)
            variable = ContinuousVariable(name)
        elif choice.type == DATE:
            values, bad, has_date, has_time = _to_dates(series, meta)
            if bad:
                n_invalid += bad
                invalid_columns.append(name)
            variable = TimeVariable(name, have_date=int(has_date), have_time=int(has_time))
        elif choice.type == CATEGORICAL:
            values, categories, undeclared = _to_categorical(series, meta)
            if undeclared:
                notes.append(f"La columna «{name}» tiene {_count(undeclared, 'valor', 'valores')} "
                             "que no estaban declarados: se han añadido.")
            variable = DiscreteVariable(name, values=categories)
        else:
            if role in (ATTRIBUTE, CLASS):
                role = META
            variable = StringVariable(name)
            values = _to_text(series)
        if choice.type == TEXT or role == META:
            metas.append(variable)
            m_data.append(values.astype(object) if choice.type != TEXT else values)
        elif role == CLASS:
            classes.append(variable)
            y_data.append(values)
        else:
            attributes.append(variable)
            x_data.append(values)

    domain = Domain(attributes, classes, metas)
    X = np.column_stack(x_data) if x_data else np.empty((n_rows, 0))
    Y = np.column_stack(y_data) if y_data else np.empty((n_rows, 0))
    M = np.column_stack(m_data).astype(object) if m_data else None
    table = Table.from_numpy(domain, X, Y if y_data else None, metas=M)
    table.name = loaded.name
    if n_invalid:
        shown = ", ".join(f"«{name}»" for name in invalid_columns[:3])
        more = "…" if len(invalid_columns) > 3 else ""
        notes.append(f"{_count(n_invalid, 'valor no encaja', 'valores no encajan')} con el "
                     f"tipo de su columna ({shown}{more}): quedan como perdidos.")
    if n_infinite:
        notes.append(f"{_count(n_infinite, 'valor infinito', 'valores infinitos')} "
                     "se han convertido en perdidos.")
    if huge_columns:
        shown = ", ".join(f"«{name}»" for name in huge_columns[:3])
        notes.append(f"Hay números enormes en {shown}: pierden precisión porque Orange "
                     "guarda los numéricos en coma flotante. Si son identificadores, "
                     "cambia el tipo a texto.")
    return table, _dedupe(notes)


def preview_column(loaded, meta, choice, count=PREVIEW_ROWS):
    """Primeros valores de una columna como texto, tal como quedarán en la tabla."""
    series = loaded.frame.iloc[:count, meta.index]
    if choice.type == NUMERIC:
        values, _, _ = _to_numeric(series, loaded.decimal or ".", loaded.thousands or None)
        return ["" if v != v else format_number(v) for v in values]
    if choice.type == DATE:
        values, _, _, has_time = _to_dates(series, meta)
        stamp = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
        return ["" if v != v else time.strftime(stamp, time.gmtime(v))
                if -2e9 < v < 3e10 else format_number(v) for v in values]
    if choice.type == CATEGORICAL:
        values, categories, _ = _to_categorical(series, meta)
        return ["" if v != v else categories[int(v)] for v in values]
    return list(_to_text(series))


def preview_columns(loaded, choices):
    return [preview_column(loaded, meta, choice)
            for meta, choice in zip(loaded.columns, choices)]


# -------------------------------------------------------------------- URL ----

class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlparse(newurl).scheme.lower()
        if scheme not in ("http", "https"):
            raise ImportFailure(f"La URL redirige a un destino no permitido ({scheme}://).")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class Downloaded:
    path: str
    display_name: str
    final_url: str
    size: int
    notes: List[str] = field(default_factory=list)


def normalise_url(url):
    url = url.strip()
    if not url:
        raise ImportFailure("Escribe una URL.")
    if "://" not in url:
        url = "https://" + url
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise ImportFailure(f"Solo se admiten URL http y https, no {parts.scheme}://.")
    if not parts.netloc:
        raise ImportFailure("La URL no es válida: falta el servidor.")
    path = urllib.parse.quote(parts.path, safe="/%:@+,;=~-._!$&'()*")
    query = urllib.parse.quote(parts.query, safe="=&%+,;:@/?-._~!$'()*")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, ""))


def rewrite_share_links(url):
    """Convierte enlaces de Google Sheets, Drive y Dropbox en descargas directas."""
    try:
        from Orange.data.io import UrlReader
        return UrlReader._trim(url)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        return url


def _download_name(response, final_url):
    disposition = response.headers.get("content-disposition") or ""
    matches = re.findall(r"filename\*?=(?:\"|.{0,10}?'[^']*')?([^\";]+)", disposition)
    if matches:
        return os.path.basename(urllib.parse.unquote(matches[-1]).strip())
    return os.path.basename(urllib.parse.urlsplit(final_url).path) or "datos"


def download(url, directory, max_megabytes=DEFAULT_MAX_DOWNLOAD_MB, progress=None,
             timeout=DOWNLOAD_TIMEOUT, deadline=DOWNLOAD_DEADLINE):
    """Descarga a un temporal, con límites de tiempo y de tamaño."""
    progress = progress or Progress()
    url = rewrite_share_links(normalise_url(url))
    limit = max(int(max_megabytes), 1) * 1024 * 1024
    opener = urllib.request.build_opener(_SafeRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    started = time.monotonic()
    target = None
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            name = _download_name(response, final_url)
            announced = response.headers.get("content-length")
            announced = int(announced) if announced and announced.isdigit() else 0
            if announced > limit:
                raise ImportFailure(
                    f"La descarga ocupa {format_int(announced // (1024 * 1024))} MB y el "
                    f"límite es {format_int(max_megabytes)} MB. Súbelo en la importación "
                    "avanzada si es lo que quieres.")
            content_type = (response.headers.get("content-type") or "").lower()
            suffix = re.sub(r"[^a-z0-9.]", "", _extension(name) or ".dat")[:12] or ".dat"
            descriptor, target = tempfile.mkstemp(suffix=suffix, prefix="url-", dir=directory)
            received = 0
            first = b""
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    progress.check()
                    if time.monotonic() - started > deadline:
                        raise ImportFailure(
                            f"La descarga ha superado el tiempo máximo de {deadline // 60} minutos.")
                    block = response.read(DOWNLOAD_BLOCK)
                    if not block:
                        break
                    if not first:
                        first = block
                        if "html" in content_type or _looks_like_html(block):
                            raise ImportFailure(
                                "La URL ha devuelto una página web (HTML) en lugar de "
                                "datos. Comprueba que apunte directamente al archivo.")
                    received += len(block)
                    if received > limit:
                        raise ImportFailure(
                            f"La descarga supera el límite de {format_int(max_megabytes)} MB. "
                            "Súbelo en la importación avanzada si es lo que quieres.")
                    output.write(block)
                    if announced:
                        progress.update(100.0 * received / announced)
            if announced and received < announced:
                raise ImportFailure("La descarga se ha cortado antes de terminar. "
                                    "Vuelve a intentarlo.")
            if not received:
                raise ImportFailure("La URL no ha devuelto ningún dato.")
            notes = []
            if final_url != url:
                notes.append(f"La URL redirige a {final_url}.")
            result = Downloaded(target, name, final_url, received, notes)
            target = None
            return result
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            if "not allowed" in str(error.reason):
                raise ImportFailure("La URL redirige a un destino no permitido.") from error
            raise ImportFailure(f"La URL redirige demasiadas veces (máximo {MAX_REDIRECTS}).") from error
        raise ImportFailure(f"El servidor ha respondido con el error {error.code} "
                            f"({error.reason}).") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, ImportFailure):
            raise error.reason from error
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise ImportFailure(f"El servidor no responde: se ha agotado el tiempo de "
                                f"espera de {timeout} s.") from error
        raise ImportFailure(f"No se puede conectar con el servidor: {error.reason}") from error
    except (socket.timeout, TimeoutError) as error:
        raise ImportFailure(f"El servidor no responde: se ha agotado el tiempo de espera "
                            f"de {timeout} s.") from error
    except (ValueError, OSError, http.client.HTTPException) as error:
        raise ImportFailure(f"No se puede descargar la URL: {error}") from error
    finally:
        if target is not None:
            try:
                os.remove(target)
            except OSError:
                pass
