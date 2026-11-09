# orange_gualbe_toolbox

Utilidades visuales de ciencia de datos para Orange Data Mining.

Add-on para [Orange Data Mining](https://orangedatamining.com/) con la categoría
de widgets **Toolbox Gualbe**.

## Instalación

Desde Orange: menú **Options → Add-ons…**, botón **Add more…**, escribe
`orange_gualbe_toolbox` y reinicia Orange.

O con `pip`, en el mismo entorno de Python que Orange:

```bash
pip install orange_gualbe_toolbox
```

## Widgets

Todos los widgets son visuales, con valores por defecto sensatos y su ayuda en
español (F1 o icono **?**).

| Widget | Qué hace |
|---|---|
| **SMOTE** | Balancea las clases generando instancias sintéticas de las minoritarias. Muestra la distribución antes y después. |
| **File Import** | Importa datos desde un archivo o una URL (CSV, TSV, TAB, XLS/XLSX y ARFF) con vista previa y edición de columnas. |
| **Test & Score Pro** | Evalúa modelos como Test & Score y los compara con tests estadísticos rigurosos, gráficos e informe exportable. |

### SMOTE

<p align="center">
  <img src="https://raw.githubusercontent.com/gualbe/orange_gualbe_toolbox/main/assets/captura_smote.png" alt="Widget SMOTE: barras por clase antes y después del balanceo" width="760">
</p>

Recibe una tabla con variable objetivo categórica y devuelve la misma tabla, con
sus filas originales intactas, más las instancias sintéticas al final.

- **Elige el algoritmo solo**: `SMOTE` con atributos numéricos, `SMOTENC` con
  atributos mixtos y `SMOTEN` si todos son categóricos.
- **Estrategias**: igualar todas las clases a la mayoritaria, solo la minoritaria,
  todas menos la minoritaria o llegar a un porcentaje de la mayoritaria.
- **Reproducible**: número de vecinos *k* y semilla aleatoria configurables.
- En las filas sintéticas las metas quedan desconocidas y, si la tabla tiene
  pesos, el peso es 1.
- Trabaja en segundo plano y explica en el propio widget qué hacer si faltan
  valores, si una clase tiene pocas instancias o si no hay nada que balancear.

Balancea solo los datos de entrenamiento: las filas sintéticas en el conjunto de
prueba darían una evaluación demasiado optimista.

### File Import

<p align="center">
  <img src="https://raw.githubusercontent.com/gualbe/orange_gualbe_toolbox/main/assets/captura_fileimport.png" alt="Widget File Import: resumen de lo cargado, tabla de columnas y vista previa de filas" width="760">
</p>

Reúne en un solo widget lo mejor de **File** y **CSV File Import** de Orange.

- **Origen**: archivo local (con recientes y arrastrar y soltar) o URL http/https.
- **Formatos**: CSV, TSV, TAB de Orange, XLS, XLSX y ARFF, también comprimidos. El
  formato, la codificación, el separador y la cabecera se detectan solos.
- **Vista previa** al instante, leyendo solo el principio del archivo.
- **Columnas**: nombre, tipo y rol editables, también varias a la vez, sin volver a
  leer el archivo.
- **Importación avanzada** plegada: codificación, separador, comillas, decimales,
  miles, filas a saltar, cabecera, valores perdidos, comentarios y máximo de filas.
- **Rápido y robusto**: carga en segundo plano y cancelable con el motor C de
  pandas, y mensajes claros ante archivos vacíos o dañados, filas irregulares, Excel
  con contraseña, URL con errores o que devuelven una página web, etc.

Las opciones y las ediciones se guardan con el workflow.

### Test & Score Pro

<p align="center">
  <img src="https://raw.githubusercontent.com/gualbe/orange_gualbe_toolbox/main/assets/captura_testscorepro.png" alt="Widget Test &amp; Score Pro: recomendación del mejor modelo y tabla de puntuaciones" width="760">
</p>

Parte de **Test & Score** de Orange, con sus mismas entradas, muestreos y salidas, y responde
a la pregunta clave: ¿la diferencia entre modelos es real o se debe al azar?

- **Tests rigurosos**: t corregido de Nadeau-Bengio y t bayesiano correlacionado (las particiones
  de una validación cruzada no son independientes), Wilcoxon, Friedman con Nemenyi y Holm, McNemar,
  corrección por comparaciones múltiples, tamaños de efecto e intervalos de confianza.
- **Cada test dice en qué se apoya** y avisa si no se cumple.
- **Gráficos**: distribución por particiones, diferencias críticas, matriz de comparaciones y
  probabilidades bayesianas, exportables como imagen.
- **Recomendación** del mejor modelo, justificada, en lenguaje llano.
- **Salidas adicionales**: puntuaciones por modelo y por partición, comparaciones, rangos y mejor
  modelo.
- **Informe** autocontenido en HTML o PDF, reproducible.
- En segundo plano y cancelable. Cambiar la métrica o los tests no vuelve a entrenar.

## Ayuda de los widgets

La ayuda (F1 / icono "?") se genera con Sphinx dentro del propio paquete y debe
reconstruirse antes de empaquetar:

```bash
pip install sphinx
python -m sphinx -b html doc orangecontrib/orange_gualbe_toolbox/_help/html
```

## Mantenimiento y publicación en PyPI

Este repositorio es la **fuente de verdad** del add-on. La publicación en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/) es automática mediante
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): al publicar una
release `vX.Y.Z` en GitHub, el workflow `.github/workflows/publish.yml` genera la
ayuda, construye el paquete, lo sube a PyPI y adjunta los ficheros a la release.
El workflow rechaza la publicación si la etiqueta no coincide con la versión de
`pyproject.toml`.

## Licencia

[GPL-3.0-or-later](https://github.com/gualbe/orange_gualbe_toolbox/blob/main/LICENSE).
