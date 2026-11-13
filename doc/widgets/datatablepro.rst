.. _data table pro:

Data Table Pro
==============

Muestra varias tablas a la vez, cada una en su pestaña, y permite cambiarlas como en una
hoja de cálculo: ordenar filas, reordenar columnas, editar celdas, buscar y reemplazar,
añadir prefijos y sufijos, renombrar columnas y cambiar su tipo y su rol. Cada tabla, con
todos sus cambios, sale por su propia salida numerada.

.. image:: ../../assets/captura_datatablepro.png
   :alt: Data Table Pro con tres tablas en pestañas, celdas editadas, barras y coincidencias resaltadas
   :width: 100%

Entradas
--------

- **Data**: una o varias tablas. Conecta tantas como quieras: cada enlace abre una pestaña.

Salidas numeradas
-----------------

Orange no permite salidas con un nombre que cambie según lo conectado, así que el widget
tiene **cinco pares de salidas numeradas**, en el orden de las pestañas:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Salida
     - Qué lleva
   * - **Data 1** … **Data 5**
     - La tabla completa de la pestaña 1 … 5, con todos los cambios y en el orden de filas y
       columnas que se ve. El filtro rápido no la recorta.
   * - **Selected Data 1** … **Selected Data 5**
     - Las filas seleccionadas en esa pestaña, con los mismos cambios. Sin selección no envía
       nada, como el Data Table de Orange.

El número de cada pestaña aparece en un distintivo: **①**, **②**… La pestaña ① va a
*Data 1* y *Selected Data 1*, la ② a *Data 2* y *Selected Data 2*, etc. El panel *Tabla*
lo repite para la pestaña abierta y el ratón sobre la pestaña también lo dice.

Cada tabla de salida se llama como el **nodo del que viene** (por ejemplo «Flores» si el
nodo *File* se ha renombrado así) o, si no se puede saber, como la tabla de entrada. Así es
fácil reconocerla después en otros widgets.

- Con **más de cinco** entradas, las que pasan de cinco se ven y se pueden editar, pero no
  tienen salida: el widget lo avisa y su pestaña lleva «—» en lugar de número.
- Al **quitar un enlace**, las pestañas siguientes suben un puesto y sus salidas también; la
  última salida que queda libre se vacía.

**Ejemplo.** Conecta dos *File* (renombrados «Clientes» y «Pedidos») a Data Table Pro.
Corrige en «Clientes» los códigos postales mal escritos con *Buscar y reemplazar* y conecta
*Data 1* a *Merge Data* y *Data 2* a su segunda entrada: el cruce usa ya los datos corregidos.

Ordenar y reordenar
-------------------

- **Ordenar filas**: pulsa el nombre de una columna; ordena de menor a mayor y, si vuelves a
  pulsar, de mayor a menor. La ordenación es estable: para ordenar por dos columnas, pulsa
  primero la secundaria y después la principal. Los valores desconocidos quedan al final.
  **Orden original** vuelve al orden de la entrada (también en el menú de la cabecera).
- **Reordenar columnas**: arrastra su cabecera. Las salidas usan el nuevo orden, sin mezclar
  atributos, objetivos y metas: cada grupo se ordena por separado.

El número a la izquierda de cada fila es el de la fila en la tabla de entrada, así que
siempre se ve de dónde viene.

Editar celdas
-------------

Doble clic (o empieza a escribir) en una celda. El valor se comprueba según el tipo:

- **Numérica**: acepta punto o coma decimal. Un texto que no es un número se rechaza con un
  aviso que dice por qué.
- **Categórica**: un valor nuevo se añade a los valores de la columna.
- **Fecha y hora**: formatos como ``2026-11-13`` o ``2026-11-13 09:30``.
- **Texto**: cualquier valor.

Dejar la celda vacía o escribir ``?`` la deja como valor desconocido. Las celdas cambiadas se
resaltan en amarillo y la pestaña lleva ✎.

**Deshacer y rehacer**: **Ctrl+Z** y **Ctrl+Y** (o los botones del panel *Cambios*) deshacen y
rehacen ediciones, reemplazos, prefijos, cambios de nombre, de tipo y de rol, hasta 50 pasos.

Buscar y reemplazar
-------------------

Escribe el texto y pulsa **Buscar** (o Intro). Todas las coincidencias se resaltan en naranja y
la actual en naranja intenso; **◀** y **▶** (o **Mayús+F3** y **F3**) van a la anterior y a la
siguiente. **Reemplazar todo** cambia todas y dice cuántos reemplazos ha hecho y en cuántas
celdas.

- **Expresión regular**: usa la sintaxis de Python. En el reemplazo, ``\1``, ``\2``… o
  ``\g<nombre>`` insertan los grupos. Por ejemplo, buscar ``^(\w+), (\w+)$`` y reemplazar por
  ``\2 \1`` convierte «García, Ana» en «Ana García».
- **Distinguir mayúsculas**: si no se marca, «madrid» encuentra «Madrid».
- **Ámbito**: toda la tabla, las columnas seleccionadas o las filas seleccionadas.

Se busca en el texto que se ve en cada celda. En una columna numérica o de fecha, un reemplazo
que no da un valor válido se rechaza y el widget dice cuántos. En una categórica, el valor
nuevo ocupa el lugar del antiguo en la lista de valores.

Prefijo y sufijo
----------------

Añade un texto delante, detrás o en los dos lados de cada celda del ámbito (columnas
seleccionadas, filas seleccionadas o toda la tabla). Se aplica a columnas de **texto** y
**categóricas**. Las **numéricas y de fecha no se tocan**, porque «Nº 5» ya no sería un número:
el widget dice cuáles se han saltado. Si quieres cambiarlas, conviértelas antes a texto. Las
celdas vacías tampoco se tocan.

Columnas: nombre, tipo y rol
----------------------------

Selecciona una celda de la columna o la columna entera (menú de la cabecera → *Seleccionar la
columna*). En el panel *Columna*:

- **Nombre**: escribe el nuevo y pulsa *Aplicar* o Intro. No puede repetirse.
- **Tipo**: categórica, numérica, fecha y hora o texto. Si algunos valores no se pueden
  convertir (por ejemplo «tres» a número), pasan a desconocidos y el widget dice cuántos y en
  qué columna. Un número convertido a fecha se toma como segundos desde 1970. Una columna de
  texto siempre es meta, como exige Orange.
- **Rol**: atributo, objetivo o meta. Se admiten **varios objetivos**.

Si hay varias columnas enteras seleccionadas, el tipo y el rol se aplican a todas.

La cabecera de cada columna muestra su tipo con un icono y, al pasar el ratón, sus
estadísticas: desconocidos, mínimo, máximo, media, mediana y desviación típica en las
numéricas y los valores más frecuentes en las categóricas.

Vista
-----

- **Filtrar filas**: muestra solo las filas que contienen el texto en alguna columna. Solo
  cambia lo que se ve; *Data N* sigue llevando todas las filas.
- **Seleccionar filas completas**: si se desmarca, se seleccionan celdas sueltas y *Selected
  Data N* lleva solo las columnas seleccionadas.
- **Barras en las columnas numéricas**: una barra bajo cada valor, proporcional a su posición
  entre el mínimo y el máximo de la columna.
- **Colores de las categóricas**: el color de cada valor, como en los gráficos de Orange.
- **Resaltar celdas editadas**.

Las columnas objetivo se tiñen de azul y las metas de gris. Con el botón derecho sobre la tabla
puedes **copiar** la selección (también Ctrl+C, como texto tabulado que se pega en una hoja de
cálculo) y dar un **color de fondo o de texto** a las filas, columnas o celdas seleccionadas.

Cambios guardados
-----------------

Los cambios de cada pestaña (ediciones, reemplazos, nombres, tipos, roles, orden y colores) se
guardan con el workflow. Al volver a abrirlo, si la tabla de entrada tiene las mismas columnas y
el mismo número de filas, se repiten; si ha cambiado, se descartan con un aviso. **Restablecer**
descarta todos los cambios de la pestaña abierta.

Tablas grandes
--------------

La vista solo lee las celdas que se ven y las operaciones trabajan por columnas enteras. Con
300.000 filas, abrir la tabla tarda menos de medio segundo; ordenar, convertir o reemplazar en
una columna, una fracción de segundo; y buscar en toda la tabla o filtrar, un par de segundos la
primera vez (los textos se guardan y las siguientes son casi inmediatas). Una tabla dispersa se muestra y se emite como densa (el widget
lo avisa) siempre que no supere 50 millones de celdas.
