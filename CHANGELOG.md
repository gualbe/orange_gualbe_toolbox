# Cambios

Cada versión se publica a la vez en GitHub (release `vX.Y.Z`) y en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/).

## 0.5.4 (2026-11-13)

- **Data Table Pro** tiene ahora dos pares de salidas, **Data 1**, **Selected Data 1**, **Data 2** y
  **Selected Data 2**, en lugar de cinco. A partir de la tercera, las pestañas se ven y se editan con
  «—» en lugar de número y sin salida, y el widget lo avisa. Al quitar un enlace, la tercera pestaña
  sube a la salida 2.
- **Compatibilidad**: en los workflows guardados con enlaces desde *Data 3* a *Data 5* o *Selected
  Data 3* a *Selected Data 5*, esos enlaces se pierden al abrirlos y Orange lo avisa. Los cambios
  guardados de todas las pestañas se conservan.

## 0.5.3 (2026-11-13)

Correcciones de **Data Table Pro**:

- Los cambios guardados con la 0.5.1 vuelven a restaurarse: la 0.5.2 tomaba su huella, de otro
  formato, por «datos distintos» y los borraba. Ahora la huella lleva versión y un cambio de formato
  nunca descarta cambios.
- Una pestaña ya cargada, o con cambios propios, nunca recibe los cambios guardados de otra: al
  quitar el enlace de «A», «B» con la misma tabla ya no hereda sus ediciones.
- La huella del contenido se calcula solo cuando hace falta y es más rápida con columnas de texto.

## 0.5.2 (2026-11-13)

Correcciones de **Data Table Pro**:

- Buscar y reemplazar vuelven a trabajar sobre el texto que se ve: en la 0.5.1, en las numéricas
  que se ven redondeadas, actuaban sobre dígitos ocultos («3.14159» → «2» daba 2265358). Si un
  reemplazo reescribe una celda que se veía redondeada, el widget avisa de que pierde los decimales
  que no se veían.
- Dos pestañas con la misma tabla ya no intercambian sus cambios al renombrar un nodo: el nombre
  exacto tiene prioridad y la recuperación por contenido espera a que estén todas las entradas; si
  es ambigua, no se aplica y se avisa.
- Ordenar, «Orden original» o «Restablecer» con una selección envían cada salida una sola vez, y el
  filtro ya no reenvía Data N.
- La huella que comprueba los cambios guardados cubre toda la tabla (antes, una muestra).
- El registro guarda las filas sueltas comprimidas: 150.000 filas alternas ocupan menos de 1 KB.
- Formato de números idéntico al de Orange también con −0 y números grandes, fechas sin pérdida de
  fracciones de segundo, y lo escrito en el filtro no se pierde al cambiar enseguida de pestaña.

## 0.5.1 (2026-11-13)

Correcciones de **Data Table Pro**:

- Abrir una celda numérica y confirmarla sin cambios ya no redondea el valor ni cuenta como edición:
  el editor recibe el número con toda su precisión. Buscar y reemplazar tampoco pierde decimales y no
  toca una celda cuyo número no cambia.
- Reemplazar una categoría por nada o por «?» la deja como valor desconocido; al pasar texto a
  categórica, «?» también es desconocido.
- Las fechas escritas sin zona se toman en la zona horaria de la columna, y una hora escrita en una
  columna de solo fechas se muestra.
- Cambiar el tipo o el rol de varias columnas es todo o nada: si una no puede, no cambia ninguna. Un
  prefijo rechazado por el máximo de valores deja la columna intacta.
- Al abrir un workflow, los cambios guardados se reconocen también por el contenido (nodo
  renombrado), se descartan con aviso si los datos son otros aunque tengan la misma forma, y se
  avisa de los que no corresponden a ninguna entrada. Los avisos de cada pestaña ya no se borran al
  llegar otra entrada, y lo restaurado se puede deshacer.
- Solo se reenvían las salidas que cambian: al seleccionar, la de la selección de esa pestaña.
- Con el filtro rápido activo, buscar, reemplazar y el prefijo actúan solo sobre las filas visibles, y
  el ámbito lo indica.
- Más rápido con tablas grandes: fechas vectorizadas, formato de números y ámbito por filas.

## 0.5.0 (2026-11-13)

- Nuevo widget **Data Table Pro**: muestra varias tablas a la vez, cada una en su pestaña, y las
  edita como una hoja de cálculo.
  - Salidas numeradas **Data 1…5** y **Selected Data 1…5** en el orden de las pestañas, que
    llevan su número; cada tabla de salida se llama como el nodo de origen. Con más de cinco
    entradas, aviso; al quitar un enlace, las salidas se recolocan.
  - Ordenar filas pulsando una columna (y volver al orden original), reordenar columnas
    arrastrando la cabecera, editar celdas con validación por tipo y deshacer y rehacer.
  - Buscar y reemplazar con expresiones regulares y grupos, distinguiendo mayúsculas, en toda la
    tabla o en las columnas o filas seleccionadas, con las coincidencias resaltadas.
  - Prefijo y sufijo, renombrar columnas y cambiar su tipo y su rol (con varios objetivos).
  - Filtro rápido, barras en las numéricas, colores de las categóricas, celdas editadas
    resaltadas, colores propios, estadísticas en la cabecera, copiar y restablecer.
  - Los cambios se guardan con el workflow. Pensado para tablas grandes: vista perezosa y
    operaciones por columnas.

## 0.4.4 (2026-11-11)

- Metadatos para Zenodo (`.zenodo.json`): cada release se archiva con DOI.
- Sin cambios funcionales en los widgets.

## 0.4.3 (2026-11-11)

- Autoría y licencia reforzadas: cabecera SPDX con el copyright de Gualberto Asencio Cortés en todos
  los ficheros de código, fichero `NOTICE` con los términos adicionales del artículo 7 de la GPL-3
  (atribución, origen de las versiones modificadas y nombres) y `CITATION.cff` para citar el add-on.
- Reconocimiento del código adaptado de Orange Data Mining: Test & Score Pro (gestión de modelos y
  muestreos, de Test & Score), la ventana deslizante (bucle de evaluación, de `Validation`) y File
  Import (archivos recientes y arrastrar y soltar, de File; nombre de las descargas, de
  `Orange.data.io`). Esas partes conservan el copyright y la licencia de Orange.
- El paquete declara la licencia `GPL-3.0-or-later` y lleva `LICENSE` y `NOTICE`. La ayuda tiene una
  página «Licencia y autoría» y el copyright en el pie, y el informe de Test & Score Pro indica la
  autoría del add-on en la tabla de reproducibilidad.

## 0.4.2 (2026-11-11)

- **Test & Score Pro** descarta al recibir los datos las filas sin valor en algún objetivo, antes de
  formar las ventanas, y lo indica en el widget y en el informe («Se han descartado N filas sin valor
  en algún objetivo (…)»). Todos los objetivos se evalúan así sobre las mismas filas.
- Corregido: con varios objetivos, una fila final sin valor (como en *y(t+1)*) dejaba la última
  iteración sin test y toda la evaluación acababa en error. Ahora una iteración sin entrenamiento o
  sin test para algún objetivo se omite con un aviso que la identifica, y las demás se evalúan.

## 0.4.1 (2026-11-11)

- Corregido en la **ventana deslizante** de **Test & Score Pro**: medida en meses o años, con el
  primer test un día 29, 30 o 31 (o un 29 de febrero), algunos días nunca se evaluaban y los tests
  tenían longitudes irregulares. Ahora todas las fechas se cuentan desde el inicio y los tests quedan
  contiguos.
- Textos más precisos con la ventana deslizante: el t sin corregir remite a Diebold-Mariano, la
  interpretación de cada comparación nombra a los dos modelos, el gráfico de comparación habla de
  iteraciones, el widget y el informe dicen que el t corregido y McNemar no se aplican, y el t
  bayesiano explica que va sin correlación (ρ = 0).

## 0.4.0 (2026-11-11)

- **Test & Score Pro**: nueva validación por **ventana deslizante** para modelos de series temporales,
  que respeta el orden del tiempo. Se ordena por una variable de tiempo (también en tablas de
  Orange3-Timeseries) o por el orden de las filas, y se elige el inicio (fila o fecha), el entrenamiento
  inicial (porcentaje, filas o hasta una fecha), el avance (creciente o de tamaño fijo), el test, el
  hueco y el paso (en filas o en tiempo, con meses y años de calendario), el fin (fecha o final) y qué
  hacer con el último test incompleto. Las filas con la misma fecha nunca se separan.
- Vista previa de las ventanas que cambia con las opciones, con el número de iteraciones y de
  entrenamientos antes de lanzar; las evaluaciones muy grandes esperan a que se pulse **Evaluar**.
- **Multi-horizonte**: con varias variables objetivo, cada una se entrena y evalúa por separado con las
  mismas ventanas. Nueva pestaña **Horizontes** (matriz modelo × objetivo y curva por horizonte) y
  nueva salida **Horizon Scores**.
- Test principal **Diebold-Mariano** con la corrección de Harvey, Leybourne y Newbold y la
  autocorrelación del horizonte; los tests sobre iteraciones quedan como orientativos.
- Pestañas **Ventanas** e **Iteraciones**; Predictions y Fold Scores incluyen la iteración, las fechas
  de cada ventana y el objetivo. El informe HTML y PDF añade la ventana, las iteraciones y los horizontes.

## 0.3.6 (2026-11-09)

- **Test & Score Pro**: nuevo diseño del informe en PDF, con una primera página de resumen, numeración,
  pie de página y gráficos a buen tamaño. Los títulos y las introducciones van siempre con su contenido,
  las tablas cortas no se parten y las largas repiten la cabecera.
- El informe, en PDF y en HTML, usa un único formato de números y agrupa los avisos repetidos.
- Corregido: la leyenda del gráfico de tiempos ya no tapa las barras.

## 0.3.5 (2026-11-09)

- Corregido en **Test & Score Pro**: con MSE y una variable objetivo muy desplazada, la protección
  frente al ruido numérico ocultaba diferencias reales entre modelos.
- Los mensajes distinguen ahora entre resultados exactamente iguales e iguales solo dentro de la
  precisión numérica.

## 0.3.4 (2026-11-09)

- Corregido en **Test & Score Pro**: una diferencia constante entre modelos ya no se confunde con
  evidencia cuando las particiones tienen valores de escala muy distinta o algún cero.
- Corregido: el ruido de redondeo de una partición enorme ya no pesa en el t ni en el bayesiano.
- Corregido: cuando la métrica vale casi cero (un ajuste perfecto), las diferencias que son solo
  ruido numérico ya no aparecen como significativas.

## 0.3.3 (2026-11-09)

- Corregido en **Test & Score Pro**: exportar el informe fallaba con modelos que sirven para
  clasificación y regresión, como SVM, Tree, Random Forest o kNN.
- Corregido: si no se puede preparar el informe, el widget lo indica y los botones siguen operativos.

## 0.3.2 (2026-11-09)

- Corregido en **Test & Score Pro**: con valores de escala muy distinta (un modelo que diverge, una
  partición enorme), la tolerancia al redondeo ocultaba diferencias reales entre otros modelos o
  particiones. Ahora cada valor se compara con su propia escala.
- Corregido: el aviso del diagrama de diferencias críticas se ajusta al ancho de la figura.
- Mejoras de redacción: la recomendación cuando se apoya en McNemar y el efecto de McNemar, que se
  indica como razón A/B.

## 0.3.1 (2026-11-09)

- Corregido en **Test & Score Pro**: diferencias casi constantes por redondeo ya no se toman por
  evidencia; se tratan igual que una diferencia exactamente constante en todos los tests y en la
  recomendación.
- Corregido: Wilcoxon decide el ganador por sus rangos y no por la media.
- Corregido: McNemar y los tests por filas dicen sobre qué aportan evidencia, y la recomendación
  ya no la atribuye a una métrica que esos tests no miden.
- Corregido: la diferencia se muestra siempre como A − B en las unidades de la métrica, también en
  las de error, y se indica en la tabla, la salida y el informe.
- Corregido: el intervalo de McNemar respeta α, y el muestreo por característica con valores
  perdidos usa las filas realmente empleadas.
- Corregido: Iman-Davenport ya no da un p = 0 falso con concordancia perfecta; LogLoss y
  Specificity se calculan en particiones de una clase cuando están definidas; los modelos con el
  mismo nombre salen igual que en Test & Score.
- El diagrama de diferencias críticas lleva en la propia figura el aviso sobre las particiones.

## 0.3.0 (2026-11-09)

- Widget **Test & Score Pro**: evalúa modelos como Test & Score, con sus mismas entradas y
  salidas, y los compara con tests paramétricos, no paramétricos y bayesianos que tienen en
  cuenta la dependencia entre particiones. Añade gráficos comparativos, recomendación del mejor
  modelo, tablas de resultados en la salida e informe exportable en HTML o PDF.

## 0.2.1 (2026-11-09)

- Corregido en **File Import**: el límite de tiempo y la cancelación de una descarga
  no se atendían hasta que llegaba un bloque completo, y un servidor lento podía
  hacerlos esperar mucho. Ahora responden en menos de un segundo.
- Corregido en **File Import**: los datos comprimidos descargados de una URL ya no
  pueden ocupar una cantidad desmesurada al descomprimirse (bombas de compresión).
  Los archivos locales no tienen este límite.

## 0.2.0 (2026-11-09)

- Widget **File Import**: importa datos desde un archivo o una URL (CSV, TSV, TAB,
  XLS/XLSX y ARFF) con detección automática del formato, vista previa, edición de
  nombre, tipo y rol de cada columna, opciones de importación avanzada y mensajes
  claros ante archivos con problemas. La carga va en segundo plano y se puede
  cancelar.

## 0.1.0 (2026-11-09)

Primera versión.

- Widget **SMOTE**: balancea las clases generando instancias sintéticas de las
  minoritarias con `imbalanced-learn`. Elige solo entre SMOTE, SMOTENC y SMOTEN
  según los atributos, muestra la distribución de clases antes y después, y
  explica en el widget cómo resolver cada problema con los datos.
