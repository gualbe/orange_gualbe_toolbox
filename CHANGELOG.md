# Cambios

Cada versión se publica a la vez en GitHub (release `vX.Y.Z`) y en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/).

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
