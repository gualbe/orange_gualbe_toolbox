# Cambios

Cada versión se publica a la vez en GitHub (release `vX.Y.Z`) y en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/).

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
