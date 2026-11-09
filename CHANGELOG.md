# Cambios

Cada versión se publica a la vez en GitHub (release `vX.Y.Z`) y en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/).

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
