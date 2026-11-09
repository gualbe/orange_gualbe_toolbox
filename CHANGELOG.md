# Cambios

Cada versión se publica a la vez en GitHub (release `vX.Y.Z`) y en
[PyPI](https://pypi.org/project/orange-gualbe-toolbox/).

## 0.1.0 (2026-11-09)

Primera versión.

- Widget **SMOTE**: balancea las clases generando instancias sintéticas de las
  minoritarias con `imbalanced-learn`. Elige solo entre SMOTE, SMOTENC y SMOTEN
  según los atributos, muestra la distribución de clases antes y después, y
  explica en el widget cómo resolver cada problema con los datos.
