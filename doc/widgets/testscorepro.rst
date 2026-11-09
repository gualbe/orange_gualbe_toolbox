.. _test & score pro:

Test & Score Pro
================

Evalúa modelos igual que **Test & Score** y además los compara con tests estadísticos
rigurosos, gráficos y un informe exportable.

.. image:: ../../assets/captura_testscorepro.png
   :alt: El widget con la recomendación del mejor modelo y la tabla de puntuaciones
   :width: 100%

Para qué sirve
--------------

Conecta unos datos y varios modelos. El widget los entrena y evalúa, resalta el mejor
en cada métrica y responde a la pregunta importante: **¿la diferencia entre modelos es
real o puede deberse al azar?** La franja superior resume la respuesta en lenguaje
llano.

Entradas
--------

- **Data**: datos de entrenamiento, con una variable objetivo.
- **Test Data**: datos de test separados (opcional).
- **Learner**: uno o varios modelos.
- **Preprocessor**: preprocesado que se aplica en cada partición (opcional).

Salidas
-------

- **Predictions** y **Evaluation Results**: las mismas que Test & Score. Se conectan a
  Confusion Matrix, ROC Analysis, etc.
- **Scores**: una fila por modelo con todas las métricas, el intervalo, el rango y los
  tiempos.
- **Fold Scores**: una fila por partición y modelo; útil para Box Plot.
- **Pairwise Comparisons**: una fila por pareja y test, con estadístico, valor p, valor p
  ajustado, efecto, intervalo, probabilidades y decisión.
- **Ranks**: rango medio y posición de cada modelo.
- **Best Learner**: el modelo con mejor valor en la métrica de comparación.

Muestreo
--------

Los mismos métodos que Test & Score: validación cruzada, por característica, aleatorio,
dejar uno fuera, sobre datos de entrenamiento y sobre datos de test. Con la misma semilla
(42 por defecto) las particiones y las predicciones coinciden con las de Test & Score.

Por defecto usa **10 particiones**: con menos, los tests tienen poca potencia.

Evaluar sobre los datos de entrenamiento da puntuaciones **optimistas**, y el widget lo
avisa.

Pestañas
--------

- **Puntuaciones**: tabla ordenable con el mejor valor de cada métrica resaltado. Elige
  las columnas en **Métricas visibles**.
- **Comparación**: cada punto es una partición, el rombo el valor global y la barra el
  intervalo de confianza.
- **Rangos**: diagrama de diferencias críticas. Los modelos unidos por una barra no se
  distinguen.
- **Matriz**: fila contra columna para el test elegido. Azul: gana la fila; naranja:
  pierde; gris: sin evidencia.
- **Bayesiano**: probabilidades del mejor modelo frente a cada uno de los demás.
- **Tiempos**: entrenamiento y predicción.
- **Tests y supuestos**: todos los números, el supuesto de cada test y sus avisos.

Cada gráfico se guarda con **Guardar imagen…** en PNG, SVG o PDF. Cambiar la métrica, α
o el ROPE recalcula los tests y los gráficos **sin volver a entrenar**. Tampoco se
reentrenan los modelos que no hayan cambiado.

Cómo elegir e interpretar los tests
-----------------------------------

.. warning::

   Las particiones de una validación cruzada **no son independientes**, porque comparten
   casi todos los datos de entrenamiento. Por eso un t-test normal da demasiados falsos
   positivos. La decisión del widget se basa en el **t corregido** y, en la pestaña
   Bayesiano, en el **t bayesiano correlacionado**.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Test
     - Qué responde
     - Cómo leerlo
   * - **t corregido (Nadeau-Bengio)**
     - ¿Difieren dos modelos, teniendo en cuenta que las particiones se solapan?
       Multiplica la varianza por 1/J + n_test/n_train.
     - Es el test principal con particiones. p ajustado < α: hay evidencia de diferencia.
   * - **t emparejado sin corregir**
     - Lo mismo, suponiendo particiones independientes.
     - **Solo como referencia.** Da demasiados positivos: en una simulación con 10
       particiones, un 15 % en vez del 5 %.
   * - **Wilcoxon**
     - Versión no paramétrica, con rangos de las diferencias.
     - No supone normalidad, pero sí independencia: orientativo. Con 5 particiones o menos
       nunca alcanza p < 0,05.
   * - **Friedman, Iman-Davenport**
     - ¿Hay alguna diferencia entre todos los modelos?
     - Se concibió para varios conjuntos de datos. Sobre particiones de uno solo es
       **descriptivo**.
   * - **Nemenyi, Holm (tras Friedman)**
     - ¿Qué parejas difieren según sus rangos?
     - Misma limitación que Friedman.
   * - **McNemar**
     - ¿Aciertan dos clasificadores en filas distintas?
     - Cada fila debe predecirse una sola vez: no se ofrece con muestreo aleatorio.
   * - **t bayesiano correlacionado**
     - ¿Qué probabilidad hay de que un modelo sea mejor, de que sean equivalentes o de que
       sea peor?
     - Se decide con P ≥ 0,95. Si no, **indeciso**: no hay datos suficientes.
   * - **Por filas (error absoluto)**
     - Regresión sin particiones: compara el error fila a fila.
     - Válido con datos de test separados. En dejar uno fuera o sobre entrenamiento, con
       reservas.

**Sobre qué aporta evidencia cada test**: los tests por particiones comparan la métrica elegida.
McNemar compara los **aciertos por fila** y los tests por filas, el **error absoluto**. Si la métrica
elegida es otra (por ejemplo AUC con dejar uno fuera), su resultado no es evidencia sobre esa métrica,
y el widget lo dice.

**Orientación de las diferencias**: en las tablas, la salida y el informe, *Diferencia (A − B)* es
siempre el valor de A menos el de B, en las unidades de la métrica. En métricas de error (RMSE, MAE…)
y en rangos, una diferencia **negativa** significa que A es mejor. El efecto sigue el mismo signo. La
columna *Decisión* indica directamente qué modelo gana.

**Empates y redondeo**: dos valores que solo difieren por el redondeo del ordenador se consideran
iguales. Cada valor se compara con su propia escala y, cuando una métrica vale casi cero (por ejemplo
un ajuste perfecto), también con el error de redondeo propio de esa métrica: el de 1 en las acotadas, el
de la variable objetivo en RMSE y MAE y, en MSE, el que resulta de ese error y del valor de cada
partición. Si la diferencia entre dos modelos es constante
en todas las particiones, no hay variación con la que estimar la incertidumbre y el test es **no
evaluable**.

**ROPE** (región de equivalencia práctica): diferencia por debajo de la cual dos modelos se
consideran iguales a efectos prácticos. En automático vale 0,01 en métricas acotadas (AUC,
CA, F1…) y el 1 % de la media en métricas de error. Puedes fijarlo en **Opciones avanzadas**.

**Comparaciones múltiples**: al hacer muchas comparaciones, alguna saldrá significativa por
azar. Por defecto se ajustan con Holm; también están Bonferroni, Benjamini-Hochberg y sin
corrección.

**Tamaños de efecto**: un valor p no dice si la diferencia importa. La g de Hedges (t) y la r
biserial de rangos (Wilcoxon) la miden: insignificante, pequeña, media o grande. En McNemar, el
efecto es la razón entre las filas en que solo acierta A y aquellas en que solo acierta B.

**Intervalos de confianza**: con particiones usan la misma corrección de Nadeau-Bengio. Sin
particiones, proceden de un remuestreo de las filas.

**«No significativo» no significa «iguales»**: solo que no hay evidencia suficiente. Para
afirmar equivalencia, mira la probabilidad bayesiana de equivalencia.

Exportar el informe
-------------------

**Exportar informe…** crea un HTML autocontenido, con las imágenes incrustadas, o un PDF. Incluye
los datos, el muestreo y la semilla, los modelos y sus parámetros, las puntuaciones, los gráficos,
cada test con su supuesto e interpretación, y las versiones del software para reproducirlo.
También funciona el informe nativo de Orange.

Problemas habituales
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situación
     - Qué hace el widget
   * - Un modelo falla
     - Los demás siguen; se indica cuál y por qué.
   * - Filas sin valor en la clase
     - Se quitan, indicando cuántas.
   * - Estratificación imposible
     - Se hace sin estratificar y se avisa.
   * - Más particiones que filas
     - Error con la solución.
   * - Métrica indefinida en una partición (AUC con una sola clase)
     - Esa partición se descarta al comparar y se indica cuántas. Las métricas que sí están
       definidas con una clase, como LogLoss, se calculan.
   * - Modelos idénticos, varianza cero o empates
     - El test lo dice en lugar de dar un número engañoso.
   * - Muy pocas particiones
     - Aviso de que la potencia es baja.
   * - Cambios en las entradas durante un cálculo
     - Se descarta el cálculo anterior.

Referencias
-----------

- Nadeau y Bengio (2003). *Inference for the generalization error*. Machine Learning 52.
- Demšar (2006). *Statistical comparisons of classifiers over multiple data sets*. JMLR 7.
- Benavoli, Corani, Demšar y Zaffalon (2017). *Time for a change: a tutorial for comparing
  multiple classifiers through Bayesian analysis*. JMLR 18.
