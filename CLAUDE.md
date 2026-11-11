# orange_gualbe_toolbox — Toolbox Gualbe

Utilidades visuales de ciencia de datos para Orange Data Mining. Responde siempre en español de España. Esta carpeta es la de la
sesión de trabajo `toolbox-mant`; la coordinadora del grupo vive en
`../orange-hub` y te manda las peticiones.

Antes de tocar widgets, ayuda o empaquetado lee
`../orange-hub/conocimiento/trampas-orange.md`.

## Identidad

- Distribución y namespace: `orange_gualbe_toolbox` / `orangecontrib.orange_gualbe_toolbox` (en PyPI
  normalizado como `orange-gualbe-toolbox`).
- Categoría: «Toolbox Gualbe», color `#7B68EE`.
- Dependencias: `Orange3>=3.32.0`, `imbalanced-learn`, `pandas>=2.0` (File Import usa
  `format="ISO8601"`), `openpyxl`, `xlrd`, `baycomp`, `scipy>=1.9` (Wilcoxon con `method`)
  y `matplotlib`. Sin más límites de versión. `statsmodels` solo se usa en los tests.
- Repo público https://github.com/gualbe/orange_gualbe_toolbox (rama `main`, fuente de
  verdad) · PyPI https://pypi.org/project/orange-gualbe-toolbox/
- Idioma de la ayuda: `es`.

## Decisiones fijadas por el usuario

- Todos los widgets deben ser visuales, intuitivos, fáciles de usar y con su ayuda.
- SMOTE elige el algoritmo solo y lo indica en el widget: `SMOTE` con atributos
  numéricos, `SMOTENC` con mixtos y `SMOTEN` si todos son categóricos.

### Decisiones del widget SMOTE (tomadas en el alta, 2026-11-09)

- La salida conserva el dominio y las filas originales (valores, metas, pesos e
  ids); las sintéticas van al final. No se añade ninguna columna de «origen»
  porque cambiaría el dominio.
- En las filas sintéticas las metas son desconocidas y el peso es 1 si la tabla
  tiene pesos (SMOTE no los usa; el widget lo avisa).
- El widget calcula el objetivo por clase y siempre pasa un `dict` a
  imbalanced-learn, así la vista previa y el resultado no pueden divergir.
  Las estrategias equivalen a `not majority`, `minority` y `not minority`, más un
  porcentaje de la mayoritaria que vale también con más de dos clases. `all` no se
  ofrece: al sobremuestrear equivale a `not majority`. Empates en la minoritaria:
  gana el primer valor de la clase.
- Solo cuentan para el mínimo de *k* + 1 las clases que van a crecer.
- Con valores perdidos no se imputa nada: error con instrucciones y sin salida.
- imbalanced-learn se importa dentro de la tarea: si falla, es un mensaje del
  widget y no desaparece del lienzo.
- Las tablas dispersas siguen dispersas a la salida. Solo `SMOTE` trabaja en
  disperso; `SMOTENC` y `SMOTEN` convierten a denso internamente.

### Decisiones del widget File Import (tomadas en su alta, 2026-11-09)

- Pedido por el usuario: integrar lo mejor de File y CSV File Import, con carga desde
  URL, vista previa, configuración de columnas, importación avanzada y los formatos
  CSV, TSV, TAB, XLS/XLSX y ARFF; optimizado y muy robusto ante datos anómalos.
- Motor sin Qt en `_fileimport.py`. Todos los formatos terminan en un DataFrame más
  una lista de `ColumnMeta`; convertir a `Table` es otro paso, así que editar columnas
  nunca vuelve a leer el archivo.
- Texto siempre con el motor C de pandas a través de `_Feed`, un objeto con `read()`
  y sin atributo `mode` (así pandas no lo envuelve en `TextIOWrapper`). En esa misma
  lectura transcodifica a UTF-8, cuenta filas irregulares con numpy, informa del
  progreso y permite cancelar. No añadir una segunda pasada sobre el archivo.
- Sin salida Data Frame: solo sirve para *Python Script*, duplica memoria y se obtiene
  con `table_to_frame`.
- Inferencia: textos con hasta 100 valores (y `filas ** 0.7`) son categóricos, si no
  texto; enteros con dos valores, categóricos (como File); columna con al menos un
  95 % de números, numérica; columnas vacías, omitidas con aviso. Nombres `class`,
  `clase`, `target` u `objetivo` se proponen como clase. El texto siempre es meta.
- Filas con campos de más o de menos: se conservan y se informa de cuántas son.
- La columna de pesos de la cabecera de Orange se importa como meta.
- Celdas combinadas de Excel: el valor se copia a todas las que abarca.
- Fechas fuera de 1677–2262: perdidas con aviso (límite de pandas en nanosegundos).
- El contenido manda sobre la extensión. URL solo http/https, también al redirigir,
  con límite de tiempo y de tamaño (500 MB por defecto, configurable).
- El estado se guarda por origen en `source_configs`: la clave es el nombre del
  archivo (sobrevive a mover el workflow) o la URL.

### Decisiones del widget Test & Score Pro (tomadas en su alta, 2026-11-09)

- Pedido por el usuario: partir de Test & Score, con más visualización, comparaciones, tests
  no paramétricos y bayesianos, informe exportable y tablas de resultados en la salida.
- Partes sin Qt: `_tsp_stats.py` (tests), `_tsp_engine.py` (evaluación y salidas),
  `_tsp_plots.py` (figuras) y `_tsp_report.py` (informe). Mismos `scorers` y `Results` que el
  nativo: las puntuaciones y las salidas Predictions y Evaluation Results deben coincidir.
- Rigor: las particiones no son independientes. El test principal con particiones es el t de
  Nadeau-Bengio. El t clásico solo se ofrece como referencia, con aviso. Friedman, Nemenyi y
  Wilcoxon sobre particiones son orientativos y lo dicen, también en la propia figura.
- Bayesiano: `baycomp` con validación cruzada (ρ = 1/k); en el resto, su misma posterior con
  ρ = n_test / n_total, que `baycomp` no permite elegir.
- Sin particiones (dejar uno fuera, sobre entrenamiento, sobre test): McNemar o tests por filas
  e intervalos por remuestreo de filas. McNemar compara aciertos y los tests por filas, error
  absoluto: la recomendación no afirma evidencia sobre otra métrica.
- Orientación: por dentro, puntuaciones orientadas (positivo = A mejor). Lo que ve el usuario
  (tabla, salida, informe) es siempre valor(A) − valor(B) en las unidades de la métrica; en
  errores y rangos, negativo significa que A es mejor. Cada test decide con su propio estadístico.
- Lo que solo difiere por redondeo (tolerancia relativa 1e-13) es igual; una diferencia constante
  es «no evaluable», no evidencia. La tolerancia va siempre valor a valor: una escala global,
  incluso por bloque, hace que un solo valor enorme borre las diferencias reales del resto.
  Dos valores son indistinguibles si sus intervalos valor ± tolerancia tienen un punto común.
  Una diferencia es constante si lo tienen todas; las diferencias limpias (`clean_differences`)
  son las que usan el t, Wilcoxon y el bayesiano.
- Los empates de Wilcoxon y de los rangos usan una tolerancia 10 veces más estricta
  (`TIE_FACTOR`): la constante necesita margen para el ruido, pero un empate falso cambia el
  valor p. Se agrupa por cercanía, no con una rejilla.
- Suelo absoluto por métrica (`noise_floor`), para cuando la métrica vale casi cero, como en un
  ajuste perfecto: 1e-13 × su escala natural (1 en las acotadas y LogLoss, 100 en MAPE, SMAPE y
  CVRMSE, escala de `y` en RMSE y MAE). En MSE no vale el cuadrado de la escala, porque ocultaría
  diferencias reales con `y` muy desplazada: su ruido es 2·δ·RMSE + δ², con δ = 1e-13·máx|y|, valor
  a valor por partición. Una métrica desconocida no lleva suelo.
- 10 particiones por defecto (el nativo, 5), porque con menos los tests apenas tienen potencia.
  Semilla 42 por defecto, la del nativo, para que las particiones coincidan.
- Un hilo por modelo, solo con 5.000 filas o más y sin preprocesador compartido. Cambiar métrica,
  α o ROPE no reentrena; tampoco se reentrena un modelo que no ha cambiado.
- Salidas nuevas: Scores, Fold Scores, Pairwise Comparisons, Ranks y Best Learner.
- Informe: `_tsp_report.py` describe el contenido como una lista de bloques y lo dibuja en HTML;
  `_tsp_pdf.py` lo maqueta en PDF. Cualquier cambio de contenido se hace en los bloques, para que
  los dos formatos digan lo mismo. Un solo formato numérico: coma decimal y científica uniforme.
- PDF con `QPainter` sobre `QPdfWriter`, midiendo alturas, en el hilo de la tarea. `QTextDocument` no
  mantiene un título con su contenido y `QtWebEngine` obliga al hilo de la interfaz. A4 apaisado por
  las tablas anchas y los gráficos. `render_pdf` devuelve un registro de posiciones para los tests.
- Maquetación: títulos, introducciones (`keep_next`) y pies de tabla (`keep_previous`) van con su
  contenido; no se parte una tabla de 8 filas o menos ni la que ocupa menos del 45 % de la altura
  útil; las largas repiten cabecera y dejan al menos 3 filas por página; los números no se parten.
  Al cambiar la maquetación, abre las páginas renderizadas: los tests no ven recortes en los gráficos.

### Decisiones de la ventana deslizante de Test & Score Pro (0.4.0, 2026-11-11)

- Pedido por el usuario: validación por ventana deslizante para series temporales, elegible junto a los
  demás muestreos, con entrenamiento inicial en %, filas o hasta fecha, inicio por fila o fecha, avance
  creciente o de tamaño fijo, test, fin por fecha y multi-horizonte (varias variables objetivo).
- Muestreo `WINDOW = 6`, añadido al final para no romper ajustes guardados. Motor sin Qt en
  `_tsp_window.py`: `plan_windows` calcula las ventanas sin entrenar (vista previa) y la evaluación usa
  ese mismo plan con `WindowValidation`, subclase de `Validation` de Orange: cada iteración es una
  partición de Results y el preprocesador se ajusta solo con el entrenamiento de cada iteración.
- Orden por una variable de tiempo (la de Orange3-Timeseries si la hay) o por el de las filas. Sin
  fecha: fuera con aviso; desordenadas: se ordenan. Un corte nunca separa filas con la misma fecha.
- Test, hueco y paso en filas o en tiempo (meses y años de calendario). Todas las fronteras se cuentan
  desde el origen, nunca desde otra frontera: si no, un inicio a fin de mes pierde días (0.4.1).
- El último test incompleto se descarta por defecto (seleccionable). Más de 2.000 entrenamientos
  (iteraciones × modelos × objetivos) esperan al botón Evaluar.
- Multi-horizonte: un modelo por objetivo e iteración; los demás objetivos pasan a metas, nunca a
  atributos. Las filas sin valor en algún objetivo se descartan al recibir los datos, antes de
  formar ventanas o particiones, para todos los objetivos a la vez (todos usan las mismas filas), y
  se informa de cuántas faltaban en cada uno (0.4.2). Una iteración sin entrenamiento o sin test
  para algún objetivo se omite con aviso; nunca para la evaluación. Objetivos numéricos y categóricos
  mezclados: error. En la comparación entran los modelos que van bien en todos los objetivos.
  Cambiar el objetivo analizado no reentrena.
- Tests: el principal es Diebold-Mariano con la corrección HLN y la varianza a largo plazo hasta
  h − 1 (h = posición del objetivo, o fijo en Opciones avanzadas), sobre la pérdida de la métrica
  elegida. No se ofrecen Nadeau-Bengio ni McNemar; t, Wilcoxon, bayesiano (ρ = 0) y Friedman sobre
  iteraciones son orientativos y lo dicen. Con tests solapados, DM usa la predicción más reciente.
- Salidas: Predictions y Fold Scores con iteración, fechas de la ventana y objetivo; nueva Horizon
  Scores. Pestañas Ventanas, Iteraciones y Horizontes tras Puntuaciones (`settings_version` 2).

## Widgets

| Widget | Fichero / clase | Qué hace |
|---|---|---|
| SMOTE | `owsmote.py` / `OWSmote` | Balancea las clases con instancias sintéticas (SMOTE, SMOTENC o SMOTEN); gráfico antes/después. |
| File Import | `owfileimport.py` / `OWFileImport` (motor en `_fileimport.py`) | Importa desde archivo o URL (CSV, TSV, TAB, XLS/XLSX y ARFF) con vista previa, edición de columnas e importación avanzada. |
| Test & Score Pro | `owtestscorepro.py` / `OWTestScorePro` (`_tsp_*.py`) | Evalúa modelos como Test & Score, también con ventana deslizante para series temporales, y los compara con tests rigurosos, gráficos, tablas de resultados e informe HTML o PDF. |

## Pruebas

```bash
../orange-hub/scripts/probar.sh .            # humo + tests/test_*.py + ayuda + paquete
../orange-hub/scripts/probar.sh . --rapido   # solo humo y tests, en Orange.app
```

Cada `tests/test_*.py` es un script ejecutable con el Python de Orange que sale con
código distinto de 0 si algo falla. `probar.sh` lanza los tests solo en Orange.app:
para Orange-Gualbe.app, ejecuta a mano
`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /Applications/Orange-Gualbe.app/Contents/MacOS/python tests/test_<widget>.py`.
Las capturas de la ayuda y el README se regeneran con `assets/generar_imagenes.py`.

## Ficheros comunes con los otros add-ons

Iguales en todos salvo el nombre del paquete: `.github/workflows/publish.yml`,
`.gitignore`, `LICENSE`, `doc/conf.py` (salvo idioma y estáticos), la estructura de
`pyproject.toml` y la cabecera de `widgets/__init__.py` (`WIDGET_HELP_PATH`). Si un
cambio toca uno de ellos, dilo en el informe a la coordinadora.
`../orange-hub/scripts/comparar.sh <ruta>` muestra cuánto difiere.

## Cierre obligatorio de cada mantenimiento

Skill `orange-release`. En resumen: pruebas en verde → `version` en
`pyproject.toml` (parche para correcciones, menor para funciones nuevas) →
sección `## X.Y.Z (AAAA-MM-DD)` en `CHANGELOG.md` → si cambia un widget, su ayuda y
el README → commit en `main` («Descripción (X.Y.Z)») →
`../orange-hub/scripts/publicar.sh .` (push, release `vX.Y.Z`, CI a PyPI e
instalación en Orange.app y Orange-Gualbe.app). Los cambios que solo afectan al
repo no necesitan versión nueva.

## Pendientes conocidos

Ninguno.
