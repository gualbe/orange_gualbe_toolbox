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
- Dependencias: `Orange3>=3.32.0` e `imbalanced-learn` (sin fijar versión).
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

## Widgets

| Widget | Fichero / clase | Qué hace |
|---|---|---|
| SMOTE | `owsmote.py` / `OWSmote` | Balancea las clases con instancias sintéticas (SMOTE, SMOTENC o SMOTEN); gráfico antes/después. |

## Pruebas

```bash
../orange-hub/scripts/probar.sh .            # humo + tests/test_*.py + ayuda + paquete
../orange-hub/scripts/probar.sh . --rapido   # solo humo y tests, en Orange.app
```

Cada `tests/test_*.py` es un script ejecutable con el Python de Orange que sale con
código distinto de 0 si algo falla. `probar.sh` lanza los tests solo en Orange.app:
para Orange-Gualbe.app, ejecuta a mano
`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /Applications/Orange-Gualbe.app/Contents/MacOS/python tests/test_smote.py`.
La captura de la ayuda y el README se regenera con `assets/generar_imagenes.py`.

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
