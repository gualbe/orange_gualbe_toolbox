# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

import os

NAME = "Toolbox Gualbe"
DESCRIPTION = "Utilidades visuales de ciencia de datos para Orange Data Mining."
ICON = "icons/category.svg"
BACKGROUND = "#7B68EE"

# Ayuda nativa (F1 / "?"): HTML generado con Sphinx DENTRO del paquete
# (orangecontrib/orange_gualbe_toolbox/_help/html) para que funcione igual instalado
# desde PyPI que en editable. Se calcula con __file__, no con el marcador
# {DEVELOP_ROOT} (no es fiable con instalaciones editables).
_HELP_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "_help", "html")
)

WIDGET_HELP_PATH = (
    (_HELP_DIR + os.sep, None),
)
