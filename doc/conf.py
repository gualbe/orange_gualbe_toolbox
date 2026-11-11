# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Gualberto Asencio Cortés
# Part of orange_gualbe_toolbox. See the LICENSE and NOTICE files for details.

# Configuración de Sphinx para la ayuda nativa de los widgets (F1 / "?").
# Se genera dentro del paquete:
#   python -m sphinx -b html doc orangecontrib/orange_gualbe_toolbox/_help/html
project = "orange_gualbe_toolbox"
author = "Gualberto Asencio Cortés"
copyright = "2026, Gualberto Asencio Cortés. GPL-3.0-or-later"
extensions = []
exclude_patterns = ["_build", "build"]
html_theme = "alabaster"
language = "es"
