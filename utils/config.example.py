# utils/config.example.py
# ---------------------------------------------------------------
# Plantilla de configuración de rutas locales para los scripts
# de preparación de datos (utils/).
#
# Uso:
#   cp utils/config.example.py utils/config.py
#   Edita config.py con las rutas reales de tu entorno local.
#
# ---------------------------------------------------------------

from pathlib import Path

# Carpetas con los CSV brutos de cada planta
# Estructura esperada: <PLANTA_RAW>/<anyo>/<mes>/*.csv
DATA_AFRISOL = Path(r"")
DATA_LECA1   = Path(r"")
DATA_E03     = Path(r"")