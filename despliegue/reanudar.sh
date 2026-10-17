#!/bin/sh
# Relanza los procesos largos que queden pendientes (tras un reinicio del entorno) y el informe de progreso.
# Todos se pueden repetir sin perder lo ya hecho.
cd "$(dirname "$0")/.."
LOG=${LOG:-/tmp/estavoz-logs}; mkdir -p "$LOG"
corriendo() { ps -eo args | grep -v grep | grep -q "$1"; }

corriendo "eldoblaje.py trabajos" || nohup .venv/bin/python -u scraper/eldoblaje.py trabajos >> "$LOG/trabajos.log" 2>&1 &
corriendo "entrevistas.py" || nohup sh -c '.venv/bin/python -u scraper/entrevistas.py buscar && .venv/bin/python -u scraper/entrevistas.py descargar' >> "$LOG/entrevistas.log" 2>&1 &
corriendo "indexar.py" || ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx nohup .venv/bin/python -u backend/indexar.py --aumentar --cache data/cache_indice_vb >> "$LOG/indexar.log" 2>&1 &
corriendo "progreso.sh" || nohup sh despliegue/progreso.sh > /dev/null 2>&1 &
echo "procesos lanzados; registros en $LOG"
