#!/bin/sh
# Escribe PROGRESO.md cada minuto con el estado de los procesos largos.
cd "$(dirname "$0")/.."
LOG=${LOG:-/tmp/estavoz-logs}
while true; do
  total=$(ls -d data/actores/*/ 2>/dev/null | wc -l)
  trabajos=$(ls data/actores/*/trabajos.json 2>/dev/null | wc -l)
  indice=$(ls data/cache_indice_vb/*.npz 2>/dev/null | wc -l)
  buscados=$(python3 -c "import json;print(len(json.load(open('data/entrevistas/episodios.json'))))" 2>/dev/null || echo 0)
  con_ep=$(python3 -c "import json;print(sum(1 for v in json.load(open('data/entrevistas/episodios.json')).values() if v))" 2>/dev/null || echo 0)
  bajados=$(ls data/entrevistas/audio/*.mp3 2>/dev/null | wc -l)
  episodios=$(python3 -c "import json;print(sum(len(v) for v in json.load(open('data/entrevistas/episodios.json')).values()))" 2>/dev/null || echo 0)
  estado() { ps -eo args | grep -v grep | grep -q "^\.venv/bin/python -u .*$1" && echo "en marcha" || echo "parado"; }
  cat > PROGRESO.md.tmp <<FIN
# Progreso de Estavoz

Actualizado: $(date '+%d/%m/%Y %H:%M') (hora del servidor, UTC)

| Tarea | Avance | Estado |
|---|---|---|
| Índice con el modelo de voz nuevo | $indice de $total dobladores | $(estado indexar.py) |
| Lista completa de trabajos ("¿Qué estás viendo?") | $trabajos de $total dobladores | $(estado "eldoblaje.py trabajos") |
| Búsqueda de entrevistas | $buscados de ~153 dobladores ($con_ep con entrevistas) | $(estado "entrevistas.py buscar") |
| Descarga de entrevistas | $bajados de $episodios episodios | $(estado "entrevistas.py descargar") |

Si todo pone "parado" y no está terminado, el entorno se ha reiniciado: avisa a Claude.
FIN
  mv PROGRESO.md.tmp PROGRESO.md
  sleep 60
done
