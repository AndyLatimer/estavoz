"""Descarga las respuestas "¿Acertamos?" de Supabase a data/respuestas.json.

Necesita la clave secreta (service_role) del proyecto, que NO va en la web:
  SUPABASE_URL=https://xxx.supabase.co SUPABASE_SERVICE_KEY=... .venv/bin/python backend/aprender.py

Después, despliegue/exportar_web.py las incorpora (con filtros) y se vuelve a subir la web.
"""
import json
import os
import sys
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "respuestas.json"


def main():
    url, clave = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and clave):
        sys.exit(__doc__)
    cab = {"apikey": clave, "Authorization": f"Bearer {clave}"}
    filas, desde = [], 0
    while True:
        r = requests.get(f"{url}/rest/v1/respuestas", headers={**cab, "Range": f"{desde}-{desde + 999}"},
                         params={"select": "id,creado,propuesto,acierto,real,huellas", "order": "id"}, timeout=60)
        r.raise_for_status()
        lote = r.json()
        filas += lote
        if len(lote) < 1000:
            break
        desde += 1000
    SALIDA.write_text(json.dumps(filas), encoding="utf-8")
    print(f"{len(filas)} respuestas -> {SALIDA}")


if __name__ == "__main__":
    main()
