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


def descargar_respuestas(url: str, clave: str, timeout: int = 60) -> list[dict]:
    """Descarga paginada de `respuestas` compatible con esquemas anteriores y ampliados."""
    base = url.rstrip("/").removesuffix("/rest/v1")
    cab = {"apikey": clave, "Authorization": f"Bearer {clave}"}
    filas, desde = [], 0
    while True:
        r = requests.get(
            f"{base}/rest/v1/respuestas",
            headers={**cab, "Range": f"{desde}-{desde + 999}"},
            params={"select": "*", "order": "id"},
            timeout=timeout,
        )
        r.raise_for_status()
        lote = r.json()
        filas += lote
        if len(lote) < 1000:
            break
        desde += 1000
    return filas


def main():
    url, clave = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and clave):
        sys.exit(__doc__)
    filas = descargar_respuestas(url, clave)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(filas), encoding="utf-8")
    print(f"{len(filas)} respuestas -> {SALIDA}")


if __name__ == "__main__":
    main()
