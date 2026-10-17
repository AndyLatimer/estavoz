"""Servidor: recibe unos segundos de audio y dice qué doblador es.

Arranque: .venv/bin/uvicorn backend.app:app --port 8000
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voz import cargar_audio, huellas, modelo, separador  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
app = FastAPI(title="Estavoz")
INDICE: dict = {}


def cargar_indice():
    datos = np.load(RAIZ / "data" / "indice.npz")
    INDICE["vectores"] = datos["vectores"]
    INDICE["etiquetas"] = datos["etiquetas"]
    # la grabación se trata igual que se trataron los clips del índice
    INDICE["separar"] = bool(datos["separado"]) if "separado" in datos.files else False
    INDICE["fichas"] = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))


@app.on_event("startup")
def arrancar():
    modelo()
    if (RAIZ / "data" / "indice.npz").exists():
        cargar_indice()
        if INDICE["separar"]:
            separador()


def puntuar(consulta: np.ndarray, top: int = 3) -> list[dict]:
    sims = consulta @ INDICE["vectores"].T            # (trozos consulta, trozos índice)
    mejor_por_trozo = sims.mean(axis=0)               # parecido medio a cada trozo del índice
    etiquetas = INDICE["etiquetas"]
    resultados = []
    for slug in np.unique(etiquetas):
        s = np.sort(mejor_por_trozo[etiquetas == slug])[::-1]
        resultados.append((float(s[:3].mean()), slug))  # media de sus 3 trozos más parecidos
    resultados.sort(reverse=True)
    return [
        {"slug": slug, "parecido": round(score, 3), **INDICE["fichas"][slug]}
        for score, slug in resultados[:top]
    ]


@app.post("/api/identificar")
async def identificar(audio: UploadFile):
    if not INDICE:
        raise HTTPException(503, "Aún no hay actores indexados")
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or "a.webm").suffix) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            consulta = huellas(cargar_audio(tmp.name, separar=INDICE["separar"], max_seg=20))
        except Exception:
            raise HTTPException(400, "No se pudo leer el audio")
    if len(consulta) == 0:
        raise HTTPException(400, "Audio demasiado corto: graba al menos 3 segundos")
    return {"resultados": puntuar(consulta)}


app.mount("/", StaticFiles(directory=RAIZ / "frontend", html=True), name="web")
