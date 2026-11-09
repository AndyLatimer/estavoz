"""Servidor: recibe unos segundos de audio y dice qué doblador es.

Arranque: .venv/bin/uvicorn backend.app:app --port 8000
"""
from contextlib import asynccontextmanager
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lda import proyectar  # noqa: E402
from metadatos_modelo import validar_artefactos_web, validar_compatibilidad_indice  # noqa: E402
from rechazo import UMBRALES_VOXBLINK_R100_CALIBRADOS, clasificar_consulta  # noqa: E402
from voz import ONNX, cargar_audio, huellas, info_modelo_actual, modelo, separador, sesion_onnx  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
INDICE: dict = {}


def cargar_indice():
    ruta_npz = RAIZ / "data" / "indice.npz"
    dir_web = RAIZ / "frontend" / "datos"
    if ruta_npz.exists():
        datos = np.load(ruta_npz)
        info_mod = validar_compatibilidad_indice(datos, info_modelo=info_modelo_actual())
        INDICE["vectores"] = datos["vectores"]
        INDICE["etiquetas"] = datos["etiquetas"]
        # la grabación se trata igual que se trataron los clips del índice
        INDICE["separar"] = bool(datos["separado"]) if "separado" in datos.files else False
        INDICE["fichas"] = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
        INDICE["info_modelo"] = info_mod
        INDICE["modo"] = "npz"
    elif (dir_web / "actores.json").exists() and (dir_web / "trozos.bin").exists():
        ruta_onnx = RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"
        resumen_web = validar_artefactos_web(dir_web, ruta_onnx=ruta_onnx if ruta_onnx.exists() else None)
        meta = json.loads((dir_web / "actores.json").read_text(encoding="utf-8"))
        dim = int(meta["dim"])
        actores_lista = meta["actores"]
        slugs = [a["id"] for a in actores_lista]
        lda_raw = np.fromfile(dir_web / "lda.bin", dtype="<f4")
        INDICE["lda_mu"] = lda_raw[:dim].astype(np.float32)
        INDICE["lda_W"] = lda_raw[dim:].reshape(dim, dim).astype(np.float32)
        INDICE["centroides"] = np.fromfile(dir_web / "centroides.bin", dtype="<f4").reshape(len(slugs), dim).astype(np.float32)
        raw_trozos = (dir_web / "trozos.bin").read_bytes()
        nA = len(slugs) + 1
        inicio = np.frombuffer(raw_trozos, dtype="<u4", count=nA, offset=0)
        n_total = int(inicio[-1])
        escala = np.frombuffer(raw_trozos, dtype="<f4", count=n_total, offset=nA * 4)
        q8 = np.frombuffer(raw_trozos, dtype=np.int8, count=n_total * dim, offset=(nA + n_total) * 4).reshape(n_total, dim)
        INDICE["vectores"] = (q8.astype(np.float32) * escala[:, None]).astype(np.float32)
        INDICE["inicio"] = inicio
        INDICE["slugs"] = slugs
        INDICE["fichas"] = {a["id"]: a for a in actores_lista}
        INDICE["separar"] = False
        INDICE["info_modelo"] = resumen_web
        INDICE["modo"] = "web"


def arrancar():
    sesion_onnx() if ONNX else modelo()
    if (RAIZ / "data" / "indice.npz").exists() or (RAIZ / "frontend" / "datos" / "actores.json").exists():
        cargar_indice()
        if INDICE.get("separar"):
            separador()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    arrancar()
    yield


app = FastAPI(title="Estavoz", lifespan=lifespan)


def _ranking_completo(consulta: np.ndarray) -> list[tuple[float, str]]:
    if INDICE.get("modo") == "web":
        qs = proyectar(consulta, INDICE["lda_mu"], INDICE["lda_W"])
        sims_cent = (qs @ INDICE["centroides"].T).mean(axis=0)
        sims_troz = (qs @ INDICE["vectores"].T).mean(axis=0)
        inicio = INDICE["inicio"]
        slugs = INDICE["slugs"]
        resultados = []
        for i, slug in enumerate(slugs):
            ini, fin = int(inicio[i]), int(inicio[i + 1])
            media_c = float(sims_cent[i])
            if fin > ini:
                s = np.sort(sims_troz[ini:fin])[::-1]
                mejores = float(s[:3].mean())
            else:
                mejores = media_c
            resultados.append((0.5 * media_c + 0.5 * mejores, slug))
        resultados.sort(reverse=True)
        return resultados

    sims = consulta @ INDICE["vectores"].T            # (trozos consulta, trozos índice)
    mejor_por_trozo = sims.mean(axis=0)               # parecido medio a cada trozo del índice
    etiquetas = INDICE["etiquetas"]
    resultados = []
    for slug in np.unique(etiquetas):
        s = np.sort(mejor_por_trozo[etiquetas == slug])[::-1]
        resultados.append((float(s[:3].mean()), slug))  # media de sus 3 trozos más parecidos
    resultados.sort(reverse=True)
    return resultados


def puntuar(consulta: np.ndarray, top: int = 3) -> list[dict]:
    resultados = _ranking_completo(consulta)
    return [
        {"slug": slug, "parecido": round(score, 3), **INDICE["fichas"][slug]}
        for score, slug in resultados[:top]
    ]


@app.get("/api/salud")
def salud():
    return {
        "ok": bool(INDICE),
        "modo_indice": INDICE.get("modo"),
        "info_modelo": INDICE.get("info_modelo"),
    }


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
    ranking = _ranking_completo(consulta)
    puntuaciones = np.array([s for s, _ in ranking], dtype=np.float32)
    slugs = [slug for _, slug in ranking]
    decision = clasificar_consulta(
        puntuaciones_ordenadas=puntuaciones,
        candidatos_ordenados=slugs,
        umbrales=UMBRALES_VOXBLINK_R100_CALIBRADOS,
        n_galeria_total=len(slugs),
        con_contexto=False,
    )
    top_res = [
        {"slug": slug, "parecido": round(score, 3), **INDICE["fichas"][slug]}
        for score, slug in ranking[:3]
    ]
    return {
        "estado": decision.estado.value,
        "motivo": decision.motivo,
        "porcentaje": decision.porcentaje_interfaz_actual,
        "resultados": top_res,
    }


app.mount("/", StaticFiles(directory=RAIZ / "frontend", html=True), name="web")
