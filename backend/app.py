"""Servidor: recibe unos segundos de audio y dice qué doblador es.

Arranque: .venv/bin/uvicorn backend.app:app --port 8000
"""
from contextlib import asynccontextmanager
import json
import os
import re
import sys
import tempfile
import unicodedata
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
        if (dir_web / "busqueda.json").exists():
            try:
                INDICE["busqueda"] = json.loads((dir_web / "busqueda.json").read_text(encoding="utf-8"))
            except Exception:
                INDICE["busqueda"] = []


def normalizar_texto(t: str) -> str:
    s = unicodedata.normalize("NFKD", t)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def sugerencias_busqueda(q: str, max_resultados: int = 10) -> list[dict]:
    entradas = INDICE.get("busqueda") or []
    if not entradas or not q:
        return []
    q_norm = normalizar_texto(q)
    palabras = q_norm.split()
    if not palabras:
        return []
    exactas, resto = [], []
    for tipo, texto, actor_indices in entradas:
        t_norm = normalizar_texto(texto)
        if all(p in t_norm for p in palabras):
            item = {"tipo": tipo, "texto": texto, "num_voces": len(actor_indices)}
            if t_norm == q_norm:
                exactas.append(item)
            else:
                resto.append(item)
            if len(exactas) + len(resto) >= 50:
                break
    return (exactas + resto)[:max_resultados]


def resolver_pista(pista: str) -> tuple[set[int], str | None, str | None]:
    entradas = INDICE.get("busqueda") or []
    if not entradas or not pista:
        return set(), None, None
    q_norm = normalizar_texto(pista)
    palabras = q_norm.split()
    if not palabras:
        return set(), None, None
    # 1. Búsqueda exacta
    for tipo, texto, actor_indices in entradas:
        if normalizar_texto(texto) == q_norm:
            return set(actor_indices), texto, tipo
    # 2. Búsqueda por palabras coincidentes
    acumulados = set()
    primer_texto, primer_tipo = None, None
    for tipo, texto, actor_indices in entradas:
        t_norm = normalizar_texto(texto)
        if all(p in t_norm for p in palabras):
            if primer_texto is None:
                primer_texto, primer_tipo = texto, tipo
            acumulados.update(actor_indices)
            if len(acumulados) >= 400:
                break
    return acumulados, primer_texto, primer_tipo


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


def _ranking_completo(consulta: np.ndarray, candidatos_indices: set[int] | None = None) -> list[tuple[float, str]]:
    if INDICE.get("modo") == "web":
        qs = proyectar(consulta, INDICE["lda_mu"], INDICE["lda_W"])
        sims_cent = (qs @ INDICE["centroides"].T).mean(axis=0)
        sims_troz = (qs @ INDICE["vectores"].T).mean(axis=0)
        inicio = INDICE["inicio"]
        slugs = INDICE["slugs"]
        beta = float(UMBRALES_VOXBLINK_R100_CALIBRADOS.beta_exceso_trozos)
        indices = [i for i in range(len(slugs)) if candidatos_indices is None or i in candidatos_indices]
        resultados = []
        for i in indices:
            ini, fin = int(inicio[i]), int(inicio[i + 1])
            media_c = float(sims_cent[i])
            if fin > ini:
                s = np.sort(sims_troz[ini:fin])[::-1]
                mejores = float(s[:3].mean())
                n_trozos = fin - ini
            else:
                mejores = media_c
                n_trozos = 1
            penal = beta * max(0.0, float(np.log(max(n_trozos, 1) / 12.0))) if beta > 0 else 0.0
            resultados.append((0.5 * (media_c + mejores - penal), slugs[i]))
        resultados.sort(reverse=True)
        return resultados

    sims = consulta @ INDICE["vectores"].T            # (trozos consulta, trozos índice)
    mejor_por_trozo = sims.mean(axis=0)               # parecido medio a cada trozo del índice
    etiquetas = INDICE["etiquetas"]
    slugs_unicos = np.unique(etiquetas)
    if candidatos_indices is not None and "slugs" in INDICE:
        candidatos_slugs = {INDICE["slugs"][i] for i in candidatos_indices if i < len(INDICE["slugs"])}
        slugs_unicos = [s for s in slugs_unicos if s in candidatos_slugs]
    resultados = []
    for slug in slugs_unicos:
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
        "total_actores": len(INDICE.get("slugs", [])),
        "total_entradas_busqueda": len(INDICE.get("busqueda", [])),
    }


@app.get("/api/buscar")
def buscar(q: str = "", max_resultados: int = 10):
    """Devuelve sugerencias de obras, personajes y actores originales ('¿Qué estás viendo?')."""
    if not q or len(q.strip()) < 2:
        return []
    return sugerencias_busqueda(q, max_resultados=max_resultados)


@app.post("/api/identificar")
async def identificar(audio: UploadFile, pista: str | None = None):
    if not INDICE:
        raise HTTPException(503, "Aún no hay actores indexados")
    sufijo = Path(audio.filename or "a.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_name = tmp.name
    try:
        try:
            consulta = huellas(cargar_audio(tmp_name, separar=INDICE["separar"], max_seg=20))
        except Exception:
            raise HTTPException(400, "No se pudo leer el audio")
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
    if len(consulta) == 0:
        raise HTTPException(400, "Audio demasiado corto: graba al menos 2 segundos")

    candidatos_indices, texto_coincidente, tipo_pista = resolver_pista(pista) if pista else (set(), None, None)
    con_contexto = bool(candidatos_indices and "slugs" in INDICE and len(candidatos_indices) < len(INDICE["slugs"]))
    ranking = _ranking_completo(consulta, candidatos_indices=candidatos_indices if con_contexto else None)
    puntuaciones = np.array([s for s, _ in ranking], dtype=np.float32)
    slugs = [slug for _, slug in ranking]
    n_galeria_total = len(INDICE.get("slugs", slugs))
    decision = clasificar_consulta(
        puntuaciones=puntuaciones,
        ids_candidatos=slugs,
        umbrales=UMBRALES_VOXBLINK_R100_CALIBRADOS,
        n_galeria_total=n_galeria_total,
        con_contexto=con_contexto,
    )
    top_res = [
        {"slug": slug, "parecido": round(score, 3), **INDICE["fichas"][slug]}
        for score, slug in ranking[:3]
    ]
    resp = {
        "estado": decision.estado.value,
        "motivo": decision.motivo,
        "porcentaje": decision.porcentaje_interfaz_actual,
        "resultados": top_res,
    }
    if con_contexto:
        resp["contexto"] = {
            "pista": pista,
            "coincidencia": texto_coincidente,
            "tipo": tipo_pista,
            "num_candidatos": len(candidatos_indices),
        }
    return resp


app.mount("/", StaticFiles(directory=RAIZ / "frontend", html=True), name="web")
