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
from aumento_acustico import DetectorVoz, aislar_voz_continua, normalizar_amplitud_pico  # noqa: E402
from filmografia import filtrar_papeles_actor, nombre_original, normalizar_texto as norm_texto_filmo  # noqa: E402
from lda import proyectar  # noqa: E402
from metadatos_modelo import validar_artefactos_web, validar_compatibilidad_indice  # noqa: E402
from rechazo import (  # noqa: E402
    UMBRALES_VOXBLINK_R100_CALIBRADOS,
    aplicar_as_norm,
    calcular_estadisticas_cohorte,
    clasificar_consulta,
    porcentaje_calibrado_r100,
)
from voz import ONNX, SAMPLE_RATE, cargar_audio, huellas, info_modelo_actual, modelo, separador, sesion_onnx  # noqa: E402

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
        INDICE["idx_por_slug"] = {s: i for i, s in enumerate(slugs)}
        INDICE["fichas"] = {a["id"]: a for a in actores_lista}
        INDICE["mu_a"], INDICE["sd_a"] = calcular_estadisticas_cohorte(INDICE["centroides"], k=50)
        INDICE["dir_web"] = dir_web
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
        t_pad = f" {t_norm} "
        if all(f" {p}" in t_pad for p in palabras):
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
    # 2. Búsqueda por palabras coincidentes (inicio de palabra)
    acumulados = set()
    primer_texto, primer_tipo = None, None
    for tipo, texto, actor_indices in entradas:
        t_pad = f" {normalizar_texto(texto)} "
        if all(f" {p}" in t_pad for p in palabras):
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
        n_total = len(slugs)
        crudos = np.empty(n_total, dtype=np.float32)
        for i in range(n_total):
            ini, fin = int(inicio[i]), int(inicio[i + 1])
            media_c = float(sims_cent[i])
            if fin > ini:
                s = np.sort(sims_troz[ini:fin])[::-1]
                mejores = float(s[:4].mean())
            else:
                mejores = media_c
            crudos[i] = 0.5 * (media_c + mejores)
        mu_a = INDICE.get("mu_a")
        sd_a = INDICE.get("sd_a")
        ajustados = aplicar_as_norm(crudos, mu_a=mu_a, sd_a=sd_a, k=50, peso=0.04)
        indices = range(n_total) if candidatos_indices is None else [i for i in range(n_total) if i in candidatos_indices]
        resultados = [(float(ajustados[i]), slugs[i]) for i in indices]
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


def _cargar_papeles_slug(slug: str) -> list:
    idx = (INDICE.get("idx_por_slug") or {}).get(slug)
    dir_web = INDICE.get("dir_web")
    if idx is not None and dir_web is not None:
        p = Path(dir_web) / "papeles" / f"{idx}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    ficha = (INDICE.get("fichas") or {}).get(slug, {})
    filas = []
    for t in ficha.get("trabajos", []):
        m = re.match(r"^(.*?) en (.+?)(?: \((\d{4})\))?$", str(t))
        if m:
            filas.append([m.group(2), m.group(1), int(m.group(3)) if m.group(3) else None, ""])
        else:
            filas.append([str(t), "", None, ""])
    return filas


def _papel_para_pista(slug: str, pista: str | None) -> dict | None:
    if not pista:
        return None
    q_norm = normalizar_texto(pista)
    palabras = q_norm.split()
    if not palabras:
        return None
    filas = _cargar_papeles_slug(slug)
    coincidencias = []
    for f in filas:
        obra, pers, anio, orig = f[0], f[1], f[2], f[3]
        texto = f" {normalizar_texto(f'{obra} {pers} {nombre_original(str(orig))}')} "
        if all(f" {w}" in texto for w in palabras):
            coincidencias.append((anio or 0, obra, pers, anio, nombre_original(str(orig))))
    if not coincidencias:
        return None
    coincidencias.sort(reverse=True)
    _, obra, pers, anio, orig = coincidencias[0]
    return {"obra": obra, "personaje": pers, "anio": anio, "actor_original": orig}


@app.get("/api/actor/{actor_id}")
def detalle_actor(actor_id: str, tipo: str = "", q: str = "", limite: int = 60, offset: int = 0):
    """Devuelve ficha ampliada y filmografía completa filtrable de un doblador."""
    if not INDICE:
        raise HTTPException(503, "Aún no hay actores indexados")
    ficha = (INDICE.get("fichas") or {}).get(actor_id)
    if ficha is None:
        raise HTTPException(404, "Doblador no encontrado")
    filas = _cargar_papeles_slug(actor_id)
    filmografia = filtrar_papeles_actor(filas, tipo=tipo, q=q, limite=max(1, min(limite, 200)), offset=max(0, offset))
    return {
        "id": actor_id,
        **ficha,
        "filmografia": filmografia,
    }


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
            senal = cargar_audio(tmp_name, separar=INDICE["separar"], max_seg=20)
        except Exception:
            raise HTTPException(400, "No se pudo leer el audio")
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
    if len(senal) < 2 * SAMPLE_RATE:
        raise HTTPException(400, "Audio demasiado corto: graba al menos 2 segundos")
    vad = DetectorVoz(sr=SAMPLE_RATE, min_voz_util_seg=1.5).analizar(senal)
    if vad.es_toma_valida and vad.segmentos:
        senal_limpia = normalizar_amplitud_pico(aislar_voz_continua(senal, vad=vad, sr=SAMPLE_RATE))
        consulta = huellas(senal_limpia)
        if len(consulta) == 0:
            consulta = huellas(normalizar_amplitud_pico(senal))
    else:
        consulta = huellas(senal)
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
    top_res = []
    for idx_r, (score, slug) in enumerate(ranking[:3]):
        item = {"slug": slug, "parecido": round(score, 3), "porcentaje": porcentaje_calibrado_r100(score), **INDICE["fichas"][slug]}
        if idx_r == 0 and pista:
            item["papel_pista"] = _papel_para_pista(slug, texto_coincidente or pista)
        top_res.append(item)
    resp = {
        "estado": decision.estado.value,
        "motivo": decision.motivo,
        "porcentaje": porcentaje_calibrado_r100(decision.puntuacion_principal),
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
