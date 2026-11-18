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
from aumento_acustico import (  # noqa: E402
    DetectorVoz,
    aislar_voz_continua,
    estimar_f0_mediana,
    normalizar_amplitud_pico,
    realzar_voz_lejana_y_atenuar_fondo,
    segmentar_intervenciones,
)
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
        INDICE["dobladores_norm"] = [
            (a["id"], a["nombre"], normalizar_texto(a["nombre"]), f" {normalizar_texto(a['nombre'])} ", i)
            for i, a in enumerate(actores_lista)
        ]
        if (dir_web / "busqueda.json").exists():
            try:
                raw_busqueda = json.loads((dir_web / "busqueda.json").read_text(encoding="utf-8"))
                INDICE["busqueda"] = raw_busqueda
                INDICE["busqueda_norm"] = [
                    (tipo, texto, t_n := normalizar_texto(texto), f" {t_n} ", actor_indices)
                    for tipo, texto, actor_indices in raw_busqueda
                ]
            except Exception:
                INDICE["busqueda"] = []
                INDICE["busqueda_norm"] = []
        else:
            INDICE["busqueda"] = []
            INDICE["busqueda_norm"] = []


def normalizar_texto(t: str) -> str:
    s = unicodedata.normalize("NFKD", t)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def sugerencias_busqueda(q: str, max_resultados: int = 10) -> list[dict]:
    entradas = INDICE.get("busqueda_norm")
    dobladores = INDICE.get("dobladores_norm") or []
    if (entradas is None and not INDICE.get("busqueda") and not dobladores) or not q:
        return []
    q_norm = normalizar_texto(q)
    palabras = q_norm.split()
    if not palabras:
        return []
    exactas, resto = [], []
    if entradas is not None:
        for tipo, texto, t_norm, t_pad, actor_indices in entradas:
            if all(f" {p}" in t_pad for p in palabras):
                item = {"tipo": tipo, "texto": texto, "num_voces": len(actor_indices)}
                (exactas if t_norm == q_norm else resto).append(item)
                if len(exactas) + len(resto) >= 50:
                    break
    else:
        for tipo, texto, actor_indices in INDICE.get("busqueda") or []:
            t_norm = normalizar_texto(texto)
            if all(f" {p}" in f" {t_norm} " for p in palabras):
                item = {"tipo": tipo, "texto": texto, "num_voces": len(actor_indices)}
                (exactas if t_norm == q_norm else resto).append(item)
                if len(exactas) + len(resto) >= 50:
                    break
    for slug, nombre, d_norm, d_pad, _idx in dobladores:
        if all(f" {p}" in d_pad for p in palabras):
            item = {"tipo": "doblador", "texto": nombre, "num_voces": 1, "actor_id": slug}
            (exactas if d_norm == q_norm else resto).append(item)
            if len(exactas) + len(resto) >= 60:
                break
    return (exactas + resto)[:max_resultados]


def _resolver_pista_individual(pista: str) -> tuple[set[int], str | None, str | None]:
    entradas = INDICE.get("busqueda_norm")
    dobladores = INDICE.get("dobladores_norm") or []
    if (entradas is None and not INDICE.get("busqueda") and not dobladores) or not pista:
        return set(), None, None
    q_norm = normalizar_texto(pista)
    palabras = q_norm.split()
    if not palabras:
        return set(), None, None
    if entradas is not None:
        for tipo, texto, t_norm, _t_pad, actor_indices in entradas:
            if t_norm == q_norm:
                return set(actor_indices), texto, tipo
        acumulados = set()
        primer_texto, primer_tipo = None, None
        for tipo, texto, _t_norm, t_pad, actor_indices in entradas:
            if all(f" {p}" in t_pad for p in palabras):
                if primer_texto is None:
                    primer_texto, primer_tipo = texto, tipo
                acumulados.update(actor_indices)
                if len(acumulados) >= 400:
                    break
        if acumulados:
            return acumulados, primer_texto, primer_tipo
    else:
        raw = INDICE.get("busqueda") or []
        for tipo, texto, actor_indices in raw:
            if normalizar_texto(texto) == q_norm:
                return set(actor_indices), texto, tipo
        acumulados = set()
        primer_texto, primer_tipo = None, None
        for tipo, texto, actor_indices in raw:
            if all(f" {p}" in f" {normalizar_texto(texto)} " for p in palabras):
                if primer_texto is None:
                    primer_texto, primer_tipo = texto, tipo
                acumulados.update(actor_indices)
                if len(acumulados) >= 400:
                    break
        if acumulados:
            return acumulados, primer_texto, primer_tipo

    # Respaldo: coincidencia directa por nombre de doblador
    for _slug, nombre, d_norm, d_pad, idx_actor in dobladores:
        if d_norm == q_norm or all(f" {p}" in d_pad for p in palabras):
            return {idx_actor}, nombre, "doblador"
    return set(), None, None


def resolver_pista(pista: str) -> tuple[set[int], str | None, str | None]:
    """Resuelve una pista simple o la intersección de varias pistas separadas por '+', ';' o '&'."""
    if not pista:
        return set(), None, None
    partes = [p.strip() for p in re.split(r"[+;&]+", pista) if p.strip()]
    if len(partes) <= 1:
        return _resolver_pista_individual(pista)
    resueltas = []
    for sub in partes:
        ids, txt, tpo = _resolver_pista_individual(sub)
        if ids and txt:
            resueltas.append((ids, txt, tpo))
    if not resueltas:
        return set(), None, None
    if len(resueltas) == 1:
        return resueltas[0]
    conjuntos = [r[0] for r in resueltas]
    textos = [r[1] for r in resueltas]
    inter = set.intersection(*conjuntos)
    if inter:
        return inter, " + ".join(textos), "interseccion"
    return set.union(*conjuntos), " + ".join(textos), "union"


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


PESOS_RANGO_TOP4 = np.array([0.34, 0.27, 0.21, 0.18], dtype=np.float32)


def _ponderar_ventanas_consulta(qs: np.ndarray) -> np.ndarray:
    """Pondera ventanas de la consulta según su coherencia mutua para atenuar ruidos o cortes atípicos."""
    n = len(qs)
    if n < 3:
        return np.ones(n, dtype=np.float32) / max(n, 1)
    S = qs @ qs.T
    coh = (S.sum(axis=1) - np.diag(S)) / float(n - 1)
    pesos = np.where(coh >= 0.35, 1.0, 0.45).astype(np.float32)
    return (pesos / max(float(pesos.sum()), 1e-9)).astype(np.float32)


def _penalizacion_f0(f0_consulta: float | None, f0_actor: float | None) -> float:
    """Penalización suave graduada ante contradicción de registro vocal (grave vs agudo)."""
    if f0_consulta is None or f0_actor is None:
        return 0.0
    if (f0_consulta < 118.0 and f0_actor > 215.0) or (f0_consulta > 215.0 and f0_actor < 118.0):
        return 0.035
    if (f0_consulta < 130.0 and f0_actor >= 210.0) or (f0_consulta > 190.0 and f0_actor <= 115.0):
        return 0.020
    return 0.0


def _ranking_completo(
    consulta: np.ndarray,
    candidatos_indices: set[int] | None = None,
    f0_consulta: float | None = None,
) -> list[tuple[float, str]]:
    if INDICE.get("modo") == "web":
        qs = proyectar(consulta, INDICE["lda_mu"], INDICE["lda_W"])
        pesos_q = _ponderar_ventanas_consulta(qs)
        q_pond = (pesos_q[:, None] * qs).sum(axis=0)
        sims_cent = q_pond @ INDICE["centroides"].T
        sims_troz = q_pond @ INDICE["vectores"].T
        inicio = INDICE["inicio"]
        slugs = INDICE["slugs"]
        fichas = INDICE["fichas"]
        n_total = len(slugs)
        crudos = np.empty(n_total, dtype=np.float32)
        max_trozos = np.empty(n_total, dtype=np.float32)
        for i in range(n_total):
            ini, fin = int(inicio[i]), int(inicio[i + 1])
            media_c = float(sims_cent[i])
            if fin > ini:
                s = np.sort(sims_troz[ini:fin])[::-1][:4]
                w = PESOS_RANGO_TOP4[: len(s)]
                mejores = float(np.dot(s, w) / max(float(w.sum()), 1e-9))
                max_trozos[i] = float(s[0])
            else:
                mejores = media_c
                max_trozos[i] = media_c
            penal = _penalizacion_f0(f0_consulta, fichas[slugs[i]].get("f0"))
            crudos[i] = 0.5 * (media_c + mejores) - penal
        mu_a = INDICE.get("mu_a")
        sd_a = INDICE.get("sd_a")
        ajustados = aplicar_as_norm(crudos, mu_a=mu_a, sd_a=sd_a, k=50, peso=0.04)
        indices = range(n_total) if candidatos_indices is None else [i for i in range(n_total) if i in candidatos_indices]
        resultados = [(float(ajustados[i]), slugs[i]) for i in indices]
        resultados.sort(reverse=True)
        if len(resultados) >= 2 and (resultados[0][0] - resultados[1][0]) < 0.035:
            s1, slug1 = resultados[0]
            s2, slug2 = resultados[1]
            idx_map = INDICE.get("idx_por_slug") or {}
            i1, i2 = idx_map.get(slug1), idx_map.get(slug2)
            if i1 is not None and i2 is not None:
                diff_max = 0.035 * float(max_trozos[i2] - max_trozos[i1])
                bono_vec = 0.0
                C_gal = INDICE.get("centroides")
                if C_gal is not None and len(resultados) >= 4:
                    vecs_idx = [idx_map[sl] for _, sl in resultados[2 : min(10, len(resultados))] if sl in idx_map]
                    if vecs_idx:
                        C_vec = C_gal[vecs_idx]
                        sim_v1 = float((C_gal[i1] @ C_vec.T).mean())
                        sim_v2 = float((C_gal[i2] @ C_vec.T).mean())
                        bono_vec = 0.025 * (sim_v2 - sim_v1)
                if s2 + diff_max + bono_vec > s1:
                    resultados[0] = (max(s1, s2 + 0.001), slug2)
                    resultados[1] = (min(s2, s1 - 0.001), slug1)
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
    candidatos_indices, texto_coincidente, tipo_pista = resolver_pista(pista) if pista else (set(), None, None)
    con_contexto = bool(candidatos_indices and "slugs" in INDICE and len(candidatos_indices) < len(INDICE["slugs"]))
    cands_set = candidatos_indices if con_contexto else None

    vad = DetectorVoz(sr=SAMPLE_RATE, min_voz_util_seg=1.5).analizar(senal)
    dialogo_info = None
    f0_consulta = None
    if vad.es_toma_valida and vad.segmentos:
        # Diarización ligera por similitud mutua entre turnos en espacio LDA
        turnos = segmentar_intervenciones(senal, vad=vad, sr=SAMPLE_RATE, min_duracion_turno_seg=2.0)
        if len(turnos) >= 2 and INDICE.get("modo") == "web":
            t_vecs, t_proys, t_durs = [], [], []
            for tr in turnos[:4]:
                ht = huellas(normalizar_amplitud_pico(realzar_voz_lejana_y_atenuar_fondo(tr, sr=SAMPLE_RATE)))
                if len(ht):
                    pt = proyectar(ht, INDICE["lda_mu"], INDICE["lda_W"]).mean(axis=0)
                    pt /= max(float(np.linalg.norm(pt)), 1e-9)
                    t_vecs.append(ht)
                    t_proys.append(pt)
                    t_durs.append(len(tr) / float(SAMPLE_RATE))
            if len(t_proys) >= 2:
                sims_mut = np.stack(t_proys) @ np.stack(t_proys).T
                i_min, j_min = np.unravel_index(int(np.argmin(sims_mut)), sims_mut.shape)
                if float(sims_mut[i_min, j_min]) < 0.45:
                    piv1, piv2 = t_proys[int(i_min)], t_proys[int(j_min)]
                    g1_idx = [k for k, p in enumerate(t_proys) if float(p @ piv1) >= float(p @ piv2)]
                    g2_idx = [k for k in range(len(t_proys)) if k not in g1_idx]
                    if g1_idx and g2_idx:
                        dur1 = sum(t_durs[k] for k in g1_idx)
                        dur2 = sum(t_durs[k] for k in g2_idx)
                        c1 = np.vstack([t_vecs[k] for k in g1_idx])
                        c2 = np.vstack([t_vecs[k] for k in g2_idx])
                        f0_1 = estimar_f0_mediana(np.concatenate([turnos[k] for k in g1_idx]), sr=SAMPLE_RATE)
                        f0_2 = estimar_f0_mediana(np.concatenate([turnos[k] for k in g2_idx]), sr=SAMPLE_RATE)
                        r1 = _ranking_completo(c1, candidatos_indices=cands_set, f0_consulta=f0_1)
                        r2 = _ranking_completo(c2, candidatos_indices=cands_set, f0_consulta=f0_2)
                        if r1 and r2 and r1[0][1] != r2[0][1]:
                            consulta = c1 if dur1 >= dur2 else c2
                            dialogo_info = {
                                "hablante_1": {"slug": r1[0][1], "nombre": INDICE["fichas"][r1[0][1]]["nombre"], "duracion_seg": round(dur1, 1), "porcentaje": porcentaje_calibrado_r100(r1[0][0])},
                                "hablante_2": {"slug": r2[0][1], "nombre": INDICE["fichas"][r2[0][1]]["nombre"], "duracion_seg": round(dur2, 1), "porcentaje": porcentaje_calibrado_r100(r2[0][0])},
                            }
                            f0_consulta = f0_1 if dur1 >= dur2 else f0_2

        if dialogo_info is None:
            voz_cont = realzar_voz_lejana_y_atenuar_fondo(aislar_voz_continua(senal, vad=vad, sr=SAMPLE_RATE), sr=SAMPLE_RATE)
            senal_limpia = normalizar_amplitud_pico(voz_cont)
            f0_consulta = estimar_f0_mediana(senal_limpia, sr=SAMPLE_RATE)
            consulta = huellas(senal_limpia)
            if len(consulta) == 0:
                consulta = huellas(normalizar_amplitud_pico(senal))
    else:
        consulta = huellas(senal)
    if len(consulta) == 0:
        raise HTTPException(400, "Audio demasiado corto: graba al menos 2 segundos")

    ranking = _ranking_completo(consulta, candidatos_indices=cands_set, f0_consulta=f0_consulta)
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
        "diagnostico": {
            "duracion_voz_seg": round(float(vad.duracion_voz_seg), 2),
            "num_segmentos": len(vad.segmentos),
            "f0_hz": round(float(f0_consulta), 1) if f0_consulta is not None else None,
        },
    }
    if con_contexto:
        resp["contexto"] = {
            "pista": pista,
            "coincidencia": texto_coincidente,
            "tipo": tipo_pista,
            "num_candidatos": len(candidatos_indices),
        }
    if dialogo_info:
        resp["dialogo"] = dialogo_info
    if INDICE.get("modo") == "web" and len(ranking) >= 2 and (ranking[0][0] - ranking[1][0]) < 0.065:
        idx_map = INDICE.get("idx_por_slug") or {}
        i1, i2 = idx_map.get(ranking[0][1]), idx_map.get(ranking[1][1])
        if i1 is not None and i2 is not None:
            sim_gem = float(INDICE["centroides"][i1] @ INDICE["centroides"][i2])
            if sim_gem >= 0.55:
                resp["gemelo_vocal"] = {
                    "slug": ranking[1][1],
                    "nombre": INDICE["fichas"][ranking[1][1]]["nombre"],
                    "similitud_centroides": round(sim_gem, 3),
                }
    return resp


app.mount("/", StaticFiles(directory=RAIZ / "frontend", html=True), name="web")
