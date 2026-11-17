"""Enriquecimiento incremental de voces, filmografías y calidad del índice web (`frontend/datos/`).

Permite mejorar Estavoz directamente con fuentes públicas verificadas sin necesitar `data/indice.npz`:
  1. Podar trozos intrusos existentes (`filtrar_trozos_intrusos`, `umbral_outlier=0.20`).
  2. Hidratar y reparar filmografías completas (`papeles/*.json`, corrigiendo filas `(VOZ)` en eldoblaje.com),
     recalcular papeles destacados multi-formato (`pelicula`, `serie`, `animacion`, `videojuego`),
     voces originales habituales (`vozh`), ruta exacta de muestra oficial (`muestra`) y `busqueda.json`.
  3. Incorporar nuevas muestras oficiales de `eldoblaje.com` (actores nuevos de `buscavoces` y muestras
     extra `Saludos`/`Muestras3`/`Muestras4` para actores con poca cobertura).
  4. Extraer fragmentos limpios de podcasts de doblaje (`La Vermella`, `La voz de tu vida`, `VozCast`,
     `Dóblame mucho`, `EL ÚLTIMO TAKE`, `En sincronía`, `CharletaCast`, `SOY TU FAN`, `La Hora Pokémon`)
     con VAD + segmentación de turnos + exclusión automática de presentadores, tanto para reforzar
     actores con pocos trozos como para dar de alta dobladores famosos sin muestra previa.

Uso:
  .venv/bin/python backend/incorporar_voces.py --todo
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import requests
from scipy.cluster.hierarchy import fcluster, linkage

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "scraper"))

from auditar_indice import filtrar_trozos_intrusos  # noqa: E402
from aumento_acustico import DetectorVoz, segmentar_intervenciones, simular_sala_tv  # noqa: E402
import eldoblaje  # noqa: E402
from filmografia import (  # noqa: E402
    nombre_original,
    normalizar_texto,
    personaje_base,
    resumen_actor_desde_papeles,
    titulo_base,
    titulo_bonito,
)
from lda import normaliza, proyectar  # noqa: E402
from metadatos_modelo import validar_artefactos_web  # noqa: E402
from simular_movil import movil as simular_movil_sintetico  # noqa: E402
from voz import SAMPLE_RATE, cargar_audio_bytes, huellas, huella_trozo  # noqa: E402

DATOS_WEB = RAIZ / "frontend" / "datos"
FOTOS_WEB = RAIZ / "frontend" / "fotos"
RUTA_ONNX = RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"
CACHE_DIR = RAIZ / "data" / "incorporacion"
REGISTRO_PATH = CACHE_DIR / "registro.json"
HF_SPACE_BASE = "https://srharrison-estavoz.static.hf.space"

FEEDS_PODCAST: dict[str, dict[str, Any]] = {
    "La Vermella": {
        "url": "https://dinamics.3cat.cat/public/podcast/catradio/xml/9/9/podprograma1999.xml",
        "skip_seg": 180,
    },
    "La voz de tu vida": {
        "url": "https://anchor.fm/s/2b183bf4/podcast/rss",
        "skip_seg": 45,
    },
    "VozCast": {
        "url": "https://anchor.fm/s/efd39b64/podcast/rss",
        "skip_seg": 60,
    },
    "Dóblame mucho": {
        "url": "https://feeds.ivoox.com/feed_fg_f11460317_filtro_1.xml",
        "skip_seg": 45,
    },
    "EL ÚLTIMO TAKE": {
        "url": "https://feeds.ivoox.com/feed_fg_f1870633_filtro_1.xml",
        "skip_seg": 45,
    },
    "En sincronía": {
        "url": "https://rss.buzzsprout.com/1112315.rss",
        "skip_seg": 45,
    },
    "CharletaCast": {
        "url": "https://anchor.fm/s/10f89d5cc/podcast/rss",
        "skip_seg": 45,
    },
    "SOY TU FAN": {
        "url": "https://feeds.ivoox.com/feed_fg_f11104200_filtro_1.xml",
        "skip_seg": 45,
    },
    "La Hora Pokémon": {
        "url": "https://feeds.ivoox.com/feed_fg_f1728172_filtro_1.xml",
        "skip_seg": 45,
    },
}


def cargar_estado_web(carpeta_datos: Path = DATOS_WEB) -> dict[str, Any]:
    meta = json.loads((carpeta_datos / "actores.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    actores = meta["actores"]
    nA = len(actores)

    lda_raw = np.fromfile(carpeta_datos / "lda.bin", dtype="<f4")
    mu, W = lda_raw[:dim].astype(np.float32), lda_raw[dim:].reshape(dim, dim).astype(np.float32)
    C = np.fromfile(carpeta_datos / "centroides.bin", dtype="<f4").reshape(nA, dim).astype(np.float32)

    raw_trozos = (carpeta_datos / "trozos.bin").read_bytes()
    inicio = np.frombuffer(raw_trozos, dtype="<u4", count=nA + 1)
    n_total = int(inicio[-1])
    escala = np.frombuffer(raw_trozos, dtype="<f4", count=n_total, offset=(nA + 1) * 4)
    q8 = np.frombuffer(raw_trozos, dtype=np.int8, count=n_total * dim, offset=(nA + 1 + n_total) * 4).reshape(n_total, dim)
    V = normaliza((q8.astype(np.float32) * escala[:, None]).astype(np.float32))
    trozos_por_actor = [V[int(inicio[i]) : int(inicio[i + 1])].copy() for i in range(nA)]

    return {
        "meta": meta,
        "dim": dim,
        "mu": mu,
        "W": W,
        "C": C,
        "actores": actores,
        "trozos_por_actor": trozos_por_actor,
    }


def guardar_estado_web(estado: dict[str, Any], carpeta_datos: Path = DATOS_WEB) -> dict[str, Any]:
    actores = estado["actores"]
    dim = int(estado["dim"])
    nA = len(actores)
    C = normaliza(np.asarray(estado["C"], dtype=np.float32)).astype("<f4")
    trozos_lista = estado["trozos_por_actor"]

    assert len(C) == nA and len(trozos_lista) == nA
    inicio = np.zeros(nA + 1, dtype="<u4")
    bloques = []
    for i, t in enumerate(trozos_lista):
        t_arr = np.asarray(t, dtype=np.float32)
        if len(t_arr) == 0:
            t_arr = C[i : i + 1].copy()
        inicio[i + 1] = inicio[i] + len(t_arr)
        bloques.append(t_arr)

    V_all = np.vstack(bloques).astype(np.float32)
    escala = (np.abs(V_all).max(axis=1) / 127.0).astype("<f4")
    escala = np.where(escala > 0, escala, np.float32(1.0 / 127.0)).astype("<f4")
    q8 = np.round(V_all / escala[:, None]).clip(-127, 127).astype(np.int8)

    C.tofile(carpeta_datos / "centroides.bin")
    with open(carpeta_datos / "trozos.bin", "wb") as f:
        f.write(inicio.tobytes())
        f.write(escala.tobytes())
        f.write(q8.tobytes())

    from dataclasses import asdict
    from rechazo import UMBRALES_VOXBLINK_R100_CALIBRADOS

    meta = dict(estado["meta"])
    meta["actores"] = actores
    meta["umbrales_rechazo"] = asdict(UMBRALES_VOXBLINK_R100_CALIBRADOS)
    (carpeta_datos / "actores.json").write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return validar_artefactos_web(carpeta_datos, ruta_onnx=RUTA_ONNX if RUTA_ONNX.exists() else None)


def podar_intrusos_estado(estado: dict[str, Any], umbral_outlier: float = 0.20, min_trozos: int = 4) -> int:
    """Elimina trozos de otros interlocutores o música (`sim < umbral_outlier`) de `trozos_por_actor`."""
    actores = estado["actores"]
    slugs = [a["id"] for a in actores]
    X_list, y_list = [], []
    for a, t in zip(slugs, estado["trozos_por_actor"]):
        if len(t):
            X_list.append(t)
            y_list.extend([a] * len(t))
    if not X_list:
        return 0
    X = np.vstack(X_list)
    y = np.array(y_list)
    C = normaliza(estado["C"])
    conservar, intrusos = filtrar_trozos_intrusos(
        X, y, C, slugs, umbral_outlier=umbral_outlier, min_trozos_por_actor=min_trozos
    )
    if not intrusos:
        return 0
    pos = 0
    for i, t in enumerate(estado["trozos_por_actor"]):
        k = len(t)
        m = conservar[pos : pos + k]
        estado["trozos_por_actor"][i] = t[m]
        pos += k
        if len(estado["trozos_por_actor"][i]) > 0:
            # Ajuste suave hacia los trozos limpios conservando 50% de estabilidad del centroide previo
            nuevo_m = normaliza(estado["trozos_por_actor"][i].mean(axis=0))
            estado["C"][i] = normaliza(0.5 * C[i] + 0.5 * nuevo_m)
    return len(intrusos)


def _cargar_registro() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRO_PATH.exists():
        return json.loads(REGISTRO_PATH.read_text(encoding="utf-8"))
    return {"clips_hechos": [], "podcasts_hechos": [], "hosts_podcast": {}}


def _guardar_registro(reg: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRO_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _embeddings_de_audio(audio: np.ndarray, mu: np.ndarray, W: np.ndarray, semilla: int, con_aumentos: bool = True) -> np.ndarray:
    """Extrae huellas limpias + `#movil` + `#sala` y las proyecta al espacio LDA."""
    audio = np.asarray(audio, dtype=np.float32)
    h_limpio = huellas(audio)
    if len(h_limpio) == 0:
        return np.zeros((0, len(mu)), dtype=np.float32)
    partes = [h_limpio]
    if con_aumentos:
        rng = np.random.default_rng(semilla)
        h_mov = huellas(simular_movil_sintetico(audio, rng))
        if len(h_mov):
            partes.append(h_mov)
        h_sal = huellas(simular_sala_tv(audio, semilla=semilla + 17))
        if len(h_sal):
            partes.append(h_sal)
    vecs = np.vstack(partes).astype(np.float32)
    return proyectar(vecs, mu, W).astype(np.float32)


def _descargar_bytes(url: str, max_bytes: int | None = None, timeout: int = 30) -> bytes | None:
    headers = {"User-Agent": "EstavozBot/0.2 (+https://github.com/AndyLatimer/estavoz)"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes}"
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=("eldoblaje.com" not in url), stream=True)
        if r.status_code not in (200, 206):
            return None
        buf = bytearray()
        for chunk in r.iter_content(1 << 16):
            buf.extend(chunk)
            if max_bytes is not None and len(buf) >= max_bytes:
                break
        return bytes(buf)
    except Exception:
        return None


def hidratar_filmografias_y_busqueda(
    estado: dict[str, Any],
    carpeta_datos: Path = DATOS_WEB,
    descargar_fotos: bool = True,
    reparar_vacios_eldoblaje: bool = True,
) -> dict[str, int]:
    """Rellena `papeles/*.json`, corrige vacíos `(VOZ)` desde eldoblaje, actualiza destacados y `busqueda.json`."""
    papeles_dir = carpeta_datos / "papeles"
    papeles_dir.mkdir(parents=True, exist_ok=True)
    if descargar_fotos:
        FOTOS_WEB.mkdir(parents=True, exist_ok=True)

    actores = estado["actores"]
    sesion = requests.Session()

    # 1. Descargar papeles/{i}.json faltantes desde Hugging Face Space si existen
    pendientes_hf = [i for i in range(len(actores)) if not (papeles_dir / f"{i}.json").exists()]
    if pendientes_hf:
        def _bajar_papel(i: int) -> None:
            p_path = papeles_dir / f"{i}.json"
            try:
                r = sesion.get(f"{HF_SPACE_BASE}/datos/papeles/{i}.json", timeout=15)
                if r.status_code == 200 and r.text.strip().startswith("["):
                    p_path.write_text(r.text, encoding="utf-8")
            except Exception:
                pass

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(_bajar_papel, pendientes_hf))

    if descargar_fotos:
        pendientes_fotos = [
            (i, a) for i, a in enumerate(actores)
            if a.get("foto") and not (FOTOS_WEB / f"{i}.jpg").exists()
        ]
        if pendientes_fotos:
            def _bajar_foto(item: tuple[int, dict]) -> None:
                i, a = item
                dest = FOTOS_WEB / f"{i}.jpg"
                id_num = str(a["id"]).split("-")[0]
                for url in (f"{HF_SPACE_BASE}/{a['foto']}", f"https://www.eldoblaje.com/datos/FotosActores/{id_num}.jpg"):
                    try:
                        r = sesion.get(url, timeout=15, allow_redirects=("eldoblaje.com" not in url))
                        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                            dest.write_bytes(r.content)
                            return
                    except Exception:
                        pass

            with ThreadPoolExecutor(12) as ex:
                list(ex.map(_bajar_foto, pendientes_fotos))

    # 2. Reparar actores con papeles vacíos o truncados a página 1
    reparados = 0
    if reparar_vacios_eldoblaje:
        for i, a in enumerate(actores):
            p_path = papeles_dir / f"{i}.json"
            filas = json.loads(p_path.read_text(encoding="utf-8")) if p_path.exists() else []
            num_prev = int(a.get("num_trabajos") or 0)
            necesita_reparacion = len(filas) == 0 or (num_prev > len(filas) + 10 and len(filas) <= 45)
            if not necesita_reparacion:
                continue
            id_num = str(a["id"]).split("-")[0]
            if not id_num.isdigit():
                continue
            ficha_web = eldoblaje.obtener_ficha_web(id_num, a["nombre"], descargar_todos_trabajos=True)
            if ficha_web is None:
                continue
            nuevas = [
                [t["obra"], t["personaje"], t["anio"], t["actor_original"]]
                for t in ficha_web["trabajos_completos"]
            ]
            if len(nuevas) >= len(filas):
                p_path.write_text(json.dumps(nuevas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                if nuevas:
                    reparados += 1
            if ficha_web["muestras"] and not a.get("muestra"):
                a["muestra"] = ficha_web["muestras"][0]["ruta"]

    # 3. Reconstruir `busqueda.json` completo a partir de todos los papeles/*.json
    from collections import defaultdict
    from filmografia import _PERSONAJES_GENERICOS  # reuse existing set
    obras, personajes, originales = defaultdict(set), defaultdict(set), defaultdict(set)
    veces_p, veces_o = defaultdict(int), defaultdict(int)
    todas_filas: list[list] = []
    for i in range(len(actores)):
        p_path = papeles_dir / f"{i}.json"
        filas = json.loads(p_path.read_text(encoding="utf-8")) if p_path.exists() else []
        todas_filas.append(filas)
        for fila in filas:
            obra_raw, pers_raw, _anio, orig_raw = fila[0], fila[1], fila[2], fila[3]
            if (o := titulo_base(str(obra_raw))):
                obras[o].add(i)
            if (p := personaje_base(str(pers_raw))) and len(p) >= 3 and p.lower() not in _PERSONAJES_GENERICOS:
                pk = p.title() if p.isupper() else p
                personajes[pk].add(i)
                veces_p[pk] += 1
            if (orig := nombre_original(str(orig_raw))):
                originales[orig].add(i)
                veces_o[orig] += 1

    entradas = [["obra", titulo_bonito(t), sorted(v)] for t, v in obras.items()]
    entradas += [["personaje", t, sorted(v)] for t, v in personajes.items() if veces_p[t] >= 3 or len(v) <= 2]
    entradas += [["actor", t, sorted(v)] for t, v in originales.items() if veces_o[t] >= 3]
    entradas.sort(key=lambda e: (-len(e[2]), e[1]))
    (carpeta_datos / "busqueda.json").write_text(
        json.dumps(entradas, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    popularidad = {e[1].lower(): len(e[2]) for e in entradas if e[0] == "obra"}

    # 4. Actualizar `trabajos`, `num_trabajos`, `resumen_tipos` y `vozh` de cada actor
    actualizados = 0
    for i, a in enumerate(actores):
        filas = todas_filas[i]
        if filas:
            resumen = resumen_actor_desde_papeles(filas, popularidad_obras=popularidad)
            a["trabajos"] = resumen["trabajos"]
            a["num_trabajos"] = resumen["num_trabajos"]
            if resumen["resumen_tipos"]:
                a["resumen_tipos"] = resumen["resumen_tipos"]
            if resumen["vozh"]:
                a["vozh"] = resumen["vozh"]
            actualizados += 1
        elif a.get("trabajos"):
            a["num_trabajos"] = max(int(a.get("num_trabajos") or 0), len(a["trabajos"]))

    return {"reparados_eldoblaje": reparados, "actores_enriquecidos": actualizados, "entradas_busqueda": len(entradas)}


def _anadir_actor_nuevo(
    estado: dict[str, Any],
    carpeta_datos: Path,
    id_eld: str,
    ficha_web: dict[str, Any],
    vecs_proy: np.ndarray,
) -> int:
    """Inserta un nuevo doblador en `estado` y escribe su `papeles/<idx>.json` y foto."""
    idx = len(estado["actores"])
    filas = [[t["obra"], t["personaje"], t["anio"], t["actor_original"]] for t in ficha_web["trabajos_completos"]]
    papeles_dir = carpeta_datos / "papeles"
    papeles_dir.mkdir(parents=True, exist_ok=True)
    (papeles_dir / f"{idx}.json").write_text(
        json.dumps(filas, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    foto_rel = ""
    if ficha_web.get("foto_url"):
        FOTOS_WEB.mkdir(parents=True, exist_ok=True)
        img = _descargar_bytes(ficha_web["foto_url"], max_bytes=2 << 20, timeout=15)
        if img:
            (FOTOS_WEB / f"{idx}.jpg").write_bytes(img)
            foto_rel = f"fotos/{idx}.jpg"

    resumen = resumen_actor_desde_papeles(filas) if filas else {"trabajos": [], "num_trabajos": 0, "resumen_tipos": {}, "vozh": []}
    actor_obj: dict[str, Any] = {
        "id": ficha_web["id"],
        "nombre": ficha_web["nombre"],
        "foto": foto_rel,
        "trabajos": resumen["trabajos"],
        "num_trabajos": resumen["num_trabajos"],
    }
    if ficha_web.get("muestras"):
        actor_obj["muestra"] = ficha_web["muestras"][0]["ruta"]
    if resumen["resumen_tipos"]:
        actor_obj["resumen_tipos"] = resumen["resumen_tipos"]
    if resumen["vozh"]:
        actor_obj["vozh"] = resumen["vozh"]

    estado["actores"].append(actor_obj)
    centro = normaliza(vecs_proy.mean(axis=0)).astype(np.float32)[None, :]
    estado["C"] = np.vstack([estado["C"], centro]).astype(np.float32)
    estado["trozos_por_actor"].append(vecs_proy.astype(np.float32))
    return idx


IDS_ELDOBLAJE_CURADOS = ("194153", "227111", "236619", "202193", "176570", "138813")
PODCAST_CURADOS_NOMBRES = {"33-jordi-brau", "871-marta-ullod", "manuel gimeno", "toni sevilla"}


def incorporar_desde_eldoblaje(
    estado: dict[str, Any],
    reg: dict[str, Any],
    carpeta_datos: Path = DATOS_WEB,
    max_bajo_conteo: int = 35,
    max_nuevos_buscavoces: int = 60,
    ids_nuevos_forzados: tuple[str, ...] = (),
    solo_forzados: bool = False,
) -> dict[str, int]:
    """Incorpora muestras extra de eldoblaje para actores con <=6 trozos y actores nuevos de buscavoces."""
    mu, W = estado["mu"], estado["W"]
    hechos = set(reg["clips_hechos"])
    por_id_num = {str(a["id"]).split("-")[0]: i for i, a in enumerate(estado["actores"])}

    reforzados = 0
    if not solo_forzados:
        # 1. Actores existentes con poca cobertura (<= 6 trozos) y muestra alternativa o extra
        bajos = [
            (len(estado["trozos_por_actor"][i]), i, a)
            for i, a in enumerate(estado["actores"])
            if len(estado["trozos_por_actor"][i]) <= 6 and str(a["id"]).split("-")[0].isdigit()
        ]
        bajos.sort()
        for _cnt, idx, a in bajos[:max_bajo_conteo]:
            id_num = str(a["id"]).split("-")[0]
            ficha_web = eldoblaje.obtener_ficha_web(id_num, a["nombre"], descargar_todos_trabajos=False)
            if ficha_web is None or not ficha_web["muestras"]:
                continue
            if not a.get("muestra"):
                a["muestra"] = ficha_web["muestras"][0]["ruta"]
            for m in ficha_web["muestras"]:
                clave = f"eld:{id_num}:{m['ruta']}"
                if clave in hechos:
                    continue
                # Si ya tenía Muestras/<id>.mp3 antes, priorizamos Saludos/Muestras3/Muestras4
                if m["ruta"].startswith("Muestras/") and len(estado["trozos_por_actor"][idx]) >= 4:
                    hechos.add(clave)
                    continue
                mp3_bytes = _descargar_bytes(eldoblaje.BASE + m["ruta"], max_bytes=4 << 20, timeout=20)
                if not mp3_bytes:
                    continue
                try:
                    audio = cargar_audio_bytes(mp3_bytes, max_seg=90)
                except Exception:
                    continue
                vad = DetectorVoz(sr=SAMPLE_RATE, min_voz_util_seg=1.5).analizar(audio)
                if not vad.es_toma_valida:
                    hechos.add(clave)
                    continue
                vecs = _embeddings_de_audio(audio, mu, W, semilla=int(id_num))
                if len(vecs) == 0:
                    continue
                # Filtrar trozos que no concuerdan con el actor si ya tiene centroide de referencia
                sims = vecs @ estado["C"][idx]
                validos = vecs[sims >= 0.20]
                if len(validos) == 0:
                    continue
                estado["trozos_por_actor"][idx] = np.vstack([estado["trozos_por_actor"][idx], validos])
                n_v = len(estado["trozos_por_actor"][idx])
                estado["C"][idx] = normaliza(
                    0.7 * estado["C"][idx] + 0.3 * normaliza(validos.mean(axis=0))
                ).astype(np.float32)
                hechos.add(clave)
                reforzados += 1
                print(f"  [eldoblaje+] {a['id']} (+{len(validos)} trozos de {m['ruta']}, total={n_v})", flush=True)

    # 2. Nuevos actores (lista explícita curada + opcionales de buscavoces_results.asp)
    descubiertos = [] if solo_forzados else eldoblaje.descubrir_buscavoces(max_paginas=10)[:max_nuevos_buscavoces]
    candidatos = list(dict.fromkeys([*ids_nuevos_forzados, *descubiertos]))
    nuevos_ids = [i for i in candidatos if i not in por_id_num]
    agregados = 0
    for id_num in nuevos_ids:
        clave_actor = f"eld_nuevo:{id_num}"
        if clave_actor in hechos:
            continue
        ficha_web = eldoblaje.obtener_ficha_web(id_num, "", descargar_todos_trabajos=False)
        if ficha_web is None:
            continue
        hechos.add(clave_actor)
        if not ficha_web["muestras"]:
            continue
        vecs_actor = []
        for m in ficha_web["muestras"]:
            mp3_bytes = _descargar_bytes(eldoblaje.BASE + m["ruta"], max_bytes=4 << 20, timeout=20)
            if not mp3_bytes:
                continue
            try:
                audio = cargar_audio_bytes(mp3_bytes, max_seg=90)
            except Exception:
                continue
            v = _embeddings_de_audio(audio, mu, W, semilla=int(id_num))
            if len(v):
                vecs_actor.append(v)
        if not vecs_actor:
            continue
        V_new = np.vstack(vecs_actor).astype(np.float32)
        if len(V_new) < 2:
            continue
        # Solo descargamos todas las páginas de filmografía si realmente tiene audio válido
        ficha_completa = eldoblaje.obtener_ficha_web(id_num, ficha_web["nombre"], descargar_todos_trabajos=True)
        if ficha_completa is not None and len(ficha_completa["trabajos_completos"]) >= len(ficha_web["trabajos_completos"]):
            ficha_web = ficha_completa
        _anadir_actor_nuevo(estado, carpeta_datos, id_num, ficha_web, V_new)
        por_id_num[id_num] = len(estado["actores"]) - 1
        agregados += 1
        print(f"  [eldoblaje NEW] {ficha_web['id']} ({len(V_new)} trozos, {len(ficha_web['trabajos_completos'])} trabajos)", flush=True)

    reg["clips_hechos"] = sorted(hechos)
    _guardar_registro(reg)
    return {"reforzados_eldoblaje": reforzados, "nuevos_eldoblaje": agregados}


def _descargar_turnos_podcast(mp3_url: str, skip_seg: int = 60, max_seg: int = 720, max_bytes: int = 12 << 20) -> list[np.ndarray]:
    datos = _descargar_bytes(mp3_url, max_bytes=max_bytes, timeout=30)
    if not datos:
        return []
    try:
        audio = cargar_audio_bytes(datos, max_seg=max_seg + skip_seg)
    except Exception:
        return []
    if skip_seg and len(audio) > (skip_seg + 40) * SAMPLE_RATE:
        audio = audio[skip_seg * SAMPLE_RATE :]
    vad = DetectorVoz(sr=SAMPLE_RATE, min_voz_util_seg=1.5).analizar(audio)
    turnos = segmentar_intervenciones(audio, vad=vad, sr=SAMPLE_RATE, min_duracion_turno_seg=3.0, max_duracion_turno_seg=7.0)
    return [t for t in turnos if len(t) >= 3 * SAMPLE_RATE][:28]


def _identificar_hosts_podcast(
    prog: str,
    episodios: list[dict[str, Any]],
    estado: dict[str, Any],
    por_nombre: dict[str, int],
    max_ejemplos: int = 5,
) -> list[np.ndarray]:
    """Construye los centroides de los presentadores habituales de un programa usando episodios de conocidos."""
    mu, W, C = estado["mu"], estado["W"], estado["C"]
    candidatos_host: list[np.ndarray] = []
    probados = 0
    for ep in episodios:
        if probados >= max_ejemplos:
            break
        tp = f" {normalizar_texto(ep['titulo'])} "
        idx_conocido = None
        for nom, i in por_nombre.items():
            if f" {nom} " in tp:
                idx_conocido = i
                break
        if idx_conocido is None:
            continue
        turnos = _descargar_turnos_podcast(ep["mp3"], skip_seg=ep["skip_seg"], max_seg=360, max_bytes=7 << 20)
        if len(turnos) < 6:
            continue
        probados += 1
        embs = np.stack([normaliza(huella_trozo(t)) for t in turnos])
        projs = proyectar(embs, mu, W)
        labels = fcluster(linkage(projs, method="average", metric="cosine"), t=0.52, criterion="distance")
        for lbl in set(labels):
            ix = np.flatnonzero(labels == lbl)
            if len(ix) < 2:
                continue
            sim_inv = float((projs[ix] @ C[idx_conocido]).mean())
            if sim_inv < 0.30:
                candidatos_host.append(normaliza(projs[ix].mean(0)))

    if len(candidatos_host) < 2:
        return []
    H = np.vstack(candidatos_host)
    labels_h = fcluster(linkage(H, method="average", metric="cosine"), t=0.45, criterion="distance")
    hosts = []
    for lbl in set(labels_h):
        ix = np.flatnonzero(labels_h == lbl)
        if len(ix) >= 2:  # aparece repetidamente en >= 2 episodios distintos -> es presentador
            hosts.append(normaliza(H[ix].mean(0)).astype(np.float32))
    return hosts


_ALIASES_NOMBRE_CA_ES = {
    "luis": "lluis", "lluis": "luis",
    "ricardo": "ricard", "ricard": "ricardo",
    "juan": "joan", "joan": "juan",
    "jaime": "jaume", "jaume": "jaime",
    "eduardo": "eduard", "eduard": "eduardo",
    "carmen": "carme", "carme": "carmen",
    "jorge": "jordi", "jordi": "jorge",
    "carlos": "carles", "carles": "carlos",
    "jose": "josep", "josep": "jose",
    "nuria": "nuria", "rosa maria": "rosa mari", "rosa mari": "rosa maria",
}
_CABECERAS_RSS = {"User-Agent": "Mozilla/5.0 (compatible; Estavoz/1.0)"}


def _claves_nombre_actor(nombre: str) -> list[str]:
    n = normalizar_texto(nombre)
    partes = n.split()
    if len(partes) < 2 or len(n) < 8:
        return []
    claves = [n]
    alt = _ALIASES_NOMBRE_CA_ES.get(partes[0])
    if alt:
        claves.append(" ".join([alt, *partes[1:]]))
    return claves


def incorporar_desde_podcasts(
    estado: dict[str, Any],
    reg: dict[str, Any],
    carpeta_datos: Path = DATOS_WEB,
    max_reforzar: int = 35,
    max_nuevos: int = 12,
    candidatos_nuevos_extra: list[str] | None = None,
    feeds_filtro: set[str] | None = None,
    objetivos_filtro: set[str] | None = None,
) -> dict[str, int]:
    """Extrae fragmentos verificados de podcasts para actores con pocos trozos y nuevos referentes."""
    mu, W = estado["mu"], estado["W"]
    por_nombre: dict[str, int] = {}
    for i, a in enumerate(estado["actores"]):
        for clave in _claves_nombre_actor(a["nombre"]):
            por_nombre.setdefault(clave, i)

    hechos_pod = set(reg["podcasts_hechos"])
    hosts_cache: dict[str, list[np.ndarray]] = {
        k: [np.array(v, dtype=np.float32) for v in vs]
        for k, vs in reg.get("hosts_podcast", {}).items()
    }

    # 1. Leer feeds RSS de programas verificados
    episodios_por_prog: dict[str, list[dict[str, Any]]] = {}
    for prog, info in FEEDS_PODCAST.items():
        if feeds_filtro is not None and prog not in feeds_filtro:
            continue
        try:
            r = requests.get(info["url"], headers=_CABECERAS_RSS, timeout=20)
            raiz = ET.fromstring(r.content)
        except Exception:
            continue
        lista = []
        for it in raiz.iter("item"):
            enc = it.find("enclosure")
            if enc is None or not enc.get("url"):
                continue
            lista.append({
                "programa": prog,
                "titulo": it.findtext("title") or "",
                "mp3": enc.get("url"),
                "skip_seg": info["skip_seg"],
            })
        episodios_por_prog[prog] = lista

    # 2. Reforzar actores existentes (con <= 12 trozos, o seleccionados explícitamente)
    tareas_refuerzo: list[tuple[int, int, dict[str, Any]]] = []
    for prog, eps in episodios_por_prog.items():
        for ep in eps:
            tp = f" {normalizar_texto(ep['titulo'])} "
            for nom, idx in por_nombre.items():
                actor_id = estado["actores"][idx]["id"]
                es_obj = actor_id in objetivos_filtro or nom in objetivos_filtro if objetivos_filtro is not None else len(estado["trozos_por_actor"][idx]) <= 12
                if f" {nom} " in tp and es_obj:
                    clave = f"pod:{hashlib.sha1(ep['mp3'].encode()).hexdigest()[:12]}:{actor_id}"
                    if clave not in hechos_pod:
                        tareas_refuerzo.append((len(estado["trozos_por_actor"][idx]), idx, ep))

    tareas_refuerzo.sort(key=lambda x: x[0])
    reforzados = 0
    for _cnt, idx, ep in tareas_refuerzo[:max_reforzar]:
        a = estado["actores"][idx]
        clave = f"pod:{hashlib.sha1(ep['mp3'].encode()).hexdigest()[:12]}:{a['id']}"
        if clave in hechos_pod:
            continue
        if objetivos_filtro is None and len(estado["trozos_por_actor"][idx]) >= 30:
            hechos_pod.add(clave)
            continue
        prog = ep["programa"]
        if prog not in hosts_cache:
            hosts_cache[prog] = _identificar_hosts_podcast(prog, episodios_por_prog[prog], estado, por_nombre)
            reg["hosts_podcast"][prog] = [h.tolist() for h in hosts_cache[prog]]

        turnos = _descargar_turnos_podcast(ep["mp3"], skip_seg=ep["skip_seg"], max_seg=540, max_bytes=10 << 20)
        hechos_pod.add(clave)
        if len(turnos) < 5:
            continue
        embs = np.stack([normaliza(huella_trozo(t)) for t in turnos])
        projs = proyectar(embs, mu, W)
        labels = fcluster(linkage(projs, method="average", metric="cosine"), t=0.52, criterion="distance")
        C_norm = normaliza(estado["C"])
        hosts = hosts_cache.get(prog, [])

        mejor_cluster = None
        for lbl in set(labels):
            ix = np.flatnonzero(labels == lbl)
            if len(ix) < 3:
                continue
            centro_c = normaliza(projs[ix].mean(0))
            sim_host = max((float(centro_c @ h) for h in hosts), default=0.0)
            if sim_host >= 0.36:
                continue
            sims_todos = (projs[ix] @ C_norm.T).mean(axis=0)
            sim_propio = float(sims_todos[idx])
            puesto = int((sims_todos > sim_propio).sum()) + 1
            if puesto <= 3 and sim_propio >= 0.30 and (sim_propio - sim_host) >= 0.06:
                cand = (sim_propio, ix)
                if mejor_cluster is None or cand[0] > mejor_cluster[0]:
                    mejor_cluster = cand

        if mejor_cluster is None:
            continue
        sim_propio, ix_sel = mejor_cluster
        audio_sel = np.concatenate([turnos[i] for i in ix_sel])[: 72 * SAMPLE_RATE]
        sem = int(hashlib.sha1(clave.encode()).hexdigest()[:8], 16)
        vecs_nuevos = _embeddings_de_audio(audio_sel, mu, W, semilla=sem)
        if len(vecs_nuevos) < 3:
            continue
        # Peso moderado para entrevista, o mayor si solo tenía 2-4 trozos previos
        peso_ent = 0.35 if len(estado["trozos_por_actor"][idx]) <= 6 else 0.22
        estado["trozos_por_actor"][idx] = np.vstack([estado["trozos_por_actor"][idx], vecs_nuevos])
        estado["C"][idx] = normaliza(
            (1.0 - peso_ent) * C_norm[idx] + peso_ent * normaliza(vecs_nuevos.mean(axis=0))
        ).astype(np.float32)
        reforzados += 1
        print(f"  [podcast+] {a['id']} ({prog}, sim={sim_propio:.3f}, +{len(vecs_nuevos)} trozos)", flush=True)

    # 3. Nuevos actores sin muestra previa verificados en podcasts (requiere >=2 episodios independientes o programa con presentador verificado)
    nuevos_agregados = 0
    if candidatos_nuevos_extra:
        # Mapa nombre -> lista de episodios donde aparece en el título
        eps_por_nuevo: dict[str, list[dict[str, Any]]] = {}
        for nom_raw in candidatos_nuevos_extra:
            n_norm = normalizar_texto(nom_raw)
            if n_norm in por_nombre:
                continue
            for prog, eps in episodios_por_prog.items():
                for ep in eps:
                    tp = f" {normalizar_texto(ep['titulo'])} "
                    if f" {n_norm} " in tp and not re.search(r"\b(i|y|amb|con)\b", ep["titulo"].lower()):
                        eps_por_nuevo.setdefault(nom_raw, []).append(ep)

        C_norm = normaliza(estado["C"])
        for nom_raw, eps in eps_por_nuevo.items():
            if nuevos_agregados >= max_nuevos:
                break
            clave_n = f"pod_nuevo:{normalizar_texto(nom_raw)}"
            if clave_n in hechos_pod:
                continue
            hechos_pod.add(clave_n)

            clusters_ep = []
            for ep in eps[:2]:
                prog = ep["programa"]
                if prog not in hosts_cache:
                    hosts_cache[prog] = _identificar_hosts_podcast(prog, episodios_por_prog[prog], estado, por_nombre)
                    reg["hosts_podcast"][prog] = [h.tolist() for h in hosts_cache[prog]]
                hosts = hosts_cache.get(prog, [])
                turnos = _descargar_turnos_podcast(ep["mp3"], skip_seg=ep["skip_seg"], max_seg=540, max_bytes=10 << 20)
                if len(turnos) < 6:
                    continue
                embs = np.stack([normaliza(huella_trozo(t)) for t in turnos])
                projs = proyectar(embs, mu, W)
                labels = fcluster(linkage(projs, method="average", metric="cosine"), t=0.50, criterion="distance")
                for lbl in set(labels):
                    ix = np.flatnonzero(labels == lbl)
                    if len(ix) < 4:
                        continue
                    c_cl = normaliza(projs[ix].mean(0))
                    sim_h = max((float(c_cl @ h) for h in hosts), default=0.0)
                    intra = float((projs[ix] @ c_cl).mean())
                    max_ex = float((projs[ix] @ C_norm.T).mean(0).max())
                    if sim_h < 0.32 and intra >= 0.52 and max_ex < 0.43:
                        clusters_ep.append((ep, turnos, ix, c_cl, sim_h, intra))

            if not clusters_ep:
                continue
            # Verificación: o aparece en >= 2 episodios con similitud mutua >= 0.45, o único programa fiable con >= 6 turnos y host conocido
            aprobados = []
            if len(clusters_ep) >= 2:
                mejor_par = None
                for a_i in range(len(clusters_ep)):
                    for b_i in range(a_i + 1, len(clusters_ep)):
                        if clusters_ep[a_i][0]["programa"] == clusters_ep[b_i][0]["programa"]:
                            continue
                        sim_ab = float(clusters_ep[a_i][3] @ clusters_ep[b_i][3])
                        if sim_ab >= 0.45 and (mejor_par is None or sim_ab > mejor_par[0]):
                            mejor_par = (sim_ab, clusters_ep[a_i], clusters_ep[b_i])
                if mejor_par is not None:
                    aprobados = [mejor_par[1], mejor_par[2]]
            if not aprobados:
                # Respaldo estricto: programa con host identificado, >= 6 turnos coherentes (intra >= 0.56)
                fuertes = [c for c in clusters_ep if len(hosts_cache.get(c[0]["programa"], [])) >= 1 and len(c[2]) >= 6 and c[5] >= 0.56]
                if fuertes:
                    aprobados = [max(fuertes, key=lambda c: len(c[2]) * c[5])]

            if not aprobados:
                continue
            encontrado = eldoblaje.buscar_actor_por_nombre(nom_raw)
            if not encontrado:
                continue
            id_eld, nombre_of = encontrado
            if any(str(a["id"]).split("-")[0] == id_eld for a in estado["actores"]):
                continue
            ficha_web = eldoblaje.obtener_ficha_web(id_eld, nombre_of, descargar_todos_trabajos=True)
            if not ficha_web or len(ficha_web["trabajos_completos"]) < 10:
                continue

            audio_all = np.concatenate([c[1][i] for c in aprobados for i in c[2]])[: 90 * SAMPLE_RATE]
            sem = int(hashlib.sha1(clave_n.encode()).hexdigest()[:8], 16)
            vecs_new = _embeddings_de_audio(audio_all, mu, W, semilla=sem)
            if len(vecs_new) < 4:
                continue
            _anadir_actor_nuevo(estado, carpeta_datos, id_eld, ficha_web, vecs_new)
            por_nombre[normalizar_texto(ficha_web["nombre"])] = len(estado["actores"]) - 1
            C_norm = normaliza(estado["C"])
            nuevos_agregados += 1
            print(
                f"  [podcast NEW] {ficha_web['id']} ({len(vecs_new)} trozos, {len(ficha_web['trabajos_completos'])} trabajos)",
                flush=True,
            )

    reg["podcasts_hechos"] = sorted(hechos_pod)
    _guardar_registro(reg)
    return {"reforzados_podcast": reforzados, "nuevos_podcast": nuevos_agregados}


REFERENTES_SIN_MUESTRA = [
    "Pere Molina", "Mercè Montalà", "Sergi Zamora", "Joan Carles Gustems",
    "Manuel Gimeno", "Jaume Comas", "Toni Sevilla", "Rosa Mari Hernández",
    "Mònica Padrós", "Joan Sanz", "Albert Trifol", "Javier García-Sáenz",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Enriquecimiento incremental de voces y filmografías de EstaVoz")
    p.add_argument("--podar", action="store_true")
    p.add_argument("--hidratar", action="store_true")
    p.add_argument("--eldoblaje", action="store_true")
    p.add_argument("--podcasts", action="store_true")
    p.add_argument("--curados", action="store_true", help="Incorporación rápida y determinista para CI")
    p.add_argument("--todo", action="store_true")
    p.add_argument("--max-refuerzo", type=int, default=35)
    args = p.parse_args()

    os.environ.setdefault("ESTAVOZ_HILOS", "4")
    estado = cargar_estado_web()
    reg = _cargar_registro()

    if args.todo or args.podar or args.curados:
        n_pod = podar_intrusos_estado(estado, umbral_outlier=0.20, min_trozos=4)
        print(f"Podados {n_pod} trozos intrusos (sim < 0.20)", flush=True)
    if args.curados:
        res_e = incorporar_desde_eldoblaje(
            estado, reg, ids_nuevos_forzados=IDS_ELDOBLAJE_CURADOS, solo_forzados=True,
        )
        print("Eldoblaje (curados):", res_e, flush=True)
        res_p = incorporar_desde_podcasts(
            estado, reg, max_reforzar=6, max_nuevos=4,
            candidatos_nuevos_extra=["Manuel Gimeno", "Toni Sevilla"],
            feeds_filtro={"La Vermella"},
            objetivos_filtro=PODCAST_CURADOS_NOMBRES,
        )
        print("Podcasts (curados):", res_p, flush=True)
        res_h = hidratar_filmografias_y_busqueda(estado, descargar_fotos=False, reparar_vacios_eldoblaje=True)
        print("Filmografías (curados):", res_h, flush=True)
    else:
        if args.todo or args.eldoblaje:
            res_e = incorporar_desde_eldoblaje(estado, reg, max_bajo_conteo=args.max_refuerzo)
            print("Eldoblaje:", res_e, flush=True)
            guardar_estado_web(estado)
        if args.todo or args.podcasts:
            res_p = incorporar_desde_podcasts(
                estado,
                reg,
                max_reforzar=args.max_refuerzo,
                candidatos_nuevos_extra=REFERENTES_SIN_MUESTRA,
            )
            print("Podcasts:", res_p, flush=True)
            guardar_estado_web(estado)
        if args.todo or args.hidratar:
            res_h = hidratar_filmografias_y_busqueda(estado)
            print("Filmografías:", res_h, flush=True)

    resumen = guardar_estado_web(estado)
    print(f"Estado web validado: {resumen['num_actores']} actores, {resumen['num_trozos']} trozos", flush=True)


if __name__ == "__main__":
    main()
