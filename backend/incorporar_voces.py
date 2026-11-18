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

from auditar_indice import filtrar_trozos_intrusos, hidratar_f0_catalogo_web  # noqa: E402
from aumento_acustico import (  # noqa: E402
    DetectorVoz,
    estimar_f0_mediana,
    segmentar_intervenciones,
    simular_sala_tv,
)
import eldoblaje  # noqa: E402
from filmografia import (  # noqa: E402
    construir_entradas_busqueda,
    exportar_indices_busqueda,
    normalizar_texto,
    resumen_actor_desde_papeles,
)
from lda import normaliza, proyectar  # noqa: E402
from metadatos_modelo import info_por_version, validar_artefactos_web  # noqa: E402
from simular_movil import movil as simular_movil_sintetico  # noqa: E402
from voz import SAMPLE_RATE, cargar_audio_bytes, huellas, huella_trozo, info_modelo_actual  # noqa: E402

DATOS_WEB = RAIZ / "frontend" / "datos"
FOTOS_WEB = RAIZ / "frontend" / "fotos"
RUTA_ONNX = RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"
CACHE_DIR = RAIZ / "data" / "incorporacion"
REGISTRO_PATH = CACHE_DIR / "registro.json"
AMPLIACIONES_NPZ = RAIZ / "data" / "ampliaciones.npz"
AMPLIACIONES_FICHAS = RAIZ / "data" / "ampliaciones_fichas.json"
ACTORES_DATA_DIR = RAIZ / "data" / "actores"
HF_SPACE_BASE = "https://srharrison-estavoz.static.hf.space"

# Referentes históricos o con poca cobertura previa cuyo refuerzo prioriza el catálogo
ACTORES_REFORZAR_PRIORITARIOS = {
    "1-gonzalo-abril",
    "4-victor-agramunt",
    "15-ruperto-ares",
    "58-roberto-cuenca-martinez",
    "68-carlos-del-pino",
    "73-rafael-calvo",
    "98-camilo-garcia",
    "102-manolo-garcia",
    "108-jose-luis-gil",
    "113-juan-carlos-gustems",
    "119-david-jenner",
    "126-alfonso-laguna",
    "134-dionisio-macias",
    "137-lluis-marco",
    "147-jose-luis-mediavilla",
    "152-pedro-molina",
    "156-adolfo-moreno",
    "161-pep-anton-munoz",
    "169-joan-pera",
    "170-roger-pera",
    "174-jordi-pons",
    "177-luis-posada",
    "197-pep-sais",
    "211-pablo-tribaldos",
    "213-alberto-trifol",
    "227-sergio-zamora",
    "228-marc-zanni",
    "231-marta-angelat",
    "249-julia-gallego",
    "264-nuria-mediavilla",
    "267-mercedes-montala",
    "272-laura-palacios",
    "284-alba-sola",
    "288-nuria-trifol",
    "334-carlos-revilla",
    "610-maria-jesus-lleonart",
    "667-adelaida-lopez",
    "794-raul-llorens",
    "865-belen-roca",
    "870-mark-ullod",
    "1325-joel-mulachs",
    "1356-elisa-beuter",
    "80794-mario-garcia",
    "102450-nerea-alfonso",
    "106383-vera-bosch",
}

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
    hidratar_f0_catalogo_web(carpeta_datos)
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


def cargar_ampliaciones(
    ruta_npz: Path = AMPLIACIONES_NPZ,
    ruta_fichas: Path = AMPLIACIONES_FICHAS,
) -> dict[str, Any]:
    """Carga las ampliaciones persistentes previas (`ampliaciones.npz` + `ampliaciones_fichas.json`)."""
    bloques: list[np.ndarray] = []
    etiquetas: list[str] = []
    clips: list[str] = []
    if ruta_npz.exists():
        datos = np.load(ruta_npz)
        if len(datos["vectores"]):
            bloques.append(np.asarray(datos["vectores"], dtype=np.float32))
            etiquetas = [str(x) for x in datos["etiquetas"]]
            clips = [str(x) for x in datos["clips"]]
    fichas: dict[str, dict[str, Any]] = {}
    if ruta_fichas.exists():
        fichas = json.loads(ruta_fichas.read_text(encoding="utf-8"))
    return {
        "bloques": bloques,
        "etiquetas": etiquetas,
        "clips": clips,
        "pares_existentes": set(zip(etiquetas, clips)),
        "fichas": fichas,
        "modificado": False,
    }


def _ficha_persistente(actor_obj: dict[str, Any]) -> dict[str, Any]:
    """Extrae metadatos estables del actor (sin rutas temporales `fotos/<idx>.jpg`)."""
    ficha: dict[str, Any] = {
        "id": actor_obj["id"],
        "nombre": actor_obj["nombre"],
        "trabajos": actor_obj.get("trabajos") or [],
        "num_trabajos": int(actor_obj.get("num_trabajos") or len(actor_obj.get("trabajos") or [])),
    }
    for k in ("muestra", "f0", "resumen_tipos", "vozh"):
        if actor_obj.get(k):
            ficha[k] = actor_obj[k]
    return ficha


def _registrar_ampliacion(
    amp: dict[str, Any] | None,
    slug: str,
    crudos: np.ndarray,
    clip_ids: list[str],
    actor_obj: dict[str, Any] | None = None,
    trabajos_completos: list[dict[str, Any]] | None = None,
    foto_bytes: bytes | None = None,
    actores_dir: Path = ACTORES_DATA_DIR,
) -> None:
    """Registra vectores crudos pre-LDA, ficha y activos (`foto.jpg`, `trabajos.json`) en memoria/disco."""
    if trabajos_completos is not None and len(trabajos_completos):
        dir_a = actores_dir / slug
        dir_a.mkdir(parents=True, exist_ok=True)
        (dir_a / "trabajos.json").write_text(
            json.dumps(
                [
                    {
                        "obra": t["obra"],
                        "personaje": t["personaje"],
                        "anio": t["anio"],
                        "actor_original": t["actor_original"],
                    }
                    for t in trabajos_completos
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    if foto_bytes:
        dir_a = actores_dir / slug
        dir_a.mkdir(parents=True, exist_ok=True)
        (dir_a / "foto.jpg").write_bytes(foto_bytes)

    if amp is None:
        return
    if len(crudos) and len(crudos) == len(clip_ids):
        mascara = np.array([(slug, cid) not in amp["pares_existentes"] for cid in clip_ids], dtype=bool)
        if mascara.any():
            amp["bloques"].append(np.asarray(crudos[mascara], dtype=np.float32))
            for cid in np.asarray(clip_ids, dtype=object)[mascara]:
                cid_s = str(cid)
                amp["etiquetas"].append(slug)
                amp["clips"].append(cid_s)
                amp["pares_existentes"].add((slug, cid_s))
            amp["modificado"] = True

    if actor_obj is not None:
        prev = amp["fichas"].get(slug, {})
        nueva = _ficha_persistente(actor_obj)
        if not nueva["trabajos"] and prev.get("trabajos"):
            nueva["trabajos"] = prev["trabajos"]
            nueva["num_trabajos"] = max(nueva["num_trabajos"], int(prev.get("num_trabajos") or 0))
        for k in ("muestra", "f0", "resumen_tipos", "vozh"):
            if k not in nueva and prev.get(k):
                nueva[k] = prev[k]
        if nueva != prev:
            amp["fichas"][slug] = nueva
            amp["modificado"] = True


def guardar_ampliaciones(
    amp: dict[str, Any],
    ruta_npz: Path = AMPLIACIONES_NPZ,
    ruta_fichas: Path = AMPLIACIONES_FICHAS,
    dim: int = 256,
) -> dict[str, int]:
    """Escribe `data/ampliaciones.npz` y `data/ampliaciones_fichas.json` para persistencia en `estavoz-datos`."""
    ruta_npz.parent.mkdir(parents=True, exist_ok=True)
    V = (
        np.vstack(amp["bloques"]).astype(np.float32)
        if amp["bloques"]
        else np.zeros((0, dim), dtype=np.float32)
    )
    info = info_modelo_actual()
    if V.shape[1] != info.dim_embedding:
        info = info_por_version("3" if V.shape[1] == 256 else "1")
    np.savez_compressed(
        ruta_npz,
        vectores=V,
        etiquetas=np.asarray(amp["etiquetas"]),
        clips=np.asarray(amp["clips"]),
        modelo_version=np.array(info.version),
        modelo_arquitectura=np.array(info.arquitectura),
        modelo_variante=np.array(info.variante),
    )
    ruta_fichas.write_text(
        json.dumps(amp["fichas"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_actores = len(set(amp["etiquetas"]) | set(amp["fichas"]))
    return {"trozos_crudos": int(len(V)), "actores_ampliados": int(n_actores)}


def _crudos_y_proyectados(
    audio: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    semilla: int,
    con_aumentos: bool = True,
    clip_base: str = "extra",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extrae huellas crudas normalizadas (`limpio`, `#movil`, `#sala`), sus `clip_id` y la proyección LDA."""
    audio = np.asarray(audio, dtype=np.float32)
    h_limpio = huellas(audio)
    if len(h_limpio) == 0:
        vacio = np.zeros((0, len(mu)), dtype=np.float32)
        return vacio, vacio, []
    partes = [h_limpio]
    clip_ids = [clip_base] * len(h_limpio)
    if con_aumentos:
        rng = np.random.default_rng(semilla)
        h_mov = huellas(simular_movil_sintetico(audio, rng))
        if len(h_mov):
            partes.append(h_mov)
            clip_ids.extend([f"{clip_base}#movil"] * len(h_mov))
        h_sal = huellas(simular_sala_tv(audio, semilla=semilla + 17))
        if len(h_sal):
            partes.append(h_sal)
            clip_ids.extend([f"{clip_base}#sala"] * len(h_sal))
    crudos = np.vstack(partes).astype(np.float32)
    return crudos, proyectar(crudos, mu, W).astype(np.float32), clip_ids


def _embeddings_de_audio(audio: np.ndarray, mu: np.ndarray, W: np.ndarray, semilla: int, con_aumentos: bool = True) -> np.ndarray:
    """Extrae huellas limpias + `#movil` + `#sala` y las proyecta al espacio LDA."""
    return _crudos_y_proyectados(audio, mu, W, semilla=semilla, con_aumentos=con_aumentos)[1]


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

    # 3. Reconstruir `busqueda.json` + fragmentos web (`busqueda_top.json`, `busqueda/00..63.json`)
    todas_filas: list[list] = []
    for i in range(len(actores)):
        p_path = papeles_dir / f"{i}.json"
        filas = json.loads(p_path.read_text(encoding="utf-8")) if p_path.exists() else []
        todas_filas.append(filas)

    entradas = construir_entradas_busqueda(todas_filas)
    exportar_indices_busqueda(entradas, carpeta_datos)

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
    f0: float | None = None,
    crudos: np.ndarray | None = None,
    clip_ids: list[str] | None = None,
    amp: dict[str, Any] | None = None,
) -> int:
    """Inserta un nuevo doblador en `estado`, guarda sus activos locales y lo registra en `ampliaciones`."""
    idx = len(estado["actores"])
    filas = [[t["obra"], t["personaje"], t["anio"], t["actor_original"]] for t in ficha_web["trabajos_completos"]]
    papeles_dir = carpeta_datos / "papeles"
    papeles_dir.mkdir(parents=True, exist_ok=True)
    (papeles_dir / f"{idx}.json").write_text(
        json.dumps(filas, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    foto_rel = ""
    img_bytes: bytes | None = None
    if ficha_web.get("foto_url"):
        FOTOS_WEB.mkdir(parents=True, exist_ok=True)
        img_bytes = _descargar_bytes(ficha_web["foto_url"], max_bytes=2 << 20, timeout=15)
        if img_bytes:
            (FOTOS_WEB / f"{idx}.jpg").write_bytes(img_bytes)
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
    if f0 is not None and 60.0 <= float(f0) <= 380.0:
        actor_obj["f0"] = round(float(f0), 1)

    estado["actores"].append(actor_obj)
    centro = normaliza(vecs_proy.mean(axis=0)).astype(np.float32)[None, :]
    estado["C"] = np.vstack([estado["C"], centro]).astype(np.float32)
    estado["trozos_por_actor"].append(vecs_proy.astype(np.float32))

    _registrar_ampliacion(
        amp,
        ficha_web["id"],
        crudos if crudos is not None else np.zeros((0, estado["dim"]), dtype=np.float32),
        clip_ids or [],
        actor_obj=actor_obj,
        trabajos_completos=ficha_web.get("trabajos_completos"),
        foto_bytes=img_bytes,
    )
    return idx


IDS_ELDOBLAJE_CURADOS = ("194153", "227111", "236619", "202193", "176570", "138813")
PODCAST_CURADOS_NOMBRES = {"33-jordi-brau", "871-marta-ullod", "manuel gimeno", "toni sevilla"}


def incorporar_desde_eldoblaje(
    estado: dict[str, Any],
    reg: dict[str, Any],
    carpeta_datos: Path = DATOS_WEB,
    max_bajo_conteo: int = 80,
    max_nuevos_buscavoces: int = 120,
    ids_nuevos_forzados: tuple[str, ...] = (),
    solo_forzados: bool = False,
    umbral_trozos_bajos: int = 10,
    amp: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Incorpora muestras extra de eldoblaje para actores con pocos trozos y actores nuevos de buscavoces."""
    mu, W = estado["mu"], estado["W"]
    hechos = set(reg["clips_hechos"])
    por_id_num = {str(a["id"]).split("-")[0]: i for i, a in enumerate(estado["actores"])}

    reforzados = 0
    if not solo_forzados:
        # 1. Actores existentes con poca cobertura o prioritarios, ordenando por prioridad + más prolíficos
        bajos = [
            (
                0 if a["id"] in ACTORES_REFORZAR_PRIORITARIOS else 1,
                len(estado["trozos_por_actor"][i]),
                -int(a.get("num_trabajos") or len(a.get("trabajos") or [])),
                i,
                a,
            )
            for i, a in enumerate(estado["actores"])
            if (len(estado["trozos_por_actor"][i]) <= umbral_trozos_bajos or a["id"] in ACTORES_REFORZAR_PRIORITARIOS)
            and str(a["id"]).split("-")[0].isdigit()
        ]
        bajos.sort()
        for _prio, _cnt, _neg_trab, idx, a in bajos[:max_bajo_conteo]:
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
                crudos, vecs, cids = _crudos_y_proyectados(audio, mu, W, semilla=int(id_num), clip_base=m["ruta"])
                if len(vecs) == 0:
                    continue
                if not a.get("f0"):
                    f0_est = estimar_f0_mediana(audio)
                    if f0_est is not None and 60.0 <= f0_est <= 380.0:
                        a["f0"] = round(float(f0_est), 1)
                # Filtrar trozos que no concuerdan con el actor si ya tiene centroide de referencia
                sims = vecs @ estado["C"][idx]
                mascara = sims >= 0.20
                validos = vecs[mascara]
                if len(validos) == 0:
                    continue
                estado["trozos_por_actor"][idx] = np.vstack([estado["trozos_por_actor"][idx], validos])
                n_v = len(estado["trozos_por_actor"][idx])
                estado["C"][idx] = normaliza(
                    0.7 * estado["C"][idx] + 0.3 * normaliza(validos.mean(axis=0))
                ).astype(np.float32)
                cids_val = [cids[k] for k in np.flatnonzero(mascara)]
                _registrar_ampliacion(amp, a["id"], crudos[mascara], cids_val, actor_obj=a)
                hechos.add(clave)
                reforzados += 1
                print(f"  [eldoblaje+] {a['id']} (+{len(validos)} trozos de {m['ruta']}, total={n_v})", flush=True)

    # 2. Nuevos actores (lista explícita curada + opcionales de buscavoces_results.asp)
    descubiertos = [] if solo_forzados else eldoblaje.descubrir_buscavoces(max_paginas=15)[:max_nuevos_buscavoces]
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
        crudos_actor = []
        vecs_actor = []
        cids_actor: list[str] = []
        f0s_actor = []
        for idx_m, m in enumerate(ficha_web["muestras"]):
            mp3_bytes = _descargar_bytes(eldoblaje.BASE + m["ruta"], max_bytes=4 << 20, timeout=20)
            if not mp3_bytes:
                continue
            try:
                audio = cargar_audio_bytes(mp3_bytes, max_seg=90)
            except Exception:
                continue
            cr_m, v_m, cids_m = _crudos_y_proyectados(audio, mu, W, semilla=int(id_num) + idx_m * 31, clip_base=m["ruta"])
            if len(v_m):
                crudos_actor.append(cr_m)
                vecs_actor.append(v_m)
                cids_actor.extend(cids_m)
                f0_m = estimar_f0_mediana(audio)
                if f0_m is not None and 60.0 <= f0_m <= 380.0:
                    f0s_actor.append(float(f0_m))
        if not vecs_actor:
            continue
        V_new = np.vstack(vecs_actor).astype(np.float32)
        if len(V_new) < 2:
            continue
        CR_new = np.vstack(crudos_actor).astype(np.float32)
        # Solo descargamos todas las páginas de filmografía si realmente tiene audio válido
        ficha_completa = eldoblaje.obtener_ficha_web(id_num, ficha_web["nombre"], descargar_todos_trabajos=True)
        if ficha_completa is not None and len(ficha_completa["trabajos_completos"]) >= len(ficha_web["trabajos_completos"]):
            ficha_web = ficha_completa
        f0_med = float(np.median(f0s_actor)) if f0s_actor else None
        _anadir_actor_nuevo(
            estado, carpeta_datos, id_num, ficha_web, V_new,
            f0=f0_med, crudos=CR_new, clip_ids=cids_actor, amp=amp,
        )
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
        # Algunos .m4a (p. ej. CharletaCast) guardan el átomo `moov` al final del archivo
        if max_bytes is None:
            return []
        datos_full = _descargar_bytes(mp3_url, max_bytes=None, timeout=60)
        if not datos_full:
            return []
        try:
            audio = cargar_audio_bytes(datos_full, max_seg=max_seg + skip_seg)
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
    "pedro": "pere", "pere": "pedro",
    "sergio": "sergi", "sergi": "sergio",
    "alberto": "albert", "albert": "alberto",
    "rafa": "rafael", "rafael": "rafa",
    "gaby": "gabriel", "gabriel": "gaby",
    "fer": "fernando", "fernando": "fer",
    "javi": "javier", "javier": "javi",
    "mercedes": "merce", "merce": "mercedes",
    "pepe": "jose",
    "paco": "francisco", "francisco": "paco",
    "alex": "alejandro", "alejandro": "alex",
    "toni": "antonio", "antonio": "toni",
    "lola": "dolores", "dolores": "lola",
    "chus": "maria jesus",
}
_VARIANTES_PREFIJO = (
    ("juan carlos", "joan carles"),
    ("joan carles", "juan carlos"),
    ("josep maria", "jose maria"),
    ("jose maria", "josep maria"),
    ("francisco javier", "javier"),
    ("javier", "francisco javier"),
    ("francisco javier", "javi"),
    ("maria lluisa", "maria luisa"),
    ("maria luisa", "maria lluisa"),
    ("m lluisa", "maria lluisa"),
    ("m luisa", "maria luisa"),
    ("maria lluisa", "m lluisa"),
    ("maria jesus", "m jesus"),
    ("m jesus", "maria jesus"),
    ("maria jesus", "chus"),
    ("rosa mari", "rosa maria"),
    ("rosa maria", "rosa mari"),
)
_CABECERAS_RSS = {"User-Agent": "Mozilla/5.0 (compatible; Estavoz/1.0)"}


def _claves_nombre_actor(nombre: str) -> list[str]:
    claves: list[str] = []

    def _agregar(cand: str) -> None:
        c = normalizar_texto(cand)
        if len(c.split()) >= 2 and len(c) >= 8 and c not in claves:
            claves.append(c)

    base = normalizar_texto(nombre)
    if len(base.split()) < 2 or len(base) < 8:
        return []
    _agregar(base)

    # Desempaquetar apodos entre paréntesis: "Fernando (fer) Delgado" -> "fernando delgado" y "fer delgado"
    sin_par = normalizar_texto(re.sub(r"\([^)]*\)", " ", nombre))
    if sin_par:
        _agregar(sin_par)
        for m_apodo in re.findall(r"\(([^)]+)\)", nombre):
            ap = normalizar_texto(m_apodo)
            partes_sp = sin_par.split()
            if ap and len(ap) >= 3 and len(partes_sp) >= 2:
                _agregar(f"{ap} {' '.join(partes_sp[1:])}")

    for base_c in list(claves):
        partes = base_c.split()
        alt = _ALIASES_NOMBRE_CA_ES.get(partes[0])
        if alt:
            _agregar(" ".join([alt, *partes[1:]]))
        for p1, p2 in _VARIANTES_PREFIJO:
            if base_c.startswith(p1 + " "):
                _agregar(p2 + base_c[len(p1):])
    return claves


def _es_entrevista_solitaria(titulo: str, nombre_norm: str) -> bool:
    """Detecta si `titulo` presenta a un único invitado alrededor de `nombre_norm` (admite `con <Nombre>`)."""
    tp = normalizar_texto(titulo)
    pos = tp.find(nombre_norm)
    if pos < 0:
        return False
    # Quedarnos solo con la cláusula del invitado (ignorando títulos de obra previos como "Oliver y Benji. Entrevista a...")
    antes = tp[:pos]
    for sep in ("entrevista a ", "charleta con ", "especial ", " amb ", " con ", ":", "|", " - "):
        idx_sep = antes.rfind(sep)
        if idx_sep >= 0:
            antes = antes[idx_sep + len(sep):]
            break
    despues = tp[pos + len(nombre_norm):]
    for sep in ("|", "#", " - ", "(", " el ", " la "):
        idx_dep = despues.find(sep)
        if idx_dep >= 0:
            despues = despues[:idx_dep]
    clausula = f" {antes.strip()} {nombre_norm} {despues.strip()} "
    return not re.search(r"\b(y|i)\b", clausula)


def _buscar_actor_con_aliases(nom_raw: str) -> tuple[str, str] | None:
    candidatos = [nom_raw]
    for clave in _claves_nombre_actor(nom_raw):
        if clave not in {normalizar_texto(c) for c in candidatos}:
            candidatos.append(clave.title())
    for cand in candidatos:
        encontrado = eldoblaje.buscar_actor_por_nombre(cand)
        if encontrado:
            return encontrado
    return None


def incorporar_desde_podcasts(
    estado: dict[str, Any],
    reg: dict[str, Any],
    carpeta_datos: Path = DATOS_WEB,
    max_reforzar: int = 80,
    max_nuevos: int = 80,
    candidatos_nuevos_extra: list[str] | None = None,
    feeds_filtro: set[str] | None = None,
    objetivos_filtro: set[str] | None = None,
    umbral_trozos_refuerzo: int = 18,
    amp: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Extrae fragmentos verificados de podcasts para actores con pocos trozos, prioritarios y nuevos referentes."""
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

    # 2. Reforzar actores existentes (con <= umbral_trozos_refuerzo, prioritarios, o seleccionados explícitamente)
    tareas_refuerzo: list[tuple[int, int, int, int, dict[str, Any]]] = []
    vistos_refuerzo: set[tuple[int, str]] = set()
    for prog, eps in episodios_por_prog.items():
        for ep in eps:
            tp = f" {normalizar_texto(ep['titulo'])} "
            for nom, idx in por_nombre.items():
                if (idx, ep["mp3"]) in vistos_refuerzo:
                    continue
                actor_obj = estado["actores"][idx]
                actor_id = actor_obj["id"]
                n_troz = len(estado["trozos_por_actor"][idx])
                es_prio = actor_id in ACTORES_REFORZAR_PRIORITARIOS
                es_obj = (
                    (actor_id in objetivos_filtro or nom in objetivos_filtro)
                    if objetivos_filtro is not None
                    else (n_troz <= umbral_trozos_refuerzo or es_prio)
                )
                if f" {nom} " in tp and es_obj:
                    clave = f"pod:{hashlib.sha1(ep['mp3'].encode()).hexdigest()[:12]}:{actor_id}"
                    if clave not in hechos_pod:
                        vistos_refuerzo.add((idx, ep["mp3"]))
                        n_trab = int(actor_obj.get("num_trabajos") or len(actor_obj.get("trabajos") or []))
                        tareas_refuerzo.append((0 if es_prio else 1, n_troz, -n_trab, idx, ep))

    tareas_refuerzo.sort(key=lambda x: (x[0], x[1], x[2]))
    reforzados = 0
    for _prio, _cnt, _neg_trab, idx, ep in tareas_refuerzo[:max_reforzar]:
        a = estado["actores"][idx]
        clave = f"pod:{hashlib.sha1(ep['mp3'].encode()).hexdigest()[:12]}:{a['id']}"
        if clave in hechos_pod:
            continue
        techo_trozos = 36
        if objetivos_filtro is None and len(estado["trozos_por_actor"][idx]) >= techo_trozos:
            hechos_pod.add(clave)
            continue
        prog = ep["programa"]
        if prog not in hosts_cache:
            hosts_cache[prog] = _identificar_hosts_podcast(prog, episodios_por_prog[prog], estado, por_nombre)
            reg["hosts_podcast"][prog] = [h.tolist() for h in hosts_cache[prog]]

        turnos = _descargar_turnos_podcast(ep["mp3"], skip_seg=ep["skip_seg"], max_seg=540, max_bytes=14 << 20)
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
        hash_ep = hashlib.sha1(ep["mp3"].encode()).hexdigest()[:10]
        clip_pod = f"podcasts/{a['id']}/voz_{hash_ep}.wav"
        sem = int(hashlib.sha1(clave.encode()).hexdigest()[:8], 16)
        crudos_nuevos, vecs_nuevos, cids_pod = _crudos_y_proyectados(
            audio_sel, mu, W, semilla=sem, clip_base=clip_pod,
        )
        if len(vecs_nuevos) < 3:
            continue
        # Peso moderado para entrevista, o mayor si solo tenía pocos trozos previos
        peso_ent = 0.35 if len(estado["trozos_por_actor"][idx]) <= 6 else 0.22
        estado["trozos_por_actor"][idx] = np.vstack([estado["trozos_por_actor"][idx], vecs_nuevos])
        estado["C"][idx] = normaliza(
            (1.0 - peso_ent) * C_norm[idx] + peso_ent * normaliza(vecs_nuevos.mean(axis=0))
        ).astype(np.float32)
        _registrar_ampliacion(amp, a["id"], crudos_nuevos, cids_pod, actor_obj=a)
        reforzados += 1
        print(f"  [podcast+] {a['id']} ({prog}, sim={sim_propio:.3f}, +{len(vecs_nuevos)} trozos)", flush=True)

    # 3. Nuevos actores sin muestra previa verificados en podcasts (requiere >=2 episodios o programa solitario fiable)
    nuevos_agregados = 0
    if candidatos_nuevos_extra:
        eps_por_nuevo: dict[str, list[tuple[bool, dict[str, Any]]]] = {}
        vistos_ep_nuevo: set[tuple[str, str]] = set()
        for nom_raw in candidatos_nuevos_extra:
            claves_cand = _claves_nombre_actor(nom_raw)
            if not claves_cand or any(c in por_nombre for c in claves_cand):
                continue
            for prog, eps in episodios_por_prog.items():
                for ep in eps:
                    if (nom_raw, ep["mp3"]) in vistos_ep_nuevo:
                        continue
                    tp = f" {normalizar_texto(ep['titulo'])} "
                    clave_hit = next((c for c in claves_cand if f" {c} " in tp), None)
                    if clave_hit:
                        vistos_ep_nuevo.add((nom_raw, ep["mp3"]))
                        es_solo = _es_entrevista_solitaria(ep["titulo"], clave_hit)
                        eps_por_nuevo.setdefault(nom_raw, []).append((es_solo, ep))

        C_norm = normaliza(estado["C"])
        for nom_raw, items_ep in eps_por_nuevo.items():
            if nuevos_agregados >= max_nuevos:
                break
            if any(c in por_nombre for c in _claves_nombre_actor(nom_raw)):
                continue
            clave_n = f"pod_nuevo:{normalizar_texto(nom_raw)}"
            if clave_n in hechos_pod:
                continue
            hechos_pod.add(clave_n)

            # Priorizar entrevistas en solitario y diversidad de programas
            items_ep.sort(key=lambda x: (0 if x[0] else 1, x[1]["programa"]))
            clusters_ep = []
            vistos_progs: set[str] = set()
            seleccion: list[tuple[bool, dict[str, Any]]] = []
            for item in items_ep:
                if item[1]["programa"] not in vistos_progs or len(seleccion) < 2:
                    seleccion.append(item)
                    vistos_progs.add(item[1]["programa"])
                if len(seleccion) >= 3:
                    break

            for es_solo, ep in seleccion:
                prog = ep["programa"]
                if prog not in hosts_cache:
                    hosts_cache[prog] = _identificar_hosts_podcast(prog, episodios_por_prog[prog], estado, por_nombre)
                    reg["hosts_podcast"][prog] = [h.tolist() for h in hosts_cache[prog]]
                hosts = hosts_cache.get(prog, [])
                turnos = _descargar_turnos_podcast(ep["mp3"], skip_seg=ep["skip_seg"], max_seg=540, max_bytes=14 << 20)
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
                        clusters_ep.append((ep, turnos, ix, c_cl, sim_h, intra, es_solo))

            if not clusters_ep:
                continue
            # Verificación: >=2 programas independientes con similitud >= 0.45, o entrevista solitaria fiable (>=6 turnos, intra>=0.56)
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
                fuertes = [
                    c for c in clusters_ep
                    if c[6] and len(hosts_cache.get(c[0]["programa"], [])) >= 1 and len(c[2]) >= 6 and c[5] >= 0.56
                ]
                if fuertes:
                    aprobados = [max(fuertes, key=lambda c: len(c[2]) * c[5])]

            if not aprobados:
                continue
            encontrado = _buscar_actor_con_aliases(nom_raw)
            if not encontrado:
                continue
            id_eld, nombre_of = encontrado
            if any(str(a["id"]).split("-")[0] == id_eld for a in estado["actores"]):
                continue
            ficha_web = eldoblaje.obtener_ficha_web(id_eld, nombre_of, descargar_todos_trabajos=True)
            if not ficha_web or len(ficha_web["trabajos_completos"]) < 10:
                continue

            audio_all = np.concatenate([c[1][i] for c in aprobados for i in c[2]])[: 90 * SAMPLE_RATE]
            hash_nuevo = hashlib.sha1(aprobados[0][0]["mp3"].encode()).hexdigest()[:10]
            clip_pod_new = f"podcasts/{ficha_web['id']}/voz_{hash_nuevo}.wav"
            sem = int(hashlib.sha1(clave_n.encode()).hexdigest()[:8], 16)
            crudos_new, vecs_new, cids_new = _crudos_y_proyectados(
                audio_all, mu, W, semilla=sem, clip_base=clip_pod_new,
            )
            if len(vecs_new) < 4:
                continue
            _anadir_actor_nuevo(
                estado, carpeta_datos, id_eld, ficha_web, vecs_new,
                f0=estimar_f0_mediana(audio_all),
                crudos=crudos_new,
                clip_ids=cids_new,
                amp=amp,
            )
            for k_of in _claves_nombre_actor(ficha_web["nombre"]):
                por_nombre[k_of] = len(estado["actores"]) - 1
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
    "Maria Lluïsa Solà", "Núria Domènech", "Roger Isasi-Isasmendi", "M. Jesús Lleonart",
    "Elena de Maeztu", "César Capilla", "Pablo Domínguez", "Belén Rodríguez",
    "Marta Dualde", "Álvaro Vancells", "Jorge Saudinos", "Chus Varona",
    "Gabriel Jiménez", "Luis Mas", "Xavi Fernández", "Laura Pastor",
    "David García Llop", "Masumi Mutsuda", "Clara Schwarze", "David Arnau",
    "Montse Ferrer", "Quim Roca", "Eva Hernández", "Jordi Navarro",
    "Noah de Diego", "Lorena Armengol", "Carlos Sante", "Mar Montoro",
    "Erik Cruz", "Ahimsa Sánchez", "Begoña Hernando", "Xabi Alkiza",
    "Pedro Tena", "Celia Guisasola", "Marta Baonza", "Javier Pérez Alarcón",
    "Tito Trifol", "Dolors Martínez", "Xavi Martín", "Carles Lladó",
    "Josep Maria Mas", "Estel Tort", "Teresa Marcos", "Fran Rodríguez",
    "Adrián Cantador", "Majo Montesinos", "Sara Gómez", "Odile Arqué",
    "Lluís Comes", "M. Lluïsa Magaña", "Gerry Núñez", "José Luis Siurana",
    "Juan Vicente Julián", "Gago Montesinos", "Ramón Higuero", "Andreu López",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Enriquecimiento incremental de voces y filmografías de EstaVoz")
    p.add_argument("--podar", action="store_true")
    p.add_argument("--hidratar", action="store_true")
    p.add_argument("--eldoblaje", action="store_true")
    p.add_argument("--podcasts", action="store_true")
    p.add_argument("--curados", action="store_true", help="Incorporación rápida y determinista para CI")
    p.add_argument("--todo", action="store_true")
    p.add_argument("--max-refuerzo", type=int, default=100)
    p.add_argument("--max-nuevos", type=int, default=80)
    p.add_argument("--sin-persistir", action="store_true", help="No escribir `data/ampliaciones.npz` ni `ampliaciones_fichas.json`")
    args = p.parse_args()

    os.environ.setdefault("ESTAVOZ_HILOS", "4")
    estado = cargar_estado_web()
    reg = _cargar_registro()
    amp = None if args.sin_persistir else cargar_ampliaciones()

    if args.todo or args.podar or args.curados:
        n_pod = podar_intrusos_estado(estado, umbral_outlier=0.20, min_trozos=4)
        print(f"Podados {n_pod} trozos intrusos (sim < 0.20)", flush=True)
    if args.curados:
        res_e = incorporar_desde_eldoblaje(
            estado, reg, ids_nuevos_forzados=IDS_ELDOBLAJE_CURADOS, solo_forzados=True, amp=amp,
        )
        print("Eldoblaje (curados):", res_e, flush=True)
        res_p = incorporar_desde_podcasts(
            estado, reg, max_reforzar=6, max_nuevos=4,
            candidatos_nuevos_extra=["Manuel Gimeno", "Toni Sevilla"],
            feeds_filtro={"La Vermella"},
            objetivos_filtro=PODCAST_CURADOS_NOMBRES,
            amp=amp,
        )
        print("Podcasts (curados):", res_p, flush=True)
        res_h = hidratar_filmografias_y_busqueda(estado, descargar_fotos=False, reparar_vacios_eldoblaje=True)
        print("Filmografías (curados):", res_h, flush=True)
    else:
        if args.todo or args.eldoblaje:
            res_e = incorporar_desde_eldoblaje(
                estado, reg, max_bajo_conteo=args.max_refuerzo,
                max_nuevos_buscavoces=max(args.max_nuevos, 80), amp=amp,
            )
            print("Eldoblaje:", res_e, flush=True)
            guardar_estado_web(estado)
        if args.todo or args.podcasts:
            res_p = incorporar_desde_podcasts(
                estado,
                reg,
                max_reforzar=args.max_refuerzo,
                max_nuevos=args.max_nuevos,
                candidatos_nuevos_extra=REFERENTES_SIN_MUESTRA,
                amp=amp,
            )
            print("Podcasts:", res_p, flush=True)
            guardar_estado_web(estado)
        if args.todo or args.hidratar:
            res_h = hidratar_filmografias_y_busqueda(estado)
            print("Filmografías:", res_h, flush=True)

    if amp is not None and amp["modificado"]:
        res_amp = guardar_ampliaciones(amp, dim=int(estado["dim"]))
        print(f"Ampliaciones persistidas: {res_amp['trozos_crudos']} trozos crudos, {res_amp['actores_ampliados']} actores", flush=True)
    resumen = guardar_estado_web(estado)
    print(f"Estado web validado: {resumen['num_actores']} actores, {resumen['num_trozos']} trozos", flush=True)


if __name__ == "__main__":
    main()
