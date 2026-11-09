"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera en frontend/:
  datos/lda.bin         media (256) + matriz W (256 x 256) de la LDA, float32
  datos/centroides.bin  huella media de cada actor ya proyectada, float32 (actores x 256)
  datos/trozos.bin      todas las huellas proyectadas, por actor: inicio de cada actor (uint32, actores + 1),
                        escala de cada huella (float32) y la huella en int8. La web mezcla el parecido con
                        la huella media y con los 3 trozos más parecidos de cada actor (+3 puntos de acierto)
  datos/actores.json    fichas, en el mismo orden que los centroides
  datos/busqueda.json   obras, personajes y actores originales con su reparto ("¿Qué estás viendo?")
  datos/papeles/<n>.json  trabajos completos de cada doblador
  fotos/<n>.jpg
La web no enlaza ni carga nada de fuera.
Si existe data/respuestas.json (backend/aprender.py), añade las respuestas "¿Acertamos?" que pasen los filtros.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx (VoxBlink2 ResNet100, v3).
"""
import hashlib
import json
import re
import shutil
import unicodedata
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
from auditar_indice import filtrar_trozos_intrusos  # noqa: E402
from lda import entrenar_lda, proyectar  # noqa: E402
from metadatos_modelo import (  # noqa: E402
    VERSION_LDA_ACTUAL,
    info_por_version,
    inspeccionar_onnx,
    validar_artefactos_web,
    validar_compatibilidad_indice,
)
from rechazo import UMBRALES_VOXBLINK_R100_CALIBRADOS  # noqa: E402


def titulo_base(t: str) -> str:
    """'LA JAULA [1ª temporada]' y 'LA JAULA (1x05)' -> 'LA JAULA' (las temporadas cuentan como una obra)."""
    t = re.sub(r"\[[^\]]*\]|\(\d+x\d+\)", "", t)
    return re.sub(r"\s+", " ", t).strip(" :-")


def personaje_base(p: str) -> str:
    """'Buzz Lightyear (voz)' / 'Sofiane (1x05)' -> 'Buzz Lightyear'."""
    p = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", p)
    return re.sub(r"\s+", " ", p).strip(" .,:-")


MINUSCULAS = set("de del la las el los lo un una unos unas y e o u en a al con por para sin su sus".split())


def titulo_bonito(t: str) -> str:
    """'TOY STORY 2: LOS JUGUETES VUELVEN A LA CARGA' -> 'Toy Story 2: los Juguetes Vuelven a la Carga'."""
    if not t.isupper():
        return t
    palabras = t.lower().split(" ")
    return " ".join(w if (k and w in MINUSCULAS) else w[:1].upper() + w[1:] for k, w in enumerate(palabras))


def nombre_original(a: str) -> str:
    """'Bale, Christian' -> 'Christian Bale'."""
    if a.count(",") == 1:
        apellidos, nombre = (x.strip() for x in a.split(","))
        a = f"{nombre} {apellidos}"
    return a.title() if a.isupper() else a


def entradas_busqueda(slugs):
    """Sugerencias de "¿Qué estás viendo?": obras, personajes y actores originales con su reparto.

    Devuelve [tipo, texto, [índices de dobladores]] ordenado de más a menos importante
    (las obras con más dobladores primero). Actores originales solo si salen en varios trabajos.
    """
    from collections import defaultdict
    obras, personajes, originales = defaultdict(set), defaultdict(set), defaultdict(set)
    veces_p, veces_o = defaultdict(int), defaultdict(int)
    for i, s in enumerate(slugs):
        f = RAIZ / "data" / "actores" / s / "trabajos.json"
        if not f.exists():
            continue
        for t in json.loads(f.read_text(encoding="utf-8")):
            if (o := titulo_base(t["obra"])):
                obras[o].add(i)
            if (p := personaje_base(t["personaje"])) and len(p) >= 3:
                personajes[p.title() if p.isupper() else p].add(i)
                veces_p[p.title() if p.isupper() else p] += 1
            if (a := nombre_original(t["actor_original"].strip())):
                originales[a].add(i)
                veces_o[a] += 1
    salida = [["obra", titulo_bonito(t), sorted(v)] for t, v in obras.items()]
    # personajes: todos los que tienen nombre propio (los genéricos como "Policía" salen muchas veces)
    salida += [["personaje", t, sorted(v)] for t, v in personajes.items() if veces_p[t] >= 3 or len(v) <= 2]
    salida += [["actor", t, sorted(v)] for t, v in originales.items() if veces_o[t] >= 3]
    salida.sort(key=lambda e: (-len(e[2]), e[1]))
    return salida


# versión del modelo de voz que usa la web; las respuestas guardan la suya y solo se aprovechan
# las del modelo en uso (1 = ECAPA, 2 = VoxBlink2 ResNet34, 3 = VoxBlink2 ResNet100)
VERSION_MODELO = "3"


def respuestas_validas(mu, W, C, slugs, ruta_respuestas=None):
    """Huellas de las respuestas de los usuarios que se pueden usar para aprender.

    Filtros contra respuestas falsas, despistes o envenenamiento:
      - solo respuestas con un actor válido y de la versión actual del modelo;
      - vectores finitos con norma no degenerada y sin duplicados exactos;
      - la voz tiene que superar el umbral dudoso calibrado (>= 0.252) y estar en el Top-5 del actor;
      - cada actor recibe como mucho tantas huellas de usuarios como trozos propios tiene.
    """
    f = Path(ruta_respuestas) if ruta_respuestas is not None else (RAIZ / "data" / "respuestas.json")
    if not f.exists():
        return [], []
    pos = {s: i for i, s in enumerate(slugs)}
    umbral_min = max(0.25, float(UMBRALES_VOXBLINK_R100_CALIBRADOS.umbral_dudoso))
    hashes_vistos: set[str] = set()
    X, y = [], []
    for r in json.loads(f.read_text(encoding="utf-8")):
        if str(r.get("version")) != VERSION_MODELO:
            continue
        actor = r.get("propuesto") if r.get("acierto") else r.get("real")
        if actor not in pos:
            continue
        h = np.asarray(r.get("huellas", []), dtype=np.float32)
        if h.ndim != 2 or len(h) == 0 or h.shape[1] != C.shape[1] or not np.all(np.isfinite(h)):
            continue
        normas = np.linalg.norm(h, axis=1, keepdims=True)
        if np.any(normas < 1e-5):
            continue
        clave_hash = hashlib.sha1(np.round(h, 4).tobytes()).hexdigest()
        if clave_hash in hashes_vistos:
            continue
        h = h / normas
        punt = (proyectar(h, mu, W) @ C.T).mean(0)
        if punt[pos[actor]] >= umbral_min and pos[actor] in np.argsort(-punt)[:5]:
            hashes_vistos.add(clave_hash)
            X.append(h)
            y += [actor] * len(h)
    return X, y


def centroides_mezcla(Xp, y, clips, slugs, peso_entrevistas=0.5, peso_respuestas=0.35):
    """Huella media de cada actor, con la voz de entrevistas (`voz_`) a medio peso (0.5)
    y las respuestas voluntarias de usuarios (`respuesta`) a peso moderado (0.35) para no desplazar
    las muestras verificadas de estudio."""
    clips_str = np.asarray(clips, dtype=str)
    entrevista = np.array(["/voz_" in c for c in clips_str], dtype=bool)
    respuesta = clips_str == "respuesta"
    C = np.zeros((len(slugs), Xp.shape[1]))
    for i, a in enumerate(slugs):
        es_actor = (y == a)
        propia = (~entrevista) & (~respuesta) & es_actor
        m_ent = entrevista & es_actor
        m_resp = respuesta & es_actor
        if propia.any():
            v = Xp[propia].mean(0)
            C[i] = v / np.linalg.norm(v)
        if m_ent.any():   # los que solo tienen voz de entrevistas (sin muestra en eldoblaje) se quedan con esa
            v = Xp[m_ent].mean(0)
            C[i] += (peso_entrevistas if propia.any() else 1.0) * v / np.linalg.norm(v)
        if m_resp.any():
            v = Xp[m_resp].mean(0)
            tiene_base = propia.any() or m_ent.any()
            C[i] += (peso_respuestas if tiene_base else 1.0) * v / np.linalg.norm(v)
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def main():
    datos = np.load(RAIZ / "data" / "indice.npz")
    ruta_onnx = RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"
    info_onnx = inspeccionar_onnx(ruta_onnx) if ruta_onnx.exists() else info_por_version(VERSION_MODELO)
    validar_compatibilidad_indice(datos, info_modelo=info_onnx, version_esperada=VERSION_MODELO)
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]
    fichas = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
    slugs = sorted(fichas)

    # la LDA aprende de los actores con 2+ clips (misma voz en grabaciones distintas)
    base = np.array([c.split("#")[0] for c in clips])  # "x.mp3#movil" es el mismo clip que "x.mp3"
    n_clips = {a: len(set(base[y == a])) for a in slugs}
    sel = np.isin(y, [a for a in slugs if n_clips[a] >= 2])
    mu, W = entrenar_lda(X[sel], y[sel])
    C = centroides_mezcla(proyectar(X, mu, W), y, clips, slugs)
    Xt, yt, clips_t = X, y, np.asarray(clips, dtype=str)

    # lo que han enseñado los usuarios entra en la huella media de cada actor
    Xr, yr = respuestas_validas(mu, W, C, slugs)
    if Xr:
        Xr, yr = np.concatenate(Xr), np.array(yr)
        propios = {a: int((y == a).sum()) for a in slugs}
        vistos, tope = {}, np.zeros(len(yr), dtype=bool)
        for i, a in enumerate(yr):
            vistos[a] = vistos.get(a, 0) + 1
            tope[i] = vistos[a] <= propios[a]
        Xr, yr = Xr[tope], yr[tope]
        Xt = np.concatenate([X, Xr])
        yt = np.concatenate([y, yr])
        clips_t = np.concatenate([clips_t, ["respuesta"] * len(yr)])
        C = centroides_mezcla(proyectar(Xt, mu, W), yt, clips_t, slugs)
        print(f"Respuestas de usuarios usadas: {len(yr)} huellas de {len(set(yr))} actores")

    # Poda automática de trozos intrusos (interlocutores secundarios o música pura con sim < 0.15)
    conservar, intrusos = filtrar_trozos_intrusos(proyectar(Xt, mu, W), yt, C, slugs, umbral_outlier=0.15)
    if intrusos:
        Xt, yt, clips_t = Xt[conservar], yt[conservar], clips_t[conservar]
        C = centroides_mezcla(proyectar(Xt, mu, W), yt, clips_t, slugs)
        print(f"Trozos intrusos filtrados: {len(intrusos)} segmentos con sim < 0.15 al centroide del actor")

    salida = RAIZ / "frontend" / "datos"
    shutil.rmtree(salida, ignore_errors=True)
    salida.mkdir(parents=True)
    np.concatenate([mu, W.ravel()]).astype("<f4").tofile(salida / "lda.bin")
    C.astype("<f4").tofile(salida / "centroides.bin")
    orden = np.argsort(np.searchsorted(slugs, yt), kind="stable")
    V = proyectar(Xt[orden], mu, W)
    escala = np.abs(V).max(1) / 127
    inicio = np.searchsorted(np.searchsorted(slugs, yt[orden]), np.arange(len(slugs) + 1))
    with open(salida / "trozos.bin", "wb") as f:
        f.write(inicio.astype("<u4").tobytes())
        f.write(escala.astype("<f4").tobytes())
        f.write(np.round(V / escala[:, None]).astype(np.int8).tobytes())

    fotos = RAIZ / "frontend" / "fotos"
    shutil.rmtree(fotos, ignore_errors=True)
    fotos.mkdir()
    actores = []
    for i, s in enumerate(slugs):
        foto = RAIZ / "data" / "actores" / s / "foto.jpg"
        if foto.exists():
            shutil.copy(foto, fotos / f"{i}.jpg")
        actores.append({
            "id": s,
            "nombre": fichas[s]["nombre"],
            "foto": f"fotos/{i}.jpg" if foto.exists() else "",
            "trabajos": fichas[s].get("trabajos", []),
            "num_trabajos": len(json.loads(t.read_text(encoding="utf-8"))) if (t := RAIZ / "data" / "actores" / s / "trabajos.json").exists()
                            else fichas[s].get("num_trabajos", 0),
        })
    entradas = entradas_busqueda(slugs)
    (salida / "busqueda.json").write_text(json.dumps(entradas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # trabajos completos de cada doblador (se pide solo el del resultado, para decir el personaje)
    papeles = salida / "papeles"
    papeles.mkdir()
    for i, s in enumerate(slugs):
        f = RAIZ / "data" / "actores" / s / "trabajos.json"
        filas = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        (papeles / f"{i}.json").write_text(json.dumps(
            [[t["obra"], t["personaje"], t["anio"], t["actor_original"]] for t in filas], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    web = {
        "dim": int(X.shape[1]),
        "version_modelo": VERSION_MODELO,
        "arquitectura_modelo": info_onnx.arquitectura,
        "variante_modelo": info_onnx.variante,
        "version_lda": VERSION_LDA_ACTUAL,
        "umbrales_rechazo": UMBRALES_VOXBLINK_R100_CALIBRADOS.a_dict(),
        "actores": actores,
    }
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    validar_artefactos_web(salida, ruta_onnx=ruta_onnx if ruta_onnx.exists() else None, version_esperada=VERSION_MODELO)
    print(f"Búsqueda: {len(entradas)} obras/personajes/actores, {(salida / 'busqueda.json').stat().st_size >> 10} KB")
    print(f"{len(slugs)} actores (LDA entrenada con {sel.sum()} trozos de "
          f"{sum(v >= 2 for v in n_clips.values())} actores) -> {salida}")


if __name__ == "__main__":
    main()
