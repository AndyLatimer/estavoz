"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera en frontend/:
  datos/lda.bin         media (192) + matriz W (192 x 192) de la LDA, float32
  datos/centroides.bin  huella media de cada actor ya proyectada, float32 (actores x 192)
  datos/actores.json    fichas, en el mismo orden que los centroides
  datos/busqueda.json   obras, personajes y actores originales con su reparto ("¿Qué estás viendo?")
  datos/papeles/<n>.json  trabajos completos de cada doblador
  fotos/<n>.jpg
La web no enlaza ni carga nada de fuera.
Si existe data/respuestas.json (backend/aprender.py), añade las respuestas "¿Acertamos?" que pasen los filtros.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx.
"""
import json
import re
import shutil
import unicodedata
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
from lda import centroides, entrenar_lda, proyectar  # noqa: E402


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


def respuestas_validas(mu, W, C, slugs):
    """Huellas de las respuestas de los usuarios que se pueden usar para aprender.

    Filtros contra respuestas falsas o despistes:
      - solo respuestas con un actor ("No lo sé" no enseña nada);
      - la voz tiene que parecerse algo a ese actor (parecido >= 0.2 y entre sus 10 más parecidos);
      - cada actor recibe como mucho tantas huellas de usuarios como trozos propios tiene.
    """
    f = RAIZ / "data" / "respuestas.json"
    if not f.exists():
        return [], []
    pos = {s: i for i, s in enumerate(slugs)}
    X, y = [], []
    for r in json.loads(f.read_text(encoding="utf-8")):
        actor = r["propuesto"] if r["acierto"] else r.get("real")
        if actor not in pos:
            continue
        h = np.asarray(r["huellas"], dtype=np.float32)
        if h.ndim != 2 or h.shape[1] != C.shape[1]:
            continue
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        punt = (proyectar(h, mu, W) @ C.T).mean(0)
        if punt[pos[actor]] >= 0.2 and pos[actor] in np.argsort(-punt)[:10]:
            X.append(h)
            y += [actor] * len(h)
    return X, y


def main():
    datos = np.load(RAIZ / "data" / "indice.npz")
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]
    fichas = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
    slugs = sorted(fichas)

    # la LDA aprende de los actores con 2+ clips (misma voz en grabaciones distintas)
    base = np.array([c.split("#")[0] for c in clips])  # "x.mp3#movil" es el mismo clip que "x.mp3"
    n_clips = {a: len(set(base[y == a])) for a in slugs}
    sel = np.isin(y, [a for a in slugs if n_clips[a] >= 2])
    mu, W = entrenar_lda(X[sel], y[sel])
    C = centroides(proyectar(X, mu, W), y, slugs)

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
        C = centroides(proyectar(np.concatenate([X, Xr]), mu, W), np.concatenate([y, yr]), slugs)
        print(f"Respuestas de usuarios usadas: {len(yr)} huellas de {len(set(yr))} actores")

    salida = RAIZ / "frontend" / "datos"
    shutil.rmtree(salida, ignore_errors=True)
    salida.mkdir(parents=True)
    np.concatenate([mu, W.ravel()]).astype("<f4").tofile(salida / "lda.bin")
    C.astype("<f4").tofile(salida / "centroides.bin")

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
    web = {"dim": int(X.shape[1]), "actores": actores}
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Búsqueda: {len(entradas)} obras/personajes/actores, {(salida / 'busqueda.json').stat().st_size >> 10} KB")
    print(f"{len(slugs)} actores (LDA entrenada con {sel.sum()} trozos de "
          f"{sum(v >= 2 for v in n_clips.values())} actores) -> {salida}")


if __name__ == "__main__":
    main()
