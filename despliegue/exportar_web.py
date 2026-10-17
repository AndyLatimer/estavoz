"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera en frontend/:
  datos/lda.bin         media (192) + matriz W (192 x 192) de la LDA, float32
  datos/centroides.bin  huella media de cada actor ya proyectada, float32 (actores x 192)
  datos/actores.json    fichas, en el mismo orden que los centroides
  datos/busqueda.json   palabra -> actores que la tienen en sus trabajos ("¿Qué estás viendo?")
  fotos/<n>.jpg
La web no enlaza ni carga nada de fuera.
Si existe data/respuestas.json (backend/aprender.py), añade las respuestas "¿Acertamos?" que pasen los filtros.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/ecapa-int8.onnx.
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


# palabras que no sirven para buscar una obra
VACIAS = set("""de del la las el los lo un una unos unas y e o u en a al con por para sin su sus the of and to in on
at an mi tu se que es temporada serie tv miniserie videojuego capitulo episodio especial parte vol volumen
pelicula film documental animacion cortometraje version doblaje redoblaje ii iii iv vi vii viii ix""".split())


def palabras(texto: str) -> set[str]:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z0-9]+", t) if (len(w) >= 3 or w.isdigit()) and w not in VACIAS}


def indice_busqueda(slugs, fichas):
    """palabra -> índices de los actores que la tienen en algún trabajo (título, personaje o actor original)."""
    indice = {}
    for i, s in enumerate(slugs):
        f = RAIZ / "data" / "actores" / s / "trabajos.json"
        if f.exists():
            filas = [f"{t['obra']} {t['personaje']} {t['actor_original']}" for t in json.loads(f.read_text(encoding="utf-8"))]
        else:  # aún sin la lista completa: los trabajos destacados de la ficha
            filas = fichas[s].get("trabajos", [])
        for w in set().union(*map(palabras, filas)) if filas else ():
            indice.setdefault(w, []).append(i)
    return indice


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
            "num_trabajos": fichas[s].get("num_trabajos", 0),
        })
    busqueda = indice_busqueda(slugs, fichas)
    (salida / "busqueda.json").write_text(json.dumps(busqueda, separators=(",", ":")), encoding="utf-8")
    web = {"dim": int(X.shape[1]), "actores": actores}
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Búsqueda: {len(busqueda)} palabras, {(salida / 'busqueda.json').stat().st_size >> 10} KB")
    print(f"{len(slugs)} actores (LDA entrenada con {sel.sum()} trozos de "
          f"{sum(v >= 2 for v in n_clips.values())} actores) -> {salida}")


if __name__ == "__main__":
    main()
