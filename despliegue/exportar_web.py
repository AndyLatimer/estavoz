"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera en frontend/:
  datos/lda.bin         media (192) + matriz W (192 x 192) de la LDA, float32
  datos/centroides.bin  huella media de cada actor ya proyectada, float32 (actores x 192)
  datos/actores.json    fichas, en el mismo orden que los centroides
  fotos/<n>.jpg
La web no enlaza ni carga nada de fuera.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/ecapa-int8.onnx.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
from lda import centroides, entrenar_lda, proyectar  # noqa: E402


def main():
    datos = np.load(RAIZ / "data" / "indice.npz")
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]
    fichas = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
    slugs = sorted(fichas)

    # la LDA aprende de los actores con 2+ clips (misma voz en grabaciones distintas)
    n_clips = {a: len(set(clips[y == a])) for a in slugs}
    sel = np.isin(y, [a for a in slugs if n_clips[a] >= 2])
    mu, W = entrenar_lda(X[sel], y[sel])
    C = centroides(proyectar(X, mu, W), y, slugs)

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
            "nombre": fichas[s]["nombre"],
            "foto": f"fotos/{i}.jpg" if foto.exists() else "",
            "trabajos": fichas[s].get("trabajos", []),
            "num_trabajos": fichas[s].get("num_trabajos", 0),
        })
    web = {"dim": int(X.shape[1]), "actores": actores}
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(slugs)} actores (LDA entrenada con {sel.sum()} trozos de "
          f"{sum(v >= 2 for v in n_clips.values())} actores) -> {salida}")


if __name__ == "__main__":
    main()
