"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera frontend/datos/huellas.bin (float32, una fila de 192 por trozo) y
frontend/datos/actores.json (fichas + a qué actor pertenece cada fila) y
frontend/fotos/<n>.jpg. La web no enlaza ni carga nada de fuera.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/ecapa-int8.onnx.
"""
import json
import shutil
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent


def main():
    datos = np.load(RAIZ / "data" / "indice.npz")
    fichas = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
    slugs = sorted(fichas)
    pos = {s: i for i, s in enumerate(slugs)}
    salida = RAIZ / "frontend" / "datos"
    salida.mkdir(parents=True, exist_ok=True)
    datos["vectores"].astype("<f4").tofile(salida / "huellas.bin")
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
    web = {
        "dim": int(datos["vectores"].shape[1]),
        "actores": actores,
        "etiquetas": [pos[s] for s in datos["etiquetas"]],
    }
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(slugs)} actores, {len(web['etiquetas'])} huellas -> {salida}")


if __name__ == "__main__":
    main()
