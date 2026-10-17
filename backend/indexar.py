"""Calcula las huellas de voz de todos los actores de data/actores/.

Estructura esperada:
  data/actores/<slug>/ficha.json   {"nombre": ..., "url": ..., "foto": ..., "trabajos": [...]}
  data/actores/<slug>/clips/*.mp3  (o wav, m4a, ogg...)

Uso: .venv/bin/python backend/indexar.py [--separar]
  --separar quita música y efectos con Demucs (más lento, más acierto con audio de películas)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voz import cargar_audio, huellas  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
ACTORES = RAIZ / "data" / "actores"
SALIDA = RAIZ / "data" / "indice.npz"
MAX_SEG = 90  # de cada clip basta con minuto y medio
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--separar", action="store_true")
    separar = p.parse_args().separar
    vectores, etiquetas, fichas = [], [], {}
    for carpeta in sorted(p for p in ACTORES.iterdir() if p.is_dir()):
        ficha_path = carpeta / "ficha.json"
        if not ficha_path.exists():
            continue
        clips = [c for c in sorted((carpeta / "clips").glob("*")) if c.suffix.lower() in EXT_AUDIO]
        n = 0
        for clip in clips:
            try:
                h = huellas(cargar_audio(clip, separar=separar, max_seg=MAX_SEG))
            except Exception as e:  # clip roto: seguimos con el resto
                print(f"  ! {clip.name}: {e}")
                continue
            vectores.append(h)
            etiquetas += [carpeta.name] * len(h)
            n += len(h)
        if n:
            fichas[carpeta.name] = json.loads(ficha_path.read_text(encoding="utf-8"))
        print(f"{carpeta.name}: {len(clips)} clips, {n} trozos")

    if not vectores:
        sys.exit("No hay clips que indexar en data/actores/")
    np.savez(SALIDA, vectores=np.concatenate(vectores), etiquetas=np.array(etiquetas), separado=separar)
    (RAIZ / "data" / "fichas.json").write_text(json.dumps(fichas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Índice guardado: {len(fichas)} actores, {len(etiquetas)} trozos")


if __name__ == "__main__":
    main()
