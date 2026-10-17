"""Mide cuánto acierta Estavoz con los clips descargados.

Para cada actor con 2 o más clips, aparta uno, toma 8 s de él como si fuera la
grabación del móvil, y busca entre todos los demás clips de todos los actores.

Uso: .venv/bin/python backend/evaluar.py [--max-actores N]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voz import SAMPLE_RATE, cargar_audio, huellas  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data" / "cache_huellas"
SEPARAR = False
MAX_SEG = 90
CONSULTA_SEG = 8


def huellas_clip(clip: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Huellas del clip entero y de varias consultas de 8 s sacadas de él (con caché)."""
    import os
    modo = ("separado" if SEPARAR else "normal") + ("-onnx" if os.environ.get("ESTAVOZ_ONNX") else "")
    cache = CACHE / modo / f"{clip.parent.parent.name}__{clip.stem}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["todo"], [d[k] for k in d.files if k.startswith("q")]
    audio = cargar_audio(clip, separar=SEPARAR, max_seg=MAX_SEG)
    todo = huellas(audio)
    paso = CONSULTA_SEG * SAMPLE_RATE
    inicios = range(0, max(len(audio) - paso, 1), paso)
    consultas = [huellas(audio[i:i + paso]) for i in list(inicios)[:3]]
    consultas = [c for c in consultas if len(c)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, todo=todo, **{f"q{i}": c for i, c in enumerate(consultas)})
    return todo, consultas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-actores", type=int)
    p.add_argument("--solo", help="usar solo estos actores (lista separada por comas)")
    p.add_argument("--separar", action="store_true", help="quitar música y efectos con Demucs")
    args = p.parse_args()
    global SEPARAR
    SEPARAR = args.separar

    carpetas = sorted(c for c in (RAIZ / "data" / "actores").iterdir() if (c / "clips").exists())
    random.seed(0)
    if args.solo:
        carpetas = [c for c in carpetas if c.name in args.solo.split(",")]
    if args.max_actores:
        carpetas = random.sample(carpetas, min(args.max_actores, len(carpetas)))

    clips = []  # (actor, huellas_todo, consultas)
    for n, c in enumerate(carpetas, 1):
        for clip in sorted((c / "clips").glob("*.mp3")):
            try:
                todo, qs = huellas_clip(clip)
            except Exception as e:
                print(f"  ! {clip}: {e}")
                continue
            if len(todo):
                clips.append((c.name, todo, qs))
        if n % 25 == 0:
            print(f"  huellas: {n}/{len(carpetas)} actores")

    actores = sorted({a for a, _, _ in clips})
    por_actor = {a: sum(1 for x, _, _ in clips if x == a) for a in actores}
    print(f"{len(actores)} actores, {len(clips)} clips")

    top1 = top3 = total = 0
    for i, (actor, _, consultas) in enumerate(clips):
        if por_actor[actor] < 2:
            continue
        # índice = todos los clips menos este
        vec = np.concatenate([t for j, (_, t, _) in enumerate(clips) if j != i])
        eti = np.array([a for j, (a, t, _) in enumerate(clips) if j != i for _ in range(len(t))])
        for q in consultas:
            sims = (q @ vec.T).mean(axis=0)
            puntos = {}
            for a in np.unique(eti):
                s = np.sort(sims[eti == a])[::-1]
                puntos[a] = s[:3].mean()
            orden = sorted(puntos, key=puntos.get, reverse=True)
            total += 1
            top1 += orden[0] == actor
            top3 += actor in orden[:3]

    print(f"\n{total} consultas de {CONSULTA_SEG} s entre {len(actores)} actores:")
    print(f"  acierta a la primera: {100 * top1 / total:.1f}%")
    print(f"  está entre los 3 primeros: {100 * top3 / total:.1f}%")


if __name__ == "__main__":
    main()
