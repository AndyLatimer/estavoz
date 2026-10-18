"""Calcula las huellas de voz de todos los actores de data/actores/.

Estructura esperada:
  data/actores/<slug>/ficha.json   {"nombre": ..., "url": ..., "foto": ..., "trabajos": [...]}
  data/actores/<slug>/clips/*.mp3  (o wav, m4a, ogg...)

Uso: .venv/bin/python backend/indexar.py [--separar] [--nuevos]
  --nuevos solo calcula los clips que aún no están en el índice y los añade
  --aumentar añade también, por cada clip, sus huellas como si se hubiera grabado con el móvil
             delante de la tele (backend/simular_movil.py, con ecos y ruidos reales); su clip se marca con "#movil"
  --separar quita música y efectos con Demucs (más lento, más acierto con audio de películas)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simular_movil import movil_real as movil  # noqa: E402
from voz import cargar_audio, huellas  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
ACTORES = RAIZ / "data" / "actores"
SALIDA = RAIZ / "data" / "indice.npz"
MAX_SEG = 90  # de cada clip basta con minuto y medio
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--separar", action="store_true")
    p.add_argument("--nuevos", action="store_true")
    p.add_argument("--aumentar", action="store_true")
    p.add_argument("--cache", help="carpeta donde guardar las huellas de cada actor (para poder reanudar)")
    p.add_argument("--parte", help="k/n: uno de cada n actores; auto: reparto dinámico entre procesos")
    args = p.parse_args()
    separar = args.separar
    hechos = set()
    if args.nuevos and SALIDA.exists():
        previo = np.load(SALIDA)
        assert bool(previo["separado"]) == separar, "el índice existente se hizo con otro modo"
        hechos = set(previo["clips"])
    vectores, etiquetas, clips_id, fichas = [], [], [], {}
    rng = np.random.default_rng(1)
    k, n_partes = map(int, args.parte.split("/")) if args.parte and args.parte != "auto" else (0, 1)
    for num, carpeta in enumerate(sorted(p for p in ACTORES.iterdir() if p.is_dir())):
        ficha_path = carpeta / "ficha.json"
        if args.parte == "auto":   # reparto dinámico: cada proceso reserva el siguiente actor libre
            guardado = Path(args.cache) / f"{carpeta.name}.npz"
            reserva = guardado.with_suffix(".reservado")
            if guardado.exists():
                pass
            else:
                try:
                    reserva.parent.mkdir(parents=True, exist_ok=True)
                    reserva.open("x").close()
                except FileExistsError:
                    continue
        elif num % n_partes != k:
            continue
        if not ficha_path.exists():
            continue
        clips = [c for c in sorted((carpeta / "clips").glob("*")) if c.suffix.lower() in EXT_AUDIO]
        n = 0
        guardado = Path(args.cache) / f"{carpeta.name}.npz" if args.cache else None
        if guardado and guardado.exists():  # ya calculado antes de un corte
            g = np.load(guardado)
            if len(g["etiquetas"]):
                vectores.append(g["vectores"]); etiquetas += list(g["etiquetas"]); clips_id += list(g["clips"])
                fichas[carpeta.name] = json.loads(ficha_path.read_text(encoding="utf-8"))
            continue
        inicio, n_vect_inicio = len(etiquetas), len(vectores)
        for clip in clips:
            if f"{carpeta.name}/{clip.name}" in hechos:
                n += 1  # ya estaba en el índice
                continue
            try:
                audio = cargar_audio(clip, separar=separar, max_seg=MAX_SEG)
                h = huellas(audio)
            except Exception as e:  # clip roto: seguimos con el resto
                print(f"  ! {clip.name}: {e}")
                continue
            vectores.append(h)
            etiquetas += [carpeta.name] * len(h)
            clips_id += [f"{carpeta.name}/{clip.name}"] * len(h)
            n += len(h)
            if args.aumentar:
                hm = huellas(movil(audio, rng))
                vectores.append(hm)
                etiquetas += [carpeta.name] * len(hm)
                clips_id += [f"{carpeta.name}/{clip.name}#movil"] * len(hm)
        if n:
            fichas[carpeta.name] = json.loads(ficha_path.read_text(encoding="utf-8"))
        if guardado:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            v = vectores[n_vect_inicio:]
            v = np.concatenate(v) if v else np.zeros((0, 1), np.float32)
            np.savez(guardado, vectores=v, etiquetas=np.array(etiquetas[inicio:]), clips=np.array(clips_id[inicio:]))
            guardado.with_suffix(".reservado").unlink(missing_ok=True)
        print(f"{carpeta.name}: {len(clips)} clips, {n} trozos")

    if hechos:  # añadir a lo que ya había
        vectores.insert(0, previo["vectores"])
        etiquetas = list(previo["etiquetas"]) + etiquetas
        clips_id = list(previo["clips"]) + clips_id
    if args.parte:   # cada parte solo rellena la caché; el índice se monta con una pasada sin --parte
        return
    if not vectores:
        sys.exit("No hay clips que indexar en data/actores/")
    np.savez(SALIDA, vectores=np.concatenate(vectores), etiquetas=np.array(etiquetas),
             clips=np.array(clips_id), separado=separar)
    (RAIZ / "data" / "fichas.json").write_text(json.dumps(fichas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Índice guardado: {len(fichas)} actores, {len(etiquetas)} trozos")


if __name__ == "__main__":
    main()
