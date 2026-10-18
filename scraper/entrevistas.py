"""Voz de los dobladores más conocidos sacada de entrevistas en podcasts y radio.

Pasos:
  .venv/bin/python scraper/entrevistas.py buscar [--max-actores 150]   # episodios con su nombre en el título
  .venv/bin/python scraper/entrevistas.py descargar                    # primeros 15 min de cada uno
  ESTAVOZ_ONNX=modelos/ecapa-int8.onnx .venv/bin/python scraper/entrevistas.py extraer
  (indexar) .venv/bin/python scraper/entrevistas.py validar          # quita los clips con la voz equivocada

Cómo se sabe qué voz es la del doblador: en cada episodio se separan las voces (agrupando trozos de 6 s
que se parecen). El presentador cambia de un programa a otro, pero el invitado es el mismo, así que la
voz que se repite en episodios de programas distintos es la suya. Si solo hay un episodio, se usa la
muestra que ya tenemos del doblador para elegir.
El resultado va a data/actores/<actor>/clips/entrevista_*.wav.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
ACTORES = RAIZ / "data" / "actores"
DIR = RAIZ / "data" / "entrevistas"
EPISODIOS = DIR / "episodios.json"
MAX_EPISODIOS = 4
MINUTOS = 15
MAX_BYTES = 40 << 20          # ~15-20 min de mp3

sesion = requests.Session()
sesion.headers["User-Agent"] = "EstavozBot/0.1 (+https://github.com/AndyLatimer/estavoz)"


def plano(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def actores_por_importancia(maximo: int, extra: list[str]) -> list[tuple[str, str]]:
    """Los dobladores con más trabajos (los más conocidos suelen estar ahí), más los pedidos a mano."""
    filas = []
    for f in ACTORES.glob("*/ficha.json"):
        ficha = json.loads(f.read_text(encoding="utf-8"))
        t = f.parent / "trabajos.json"
        n = len(json.loads(t.read_text(encoding="utf-8"))) if t.exists() else ficha.get("num_trabajos", 0)
        filas.append((n, f.parent.name, ficha["nombre"]))
    filas.sort(reverse=True)
    elegidos = [(s, n) for _, s, n in filas[:maximo]]
    vistos = {s for s, _ in elegidos}
    elegidos += [(s, n) for _, s, n in filas if s in extra and s not in vistos]
    return elegidos


def buscar(maximo: int, extra: list[str]):
    DIR.mkdir(parents=True, exist_ok=True)
    hechos = json.loads(EPISODIOS.read_text(encoding="utf-8")) if EPISODIOS.exists() else {}
    for slug, nombre in actores_por_importancia(maximo, extra):
        if slug in hechos:
            continue
        nom = plano(nombre)
        if len(nom.split()) < 2:
            hechos[slug] = []
            continue
        resultados = []
        for intento in range(3):
            r = sesion.get("https://itunes.apple.com/search", timeout=30, params={
                "term": nombre, "media": "podcast", "entity": "podcastEpisode", "country": "es", "limit": 50})
            try:
                resultados = r.json().get("results", [])
                break
            except ValueError:  # el directorio limita las peticiones seguidas
                time.sleep(15 * (intento + 1))
        elegidos, programas = [], set()
        for e in sorted(resultados, key=lambda e: -(e.get("trackTimeMillis") or 0)):
            titulo = e.get("trackName", "")
            minutos = (e.get("trackTimeMillis") or 0) / 60000
            if f" {nom} " not in f" {plano(titulo)} " or minutos < 5 or not e.get("episodeUrl"):
                continue
            if e.get("collectionId") in programas:   # mejor programas distintos (presentadores distintos)
                continue
            programas.add(e.get("collectionId"))
            elegidos.append({"titulo": titulo, "programa": e.get("collectionName", ""), "mp3": e["episodeUrl"]})
            if len(elegidos) == MAX_EPISODIOS:
                break
        hechos[slug] = elegidos
        print(f"{slug}: {len(elegidos)} episodios " + " | ".join(e["titulo"][:50] for e in elegidos), flush=True)
        EPISODIOS.write_text(json.dumps(hechos, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(3)


def descargar():
    hechos = json.loads(EPISODIOS.read_text(encoding="utf-8"))
    for slug, eps in hechos.items():
        for i, ep in enumerate(eps):
            destino = DIR / "audio" / f"{slug}__{i}.mp3"
            if destino.exists():
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            try:
                with sesion.get(ep["mp3"], timeout=60, stream=True, headers={"Range": f"bytes=0-{MAX_BYTES}"}) as r:
                    r.raise_for_status()
                    leido = 0
                    with open(destino.with_suffix(".part"), "wb") as f:
                        for trozo in r.iter_content(1 << 16):
                            f.write(trozo)
                            leido += len(trozo)
                            if leido >= MAX_BYTES:
                                break
                destino.with_suffix(".part").rename(destino)
                print(f"ok {destino.name} ({leido >> 20} MB)", flush=True)
            except Exception as e:
                print(f"  ! {slug} {i}: {str(e)[:120]}", flush=True)
            time.sleep(0.5)


def voces(h: np.ndarray, umbral: float = 0.45, minimo: int = 8) -> list[np.ndarray]:
    """Agrupa los trozos de un episodio por voz. Devuelve los índices de cada grupo grande."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(h) < minimo:
        return []
    etiquetas = fcluster(linkage(h, method="average", metric="cosine"), t=1 - umbral, criterion="distance")
    grupos = [np.flatnonzero(etiquetas == e) for e in np.unique(etiquetas)]
    return sorted([g for g in grupos if len(g) >= minimo], key=len, reverse=True)


def extraer():
    import soundfile as sf
    from voz import SAMPLE_RATE, cargar_audio, huellas

    hechos = json.loads(EPISODIOS.read_text(encoding="utf-8"))
    datos = np.load(RAIZ / "data" / "indice_ecapa.npz")   # muestras actuales, con el mismo modelo (ECAPA)
    X, y = datos["vectores"], datos["etiquetas"]
    paso = 6 * SAMPLE_RATE
    analizados_f = DIR / "analizados.json"
    analizados = set(json.loads(analizados_f.read_text())) if analizados_f.exists() else set()

    def marcar(slug):
        analizados.add(slug)
        analizados_f.write_text(json.dumps(sorted(analizados)))

    for slug, eps in hechos.items():
        if slug in analizados or list((ACTORES / slug / "clips").glob("entrevista_*.wav")) or not eps:
            continue
        episodios = []   # (audio, huellas, grupos)
        for i in range(len(eps)):
            f = DIR / "audio" / f"{slug}__{i}.mp3"
            if not f.exists():
                continue
            try:
                audio = cargar_audio(f, max_seg=MINUTOS * 60)
            except Exception:
                continue
            h = huellas(audio)
            episodios.append((audio, h, voces(h)))
        if not episodios:
            marcar(slug)
            continue
        centro = lambda h, g: (lambda m: m / np.linalg.norm(m))(h[g].mean(0))
        candidatos = [(e, g) for e, (_, h, gs) in enumerate(episodios) for g in gs]
        if not candidatos:
            marcar(slug)
            continue
        # parecido de cada voz de cada episodio con las voces de los OTROS episodios
        puntos = []
        for e, g in candidatos:
            c = centro(episodios[e][1], g)
            otros = [max((float(c @ centro(episodios[e2][1], g2)) for g2 in episodios[e2][2]), default=0)
                     for e2 in range(len(episodios)) if e2 != e]
            repetida = np.mean(otros) if otros else 0.0
            ref = X[y == slug]
            muestra = float((ref @ c).mean()) if len(ref) else 0.0
            puntos.append((repetida, muestra, e, g))
        if len(episodios) >= 2:
            mejor = max(puntos, key=lambda p: p[0] + 0.3 * p[1])
            fiable = mejor[0] >= 0.5        # la misma voz aparece claramente en otro programa
        else:
            mejor = max(puntos, key=lambda p: p[1])
            fiable = mejor[1] >= 0.45       # solo un episodio: tiene que parecerse a la muestra
        repetida, muestra, e, g = mejor
        print(f"{slug}: {len(episodios)} episodios, voz elegida en ep. {e} ({len(g) * 6} s) "
              f"repetida {repetida:.2f} muestra {muestra:.2f} -> {'SÍ' if fiable else 'no'}", flush=True)
        if not fiable:
            marcar(slug)
            continue
        # se guardan los trozos de esa voz en todos los episodios donde aparece
        c = centro(episodios[e][1], g)
        for k, (audio, h, gs) in enumerate(episodios):
            trozos = np.flatnonzero(h @ c >= 0.5)
            if len(trozos) >= 5:
                voz_suya = np.concatenate([audio[i * paso:(i + 1) * paso] for i in trozos])
                sf.write(ACTORES / slug / "clips" / f"entrevista_{k}.wav", voz_suya[: 90 * SAMPLE_RATE], SAMPLE_RATE)
        marcar(slug)


def validar(minimo: float = 0.4):
    """Comprobación final con el índice (modelo nuevo): un clip de entrevista se queda solo si su voz
    se parece a la muestra original del doblador (eldoblaje/podcast). A veces la voz que se repite
    entre programas es la de un presentador o colaborador común, no la del doblador."""
    datos = np.load(RAIZ / "data" / "indice.npz")
    X, y, c = datos["vectores"], datos["etiquetas"], datos["clips"]
    limpio = np.array(["#" not in x for x in c])
    original = limpio & np.array(["entrevista_" not in x for x in c])
    centro = lambda m: (lambda v: v / np.linalg.norm(v))(X[m].mean(0))
    quitados = 0
    for clip in sorted(set(c[limpio & ~original])):
        actor = clip.split("/")[0]
        ref = original & (y == actor)
        parecido = float(centro(c == clip) @ centro(ref)) if ref.any() else 0.0
        if parecido < minimo:
            (ACTORES / clip).unlink(missing_ok=True)
            quitados += 1
            print(f"fuera {clip} (parecido {parecido:.2f})")
    print(f"{quitados} clips de entrevista quitados")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["buscar", "descargar", "extraer", "validar"])
    p.add_argument("--max-actores", type=int, default=150)
    p.add_argument("--extra", default="108-jose-luis-gil,205-claudio-serrano,32-eduardo-bosch")
    a = p.parse_args()
    if a.paso == "buscar":
        buscar(a.max_actores, a.extra.split(","))
    elif a.paso == "descargar":
        descargar()
    elif a.paso == "validar":
        validar()
    else:
        extraer()
