"""Más voz de cada doblador a partir de entrevistas en podcasts (feeds RSS públicos).

Pasos:
  .venv/bin/python scraper/podcasts.py buscar      # podcasts de doblaje (directorio de Apple) y episodios que nombran a un doblador
  .venv/bin/python scraper/podcasts.py descargar   # baja el mp3 de esos episodios
  .venv/bin/python scraper/podcasts.py extraer     # se queda solo con los trozos donde habla ese doblador

En una entrevista también hablan el presentador y otros invitados, así que "extraer" solo
guarda los trozos cuya voz se parece mucho a la huella que ya tenemos del doblador y en los
que, además, es el más parecido de todos. El resultado va a data/actores/<actor>/clips/podcast_*.wav.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

ACTORES = RAIZ / "data" / "actores"
DIR = RAIZ / "data" / "podcasts"
EPISODIOS = DIR / "episodios.json"
TERMINOS = ["doblaje", "actor de doblaje", "actores de doblaje", "actriz de doblaje", "doblaje entrevista",
            "doblaje España", "voces del doblaje", "dobladores", "locución doblaje", "el doblaje"]
MAX_EPISODIOS_POR_ACTOR = 3

sesion = requests.Session()
sesion.headers["User-Agent"] = "EstavozBot/0.1 (+https://github.com/AndyLatimer/estavoz)"


def plano(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def nombres_actores() -> dict[str, str]:
    """slug -> nombre normalizado ('eduardo bosch'). Solo nombres de 2+ palabras, para no confundir."""
    out = {}
    for f in ACTORES.glob("*/ficha.json"):
        n = plano(json.loads(f.read_text(encoding="utf-8"))["nombre"])
        if len(n.split()) >= 2 and len(n) >= 8:
            out[f.parent.name] = n
    return out


def feeds() -> dict[str, str]:
    encontrados = {}
    for t in TERMINOS:
        for intento in range(3):
            r = sesion.get("https://itunes.apple.com/search", timeout=30,
                           params={"media": "podcast", "country": "es", "limit": 200, "term": t})
            try:
                resultados = r.json().get("results", [])
                break
            except ValueError:  # el directorio limita peticiones seguidas: esperar y reintentar
                print(f"  ! '{t}': HTTP {r.status_code}, reintento")
                time.sleep(10 * (intento + 1))
        else:
            continue
        for p in resultados:
            if p.get("feedUrl"):
                encontrados[p["feedUrl"]] = p["collectionName"]
        time.sleep(3)
    return encontrados


def episodios_de(url: str) -> list[dict]:
    try:
        r = sesion.get(url, timeout=60)
        raiz = ET.fromstring(r.content)
    except Exception as e:
        print(f"  ! {url}: {e}")
        return []
    out = []
    for item in raiz.iter("item"):
        enc = item.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        out.append({"titulo": item.findtext("title") or "", "mp3": enc.get("url")})
    return out


def buscar():
    DIR.mkdir(parents=True, exist_ok=True)
    nombres = nombres_actores()
    fs = feeds()
    print(f"{len(fs)} podcasts encontrados")
    elegidos = {}  # slug -> [episodios]
    for url, podcast in fs.items():
        eps = episodios_de(url)
        for ep in eps:
            t = f" {plano(ep['titulo'])} "
            for slug, n in nombres.items():
                if f" {n} " in t:
                    elegidos.setdefault(slug, []).append({**ep, "podcast": podcast})
        time.sleep(0.5)
    for slug in elegidos:
        elegidos[slug] = elegidos[slug][:MAX_EPISODIOS_POR_ACTOR]
    EPISODIOS.write_text(json.dumps(elegidos, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(v) for v in elegidos.values())
    print(f"{len(elegidos)} dobladores con episodios ({total} episodios):")
    for slug, eps in sorted(elegidos.items()):
        print(f"  {slug}: " + " | ".join(e["titulo"][:60] for e in eps))


def descargar():
    elegidos = json.loads(EPISODIOS.read_text(encoding="utf-8"))
    for slug, eps in elegidos.items():
        for i, ep in enumerate(eps):
            destino = DIR / "audio" / f"{slug}__{i}.mp3"
            if destino.exists():
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            try:
                with sesion.get(ep["mp3"], timeout=120, stream=True) as r:
                    r.raise_for_status()
                    with open(destino.with_suffix(".part"), "wb") as f:
                        for trozo in r.iter_content(1 << 16):
                            f.write(trozo)
                destino.with_suffix(".part").rename(destino)
                print(f"ok {destino.name} ({destino.stat().st_size >> 20} MB)")
            except Exception as e:
                print(f"  ! {slug}: {e}")
            time.sleep(1)


def extraer(umbral: float, max_seg: int):
    import soundfile as sf
    from lda import centroides, entrenar_lda, proyectar
    from voz import SAMPLE_RATE, cargar_audio, huellas

    datos = np.load(RAIZ / "data" / "indice.npz")
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]
    slugs = sorted(set(y))
    multi = [a for a in slugs if len(set(clips[y == a])) >= 2]
    mu, W = entrenar_lda(X[np.isin(y, multi)], y[np.isin(y, multi)])
    C = centroides(proyectar(X, mu, W), y, slugs)
    pos = {s: i for i, s in enumerate(slugs)}

    hechos_f = DIR / "analizados.json"
    hechos = set(json.loads(hechos_f.read_text())) if hechos_f.exists() else set()
    for mp3 in sorted((DIR / "audio").glob("*.mp3")):
        slug = mp3.stem.split("__")[0]
        destino = ACTORES / slug / "clips" / f"podcast_{mp3.stem.split('__')[1]}.wav"
        if destino.exists() or mp3.name in hechos or slug not in pos:
            continue
        audio = cargar_audio(mp3, max_seg=max_seg)
        h = huellas(audio)                      # una huella cada 6 s
        if not len(h):
            continue
        punt = proyectar(h, mu, W) @ C.T
        suyo = punt[:, pos[slug]]
        buenos = (suyo >= umbral) & (punt.argmax(1) == pos[slug])
        paso = 6 * SAMPLE_RATE
        voz_suya = np.concatenate([audio[i * paso:(i + 1) * paso] for i in np.flatnonzero(buenos)] or [np.zeros(0)])
        print(f"{mp3.name}: {buenos.sum()}/{len(h)} trozos son de {slug} ({len(voz_suya) / SAMPLE_RATE:.0f} s)")
        if buenos.sum() >= 3:
            sf.write(destino, voz_suya[: 90 * SAMPLE_RATE], SAMPLE_RATE)
        hechos.add(mp3.name)
        hechos_f.write_text(json.dumps(sorted(hechos)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["buscar", "descargar", "extraer"])
    p.add_argument("--umbral", type=float, default=0.35, help="parecido mínimo con la huella del doblador")
    p.add_argument("--max-seg", type=int, default=1800, help="minutos de cada episodio a analizar (en segundos)")
    a = p.parse_args()
    {"buscar": buscar, "descargar": descargar}.get(a.paso, lambda: extraer(a.umbral, a.max_seg))()
