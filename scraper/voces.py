"""Más voz para cada doblador: entrevistas, programas de radio y podcasts donde sale hablando.

Pasos (todos se pueden cortar y reanudar):
  .venv/bin/python scraper/voces.py buscar              # episodios donde aparece su nombre (título o descripción)
  .venv/bin/python scraper/voces.py procesar [--hilos 1] # bajar, encontrar su voz, guardar, borrar el episodio
  (varios procesos a la vez con "procesar": cada uno reserva el siguiente episodio libre)
  .venv/bin/python scraper/voces.py indexar              # huellas (limpias y "#movil") de los clips nuevos
  .venv/bin/python scraper/voces.py juntar               # las añade a data/indice.npz

Cómo se sabe qué voz es la del doblador: muchos episodios son de homónimos (otro "José Luis Gil") o la voz
que más se oye es la del presentador. Así que en cada episodio se separan las voces (ECAPA, rápido) y cada
voz se pasa por el propio reconocedor de la web (ResNet100 + LDA, contra los 793 dobladores): solo se queda
si sale el doblador del episodio como el más parecido de todos. Si sale otro doblador cuyo nombre también
aparece en el episodio (programas de doblaje con varios invitados), se guarda para ese.
El resultado va a data/actores/<actor>/clips/voz_<episodio>.flac (como mucho 90 s por episodio).
"""
import argparse
import hashlib
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
DIR = RAIZ / "data" / "voces"
CANDIDATOS = DIR / "candidatos.json"
HECHOS = DIR / "hechos"              # un .json por episodio con lo que se decidió
HUELLAS = DIR / "huellas"            # un .npz por clip nuevo (paso indexar)
MAX_EPISODIOS = 6
MAX_POR_PROGRAMA = 2
MAX_BYTES = 12 << 20                 # ~10-12 min de mp3
MINUTOS = 10
SEG_POR_EPISODIO = 90
SEG_INDICE = 48                      # para las huellas bastan 8 trozos de cada clip
MAX_SEG_ACTOR = 6 * 60               # con 6 min de voz nueva por doblador ya vale
PALABRAS_DOBLAJE = re.compile(r"\b(doblaje|doblador|dobladora|dobla|voz|voces|locutor|locutora|actor de voz)\b")

sesion = requests.Session()
sesion.headers["User-Agent"] = "EstavozBot/0.1 (+https://github.com/AndyLatimer/estavoz)"


def plano(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def contiene(texto_plano: str, nombre_plano: str) -> bool:
    return f" {nombre_plano} " in f" {texto_plano} "


def fichas() -> dict:
    return json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))


def itunes(termino: str) -> list[dict] | None:
    """None si el directorio no contesta (corta un rato cuando se le pregunta mucho seguido)."""
    for intento in range(4):
        try:
            r = sesion.get("https://itunes.apple.com/search", timeout=30, params={
                "term": termino, "media": "podcast", "entity": "podcastEpisode", "country": "es", "limit": 200})
            return r.json().get("results", [])
        except (ValueError, requests.RequestException):
            time.sleep(60 * 2 ** intento)
    return None


def buscar():
    DIR.mkdir(parents=True, exist_ok=True)
    hechos = json.loads(CANDIDATOS.read_text(encoding="utf-8")) if CANDIDATOS.exists() else {}
    for slug, ficha in sorted(fichas().items(), key=lambda x: -int(x[1].get("num_trabajos") or 0)):
        if slug in hechos:
            continue
        nom = plano(ficha["nombre"])
        if len(nom.split()) < 2:
            hechos[slug] = []
            continue
        vistos, eps = set(), []
        for termino in (ficha["nombre"],):
            resultados = itunes(termino)
            if resultados is None:
                break
            for e in resultados:
                url = e.get("episodeUrl")
                titulo, desc = plano(e.get("trackName")), plano(e.get("description") or e.get("shortDescription"))
                minutos = (e.get("trackTimeMillis") or 0) / 60000
                if not url or url in vistos or minutos < 4 or not (contiene(titulo, nom) or contiene(desc, nom)):
                    continue
                vistos.add(url)
                todo = f"{titulo} {desc} {plano(e.get('collectionName'))}"
                prioridad = 2 * bool(PALABRAS_DOBLAJE.search(todo)) + contiene(titulo, nom)
                eps.append({"titulo": e.get("trackName", ""), "programa": e.get("collectionName", ""),
                            "id_programa": e.get("collectionId"), "mp3": url, "minutos": round(minutos),
                            "texto": f"{titulo} {desc}"[:3000], "prioridad": prioridad})
            time.sleep(6)   # el directorio corta si se le pregunta más de ~10 veces por minuto
        if resultados is None:
            print(f"{slug}: el directorio no contesta, se reintentará", flush=True)
            continue
        eps.sort(key=lambda e: (-e["prioridad"], -min(e["minutos"], 60)))
        elegidos, por_programa = [], {}
        for e in eps:
            if por_programa.get(e["id_programa"], 0) >= MAX_POR_PROGRAMA:
                continue
            por_programa[e["id_programa"]] = por_programa.get(e["id_programa"], 0) + 1
            elegidos.append(e)
            if len(elegidos) == MAX_EPISODIOS:
                break
        hechos[slug] = elegidos
        print(f"{slug}: {len(elegidos)} de {len(eps)} episodios", flush=True)
        CANDIDATOS.write_text(json.dumps(hechos, ensure_ascii=False), encoding="utf-8")


class Reconocedor:
    """Lo mismo que hace la web (index.html: proyectar + puntuar), con los datos exportados."""
    def __init__(self, hilos=1):
        import onnxruntime as ort
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = hilos
        self.ecapa = ort.InferenceSession(str(RAIZ / "modelos" / "ecapa-int8.onnx"), opciones)
        self.r100 = ort.InferenceSession(str(RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"), opciones)
        datos = RAIZ / "frontend" / "datos"
        meta = json.loads((datos / "actores.json").read_text(encoding="utf-8"))
        self.ids = [a["id"] for a in meta["actores"]]
        d = self.dim = meta["dim"]
        lda = np.fromfile(datos / "lda.bin", "<f4")
        self.mu, self.W = lda[:d], lda[d:].reshape(d, d)
        self.C = np.fromfile(datos / "centroides.bin", "<f4").reshape(-1, d)
        troz = (datos / "trozos.bin").read_bytes()
        nA = len(self.ids) + 1
        self.inicio = np.frombuffer(troz, "<u4", nA)
        n = int(self.inicio[-1])
        escala = np.frombuffer(troz, "<f4", n, nA * 4)
        self.V = np.frombuffer(troz, np.int8, n * d, (nA + n) * 4).reshape(n, d) * escala[:, None]
        self.actor_de = np.repeat(np.arange(len(self.ids)), np.diff(self.inicio))

    @staticmethod
    def _huella(s, audio):
        v = s.run(None, {"wav": audio[None].astype(np.float32)})[0][0]
        return v / np.linalg.norm(v)

    def huella_ecapa(self, audio):
        return self._huella(self.ecapa, audio)

    def huella_r100(self, audio):
        return self._huella(self.r100, audio)

    def puntuar(self, huellas):
        q = (huellas - self.mu) @ self.W
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        media = (q @ self.C.T).mean(0)
        s = (q @ self.V.T).mean(0)
        mejores = np.full(len(self.ids), -1.0)
        orden = np.lexsort((-s, self.actor_de))
        for a in range(len(self.ids)):
            ini, fin = self.inicio[a], self.inicio[a + 1]
            if fin > ini:
                mejores[a] = s[orden[ini:min(fin, ini + 3)]].mean()
        return (media + mejores) / 2


def voces(h: np.ndarray, umbral: float = 0.45, minimo: int = 5) -> list[np.ndarray]:
    """Agrupa los trozos de un episodio por voz. Devuelve los índices de cada grupo grande."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(h) < minimo:
        return []
    etiquetas = fcluster(linkage(h, method="average", metric="cosine"), t=1 - umbral, criterion="distance")
    grupos = [np.flatnonzero(etiquetas == e) for e in np.unique(etiquetas)]
    return sorted([g for g in grupos if len(g) >= minimo], key=len, reverse=True)


def segundos_nuevos(slug: str) -> float:
    import soundfile as sf
    return sum(sf.info(str(f)).duration for f in (ACTORES / slug / "clips").glob("voz_*.flac"))


def procesar_episodio(rec, slug, ep, nombres_planos, ficha_nombres):
    import soundfile as sf
    from voz import SAMPLE_RATE, cargar_audio
    clave = hashlib.sha1(ep["mp3"].encode()).hexdigest()[:12]
    tmp = DIR / "tmp" / f"{clave}.mp3"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sesion.get(ep["mp3"], timeout=60, stream=True, headers={"Range": f"bytes=0-{MAX_BYTES}"}) as r:
            r.raise_for_status()
            leido = 0
            with open(tmp, "wb") as f:
                for trozo in r.iter_content(1 << 16):
                    f.write(trozo)
                    leido += len(trozo)
                    if leido >= MAX_BYTES:
                        break
        audio = cargar_audio(tmp, max_seg=MINUTOS * 60)
    except Exception as e:
        return {"error": str(e)[:200]}
    finally:
        tmp.unlink(missing_ok=True)
    paso = 6 * SAMPLE_RATE
    trozos = [audio[i:i + paso] for i in range(0, len(audio) - paso + 1, paso)]
    # fuera silencios y casi silencios
    energia = np.array([np.sqrt(np.mean(t ** 2)) for t in trozos]) if trozos else np.zeros(0)
    validos = np.flatnonzero(energia > max(1e-3, 0.25 * np.median(energia))) if len(energia) else []
    if len(validos) < 5:
        return {"voces": [], "nota": "corto"}
    h = np.stack([rec.huella_ecapa(trozos[i]) for i in validos])
    grupos = voces(h)[:3]
    texto = plano(ep["titulo"]) + " " + ep.get("texto", "")
    resultado, elegidas = [], {}
    for g in grupos:
        c = h[g].mean(0)
        c /= np.linalg.norm(c)
        centrales = g[np.argsort(-(h[g] @ c))[:2]]   # 12 s, como una consulta de la web
        punt = rec.puntuar(np.stack([rec.huella_r100(trozos[validos[i]]) for i in centrales]))
        mejor = int(np.argmax(punt))
        puesto = int((punt > punt[rec.ids.index(slug)]).sum()) + 1 if slug in rec.ids else 9999
        info = {"segundos": int(len(g) * 6), "mejor": rec.ids[mejor], "parecido": round(float(punt[mejor]), 3),
                "puesto_actor": puesto, "parecido_actor": round(float(punt[rec.ids.index(slug)]), 3) if slug in rec.ids else None}
        resultado.append(info)
        # Cualquier voz sale parecida (0,3-0,4) a alguno de los 793, pero que el doblador del episodio
        # quede entre los 5 primeros por casualidad pasa menos de 1 vez de cada 100.
        if puesto <= 5 and info["parecido_actor"] >= 0.25:
            quien, orden = slug, (puesto, -info["parecido_actor"])
        # otro doblador nombrado en el episodio (programas de doblaje con varios invitados)
        elif rec.ids[mejor] != slug and punt[mejor] >= 0.3 and contiene(texto, nombres_planos.get(rec.ids[mejor], "@@")):
            quien, orden = rec.ids[mejor], (1, -float(punt[mejor]))
        else:
            continue
        if quien not in elegidas or orden < elegidas[quien][0]:   # una voz por doblador y episodio: la más clara
            elegidas[quien] = (orden, g, c, info)
    for quien, (_, g, c, info) in elegidas.items():
        suyos = g[(h[g] @ c) >= 0.55]
        voz = np.concatenate([trozos[validos[i]] for i in suyos])[: SEG_POR_EPISODIO * SAMPLE_RATE]
        destino = ACTORES / quien / "clips" / f"voz_{clave}.flac"
        if not destino.exists():
            sf.write(destino, voz, SAMPLE_RATE)
            info["guardado"] = f"{quien}/{destino.name} ({len(voz) // SAMPLE_RATE} s)"
    return {"voces": resultado}


def procesar(hilos: int):
    rec = Reconocedor(hilos)
    candidatos = json.loads(CANDIDATOS.read_text(encoding="utf-8"))
    fs = fichas()
    nombres_planos = {s: plano(f["nombre"]) for s, f in fs.items() if len(plano(f["nombre"]).split()) >= 2}
    HECHOS.mkdir(parents=True, exist_ok=True)
    # primero los episodios que hablan de doblaje (38 % traen su voz); los que solo lo nombran de pasada
    # en la descripción casi nunca (7 %) y se saltan
    trabajo = [(-ep["prioridad"], k, slug, ep) for slug, eps in candidatos.items() for k, ep in enumerate(eps) if ep["prioridad"]]
    trabajo.sort(key=lambda x: x[:2])
    for _, k, slug, ep in trabajo:
        hecho = HECHOS / f"{slug}__{k}.json"
        reserva = hecho.with_suffix(".reservado")
        if hecho.exists():
            continue
        try:
            reserva.open("x").close()
        except FileExistsError:
            continue
        if slug in fs and segundos_nuevos(slug) >= MAX_SEG_ACTOR:
            r = {"nota": "ya tiene bastante voz"}
        else:
            t0 = time.time()
            r = procesar_episodio(rec, slug, ep, nombres_planos, fs)
            r["tiempo"] = round(time.time() - t0)
        r["episodio"] = ep["titulo"]
        hecho.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        reserva.unlink(missing_ok=True)
        guardados = [v["guardado"] for v in r.get("voces", []) if "guardado" in v]
        print(f"{slug} #{k}: {', '.join(guardados) or r.get('nota') or r.get('error', '')[:60] or 'nada'}", flush=True)


def indexar(hilos: int):
    """Huellas de los clips nuevos, igual que backend/indexar.py --aumentar (limpio + "#movil")."""
    import os
    os.environ.setdefault("ESTAVOZ_ONNX", str(RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"))
    os.environ.setdefault("ESTAVOZ_HILOS", str(hilos))
    from simular_movil import movil_real as movil
    from voz import cargar_audio, huellas
    HUELLAS.mkdir(parents=True, exist_ok=True)
    for clip in sorted(ACTORES.glob("*/clips/voz_*.flac")):
        slug = clip.parent.parent.name
        destino = HUELLAS / f"{slug}__{clip.stem}.npz"
        reserva = destino.with_suffix(".reservado")
        if destino.exists():
            continue
        try:
            reserva.open("x").close()
        except FileExistsError:
            continue
        audio = cargar_audio(clip, max_seg=SEG_INDICE)
        rng = np.random.default_rng(int(hashlib.sha1(clip.name.encode()).hexdigest()[:8], 16))
        np.savez(destino, limpio=huellas(audio), movil=huellas(movil(audio, rng)))
        reserva.unlink(missing_ok=True)
        print(f"  {slug}/{clip.name}", flush=True)


def juntar():
    """Añade al índice las huellas de los clips nuevos que sigan existiendo (y quita las de los borrados)."""
    salida = RAIZ / "data" / "indice.npz"
    d = np.load(salida)
    X, y, c = d["vectores"], d["etiquetas"], d["clips"]
    viejos = np.array(["/voz_" not in x for x in c])
    X, y, c = X[viejos], y[viejos], c[viejos]
    nX, ny, nc = [X], list(y), list(c)
    for f in sorted(HUELLAS.glob("*.npz")):
        slug, nombre = f.stem.split("__")
        if not (ACTORES / slug / "clips" / f"{nombre}.flac").exists():
            continue
        h = np.load(f)
        for tipo, sufijo in (("limpio", ""), ("movil", "#movil")):
            if len(h[tipo]):
                nX.append(h[tipo]); ny += [slug] * len(h[tipo]); nc += [f"{slug}/{nombre}.flac{sufijo}"] * len(h[tipo])
    np.savez(salida, vectores=np.concatenate(nX), etiquetas=np.array(ny), clips=np.array(nc), separado=d["separado"])
    nuevos = sum("/voz_" in x and "#" not in x for x in nc)
    print(f"Índice: {len(ny)} trozos, {nuevos} limpios de voz nueva, de {len({s for s, x in zip(ny, nc) if '/voz_' in x})} dobladores")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["buscar", "procesar", "indexar", "juntar"])
    p.add_argument("--hilos", type=int, default=1)
    a = p.parse_args()
    if a.paso == "buscar":
        buscar()
    elif a.paso == "procesar":
        procesar(a.hilos)
    elif a.paso == "indexar":
        indexar(a.hilos)
    else:
        juntar()
