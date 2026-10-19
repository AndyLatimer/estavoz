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


def bajar_trozos(ep):
    """Primeros minutos de un episodio en trozos de 6 s, sin silencios. Devuelve (clave, trozos) o un error."""
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
        return clave, str(e)[:200]
    finally:
        tmp.unlink(missing_ok=True)
    paso = 6 * SAMPLE_RATE
    trozos = [audio[i:i + paso] for i in range(0, len(audio) - paso + 1, paso)]
    # fuera silencios y casi silencios
    energia = np.array([np.sqrt(np.mean(t ** 2)) for t in trozos]) if trozos else np.zeros(0)
    validos = np.flatnonzero(energia > max(1e-3, 0.25 * np.median(energia))) if len(energia) else []
    return clave, [trozos[i] for i in validos]


def procesar_episodio(rec, slug, ep, nombres_planos, ficha_nombres):
    import soundfile as sf
    from voz import SAMPLE_RATE
    clave, trozos = bajar_trozos(ep)
    if isinstance(trozos, str):
        return {"error": trozos}
    if len(trozos) < 5:
        return {"voces": [], "nota": "corto"}
    h = np.stack([rec.huella_ecapa(t) for t in trozos])
    grupos = voces(h)[:3]
    texto = plano(ep["titulo"]) + " " + ep.get("texto", "")
    resultado, elegidas = [], {}
    for g in grupos:
        c = h[g].mean(0)
        c /= np.linalg.norm(c)
        centrales = g[np.argsort(-(h[g] @ c))[:2]]   # 12 s, como una consulta de la web
        punt = rec.puntuar(np.stack([rec.huella_r100(trozos[i]) for i in centrales]))
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
        voz = np.concatenate([trozos[i] for i in suyos])[: SEG_POR_EPISODIO * SAMPLE_RATE]
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


# --- Dobladores sin ninguna muestra en eldoblaje ----------------------------------------------------
# De ~9.600 dobladores sin muestra no hay voz con la que comparar, así que se busca una voz que se
# repita en entrevistas de programas distintos (el presentador cambia, el invitado no). Se confirma con
# la ResNet100 (parecido proyectado >= 0,45: dos entrevistas de la misma persona dan de mediana 0,52 y
# de personas distintas casi nunca pasan de 0,39) y se descarta si esa voz ya es de un doblador conocido
# (presentadores que también doblan).
NUEVOS = DIR / "nuevos.json"
NUEVOS_HECHOS = DIR / "nuevos_hechos"
# sin muestra con la que comparar, el contexto importa más: "voz" a secas sale en cualquier sitio
# (economistas, físicos y psiquiatras con el mismo nombre que un doblador se colaban)
DOBLAJE_ESTRICTO = re.compile(r"\b(doblaje|doblajes|doblador|dobladora|dobladores|dobladoras|doblatge|"
                              r"actor de voz|actriz de voz|actor de doblaje|actriz de doblaje)\b")


def doblajes_aprox(d: dict) -> int:
    return d["paginas"] * 50 if d["paginas"] > 1 else d["filas"]


def buscar_nuevos(maximo: int, minimo_doblajes: int):
    sin = json.loads((RAIZ / "data" / "eldoblaje_sin_audio.json").read_text(encoding="utf-8"))
    hechos = json.loads(NUEVOS.read_text(encoding="utf-8")) if NUEVOS.exists() else {}
    orden = sorted(sin.items(), key=lambda x: -doblajes_aprox(x[1]))
    orden = [(i, d) for i, d in orden if doblajes_aprox(d) >= minimo_doblajes][:maximo]
    for id_, d in orden:
        if id_ in hechos:
            continue
        d = {**d, "nombre": re.sub(r"\s*\([^)]*\)", "", d["nombre"]).strip()}   # "Fernando (nano) Castro"
        nom = plano(d["nombre"])
        if len(nom.split()) < 2:
            hechos[id_] = {"nombre": d["nombre"], "eps": []}
            continue
        resultados = itunes(d["nombre"])
        # con nombres corrientes ("Daniel García") los homónimos tapan las entrevistas de doblaje
        if resultados is not None and not any(contiene(plano(e.get("trackName")), nom) and DOBLAJE_ESTRICTO.search(
                plano(f"{e.get('trackName')} {e.get('description')} {e.get('collectionName')}")) for e in resultados):
            time.sleep(6)
            mas = itunes(f"{d['nombre']} doblaje")
            resultados = None if mas is None else resultados + mas
        if resultados is None:
            print(f"{d['nombre']}: el directorio no contesta, se reintentará", flush=True)
            continue
        eps, programas = [], set()
        for e in sorted(resultados, key=lambda e: -bool(DOBLAJE_ESTRICTO.search(plano(
                f"{e.get('trackName')} {e.get('description')} {e.get('collectionName')}")))):
            minutos = (e.get("trackTimeMillis") or 0) / 60000
            # su nombre en el título (es el invitado), un programa distinto cada vez
            if not e.get("episodeUrl") or minutos < 8 or not contiene(plano(e.get("trackName")), nom) \
                    or e.get("collectionId") in programas:
                continue
            programas.add(e.get("collectionId"))
            texto = plano(f"{e.get('trackName')} {e.get('description')} {e.get('collectionName')}")
            eps.append({"titulo": e.get("trackName", ""), "programa": e.get("collectionName", ""), "mp3": e["episodeUrl"],
                        "doblaje": bool(DOBLAJE_ESTRICTO.search(texto))})
            if len(eps) == 4:
                break
        hechos[id_] = {"nombre": d["nombre"], "doblajes": doblajes_aprox(d), "eps": eps}
        print(f"{d['nombre']} ({doblajes_aprox(d)} doblajes): {len(eps)} programas", flush=True)
        NUEVOS.write_text(json.dumps(hechos, ensure_ascii=False), encoding="utf-8")
        time.sleep(6)


def procesar_nuevo(rec, id_, cand):
    import soundfile as sf
    import eldoblaje
    from voz import SAMPLE_RATE
    if not any(ep.get("doblaje") for ep in cand["eps"]):
        return {"nota": "ningún episodio habla de doblaje"}
    episodios, de_doblaje = [], []   # (clave, trozos, huellas ECAPA, grupos); si el episodio habla de doblaje
    for ep in cand["eps"]:
        clave, trozos = bajar_trozos(ep)
        if isinstance(trozos, str) or len(trozos) < 5:
            continue
        h = np.stack([rec.huella_ecapa(t) for t in trozos])
        episodios.append((clave, trozos, h, voces(h)[:4]))
        de_doblaje.append(ep.get("doblaje", False))
    if len(episodios) < 2:
        return {"nota": "menos de 2 episodios útiles"}
    centro = lambda v: v / np.linalg.norm(v)
    # la pareja de voces (de episodios distintos) que más se parecen
    mejor = None
    for a in range(len(episodios)):
        for b in range(a + 1, len(episodios)):
            if not (de_doblaje[a] or de_doblaje[b]):   # un homónimo entrevistado dos veces no cuenta
                continue
            for ga in episodios[a][3]:
                for gb in episodios[b][3]:
                    s = float(centro(episodios[a][2][ga].mean(0)) @ centro(episodios[b][2][gb].mean(0)))
                    if s > 0.97:   # el mismo audio publicado en dos programas: no es una voz que se repita
                        continue
                    if mejor is None or s > mejor[0]:
                        mejor = (s, a, ga, b, gb)
    if mejor is None or mejor[0] < 0.5:
        return {"nota": "ninguna voz se repite", "parecido_ecapa": round(mejor[0], 3) if mejor else None}
    s_ecapa, a, ga, b, gb = mejor

    def r100(e, g):
        _, trozos, h, _ = episodios[e]
        c = centro(h[g].mean(0))
        return np.stack([rec.huella_r100(trozos[i]) for i in g[np.argsort(-(h[g] @ c))[:2]]])
    ha, hb = r100(a, ga), r100(b, gb)
    pa = centro(np.mean([centro(v) for v in (ha - rec.mu) @ rec.W], 0))
    pb = centro(np.mean([centro(v) for v in (hb - rec.mu) @ rec.W], 0))
    s_r100 = float(pa @ pb)
    punt = rec.puntuar(np.concatenate([ha, hb]))
    conocido = rec.ids[int(np.argmax(punt))]
    info = {"parecido_ecapa": round(s_ecapa, 3), "parecido_r100": round(s_r100, 3),
            "conocido_mas_parecido": conocido, "parecido_conocido": round(float(punt.max()), 3)}
    if s_r100 < 0.45:
        return {**info, "nota": "no se confirma"}
    if punt.max() >= 0.45:
        return {**info, "nota": "es la voz de un doblador que ya conocemos"}
    # ficha de eldoblaje (nombre, foto, trabajos) aunque no tenga muestras
    if eldoblaje.ficha(id_, cand["nombre"], sin_audio=True) != "ok":
        return {**info, "nota": "no se pudo bajar su ficha"}
    carpeta = next(ACTORES.glob(f"{id_}-*"))
    c = centro(episodios[a][2][ga].mean(0) + episodios[b][2][gb].mean(0))
    guardados = []
    for clave, trozos, h, _ in episodios:
        suyos = np.flatnonzero(h @ c >= 0.55)
        if len(suyos) >= 5:
            voz = np.concatenate([trozos[i] for i in suyos])[: SEG_POR_EPISODIO * SAMPLE_RATE]
            sf.write(carpeta / "clips" / f"voz_{clave}.flac", voz, SAMPLE_RATE)
            guardados.append(f"voz_{clave}.flac ({len(voz) // SAMPLE_RATE} s)")
    return {**info, "actor": carpeta.name, "guardados": guardados}


def procesar_nuevos(hilos: int, esperar: bool = False):
    """Con esperar=True no termina: vuelve a mirar cada minuto si buscar_nuevos ha encontrado más."""
    rec = Reconocedor(hilos)
    NUEVOS_HECHOS.mkdir(parents=True, exist_ok=True)
    while True:
        if not procesar_nuevos_pendientes(rec) and not esperar:
            return
        time.sleep(60)


def procesar_nuevos_pendientes(rec) -> int:
    candidatos = json.loads(NUEVOS.read_text(encoding="utf-8"))
    orden = sorted(((i, c) for i, c in candidatos.items() if len(c["eps"]) >= 2), key=lambda x: -x[1].get("doblajes", 0))
    hechos = 0
    for id_, cand in orden:
        hecho = NUEVOS_HECHOS / f"{id_}.json"
        reserva = hecho.with_suffix(".reservado")
        if hecho.exists():
            continue
        try:
            reserva.open("x").close()
        except FileExistsError:
            continue
        t0 = time.time()
        r = procesar_nuevo(rec, id_, cand)
        r["tiempo"] = round(time.time() - t0)
        hecho.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        reserva.unlink(missing_ok=True)
        print(f"{cand['nombre']}: {r.get('actor', '')} {r.get('guardados', '')} {r.get('nota', '')}", flush=True)
        hechos += 1
    return hechos


# --- Podcasts de doblaje: un invitado por episodio --------------------------------------------------
# En "La voz de tu vida", "VozCast", "Dóblame mucho"... cada episodio entrevista a un doblador (sale en el
# título). El presentador es el mismo en todos, así que su voz se reconoce porque se repite entre
# episodios; la otra voz principal es la del invitado. Así vale un solo episodio, también para dobladores
# sin muestra. Los invitados que ya conocemos sirven para comprobar que se elige bien la voz.
FEEDS = {
    "La voz de tu vida": "https://anchor.fm/s/2b183bf4/podcast/rss",
    "VozCast": "https://anchor.fm/s/efd39b64/podcast/rss",
    "Dóblame mucho": "https://feeds.ivoox.com/feed_fg_f11460317_filtro_1.xml",
    "EL ÚLTIMO TAKE": "https://feeds.ivoox.com/feed_fg_f1870633_filtro_1.xml",
    "En sincronía": "https://rss.buzzsprout.com/1112315.rss",
    "CharletaCast": "https://anchor.fm/s/10f89d5cc/podcast/rss",
    "Contando con Ellas": "https://anchor.fm/s/ff5d1560/podcast/rss",
    "La Hora Pokémon": "https://feeds.ivoox.com/feed_fg_f1728172_filtro_1.xml",
}
PROGRAMAS = DIR / "programas"


def nombres_eldoblaje() -> dict:
    """id de eldoblaje -> nombre normalizado ("APELLIDO, NOMBRE (apodo)" -> "nombre apellido")."""
    lista = json.loads((RAIZ / "data" / "eldoblaje_lista.json").read_text(encoding="utf-8"))
    out = {}
    for i, v in lista.items():
        ap, _, no = v.partition(",")
        n = plano(re.sub(r"\(.*?\)", "", f"{no} {ap}"))
        if len(n.split()) >= 2:
            out[i] = n
    return out


def ids_conocidos() -> dict:
    """id de eldoblaje -> carpeta, de los dobladores que ya están en el índice."""
    return {str(json.loads((ACTORES / s / "ficha.json").read_text(encoding="utf-8"))["id_eldoblaje"]): s for s in fichas()}


def buscar_programas(minimo_episodios: int = 4):
    """Episodios cuyo título nombra a un solo doblador (conocido o con 50+ doblajes), en los programas de
    FEEDS y en los de programas/feeds.json (búsqueda de podcasts de cine, anime, videojuegos...).
    De los programas que no son de doblaje solo se quedan los que tienen varios episodios así."""
    import xml.etree.ElementTree as ET
    nombres = nombres_eldoblaje()
    sin_audio = json.loads((RAIZ / "data" / "eldoblaje_sin_audio.json").read_text(encoding="utf-8"))
    conocidos = ids_conocidos()
    validos = {i for i in nombres if i in conocidos or doblajes_aprox(sin_audio.get(i, {"paginas": 1, "filas": 0})) >= 50}
    por_nombre = {}
    for i in validos:
        por_nombre.setdefault(nombres[i], []).append(i)
    extra = PROGRAMAS / "feeds.json"
    feeds = {**(json.loads(extra.read_text(encoding="utf-8")) if extra.exists() else {}), **FEEDS}
    episodios = []
    for prog, url in feeds.items():
        try:
            raiz = ET.fromstring(sesion.get(url, timeout=60).content)
        except Exception as e:
            print(f"  ! {prog}: {str(e)[:80]}")
            continue
        del_programa = []
        for it in raiz.iter("item"):
            enc = it.find("enclosure")
            if enc is None or not enc.get("url"):
                continue
            t = f" {plano(it.findtext('title'))} "
            ids = sorted({i for n, l in por_nombre.items() if f" {n} " in t for i in l})
            if len(ids) == 1:   # un solo invitado con nombre de doblador
                del_programa.append({"programa": prog, "titulo": it.findtext("title"), "mp3": enc.get("url"), "id": ids[0]})
        if prog in FEEDS or len({e["id"] for e in del_programa}) >= minimo_episodios:
            episodios += del_programa
            print(f"{prog}: {len(del_programa)} episodios", flush=True)
    PROGRAMAS.mkdir(parents=True, exist_ok=True)
    (PROGRAMAS / "episodios.json").write_text(json.dumps(episodios, ensure_ascii=False), encoding="utf-8")
    print(f"{len(episodios)} episodios con un doblador en el título")


def analizar_programas(hilos: int, max_conocidos: int = 40):
    """Voces de cada episodio: duración, huella ECAPA y huella proyectada de la ResNet100 (y su audio)."""
    import soundfile as sf
    from voz import SAMPLE_RATE
    rec = Reconocedor(hilos)
    conocidos = ids_conocidos()
    episodios = json.loads((PROGRAMAS / "episodios.json").read_text(encoding="utf-8"))
    # todos los de dobladores sin muestra y, por programa, unos cuantos de conocidos para comprobar
    vistos, elegidos = {}, []
    for ep in episodios:
        if ep["id"] in conocidos:
            vistos[ep["programa"]] = vistos.get(ep["programa"], 0) + 1
            if vistos[ep["programa"]] > max_conocidos:
                continue
        elegidos.append(ep)
    for ep in elegidos:
        clave = hashlib.sha1(ep["mp3"].encode()).hexdigest()[:12]
        hecho = PROGRAMAS / f"{clave}.npz"
        reserva = hecho.with_suffix(".reservado")
        if hecho.exists():
            continue
        try:
            reserva.open("x").close()
        except FileExistsError:
            continue
        _, trozos = bajar_trozos(ep)
        datos = {"n": np.array(0)}
        if not isinstance(trozos, str) and len(trozos) >= 5:
            h = np.stack([rec.huella_ecapa(t) for t in trozos])
            for k, g in enumerate(voces(h)[:4]):
                c = h[g].mean(0)
                c /= np.linalg.norm(c)
                r100 = np.stack([rec.huella_r100(trozos[i]) for i in g[np.argsort(-(h[g] @ c))[:2]]])
                suyos = g[(h[g] @ c) >= 0.55]
                sf.write(PROGRAMAS / f"{clave}_{k}.flac", np.concatenate([trozos[i] for i in suyos])[: SEG_POR_EPISODIO * SAMPLE_RATE], SAMPLE_RATE)
                datos.update({f"ecapa{k}": c, f"r100_{k}": r100, f"seg{k}": np.array(len(g) * 6)})
                datos["n"] = np.array(k + 1)
        np.savez(hecho, **datos)
        reserva.unlink(missing_ok=True)
        print(f"{ep['programa']}: {ep['titulo'][:60]} -> {int(datos['n'])} voces", flush=True)


def elegir_invitados(aplicar: bool = False):
    """Quita en cada programa la voz que se repite (presentador) y se queda con la principal que queda.

    Con los invitados conocidos se comprueba cada programa: solo es de fiar si en la mayoría se eligió la
    voz más parecida a su muestra (en programas que no son de doblaje, el nombre del título puede ser de
    un homónimo). De los programas de fiar se guarda la voz de los invitados nuevos y, de los conocidos,
    la que se confirma (su doblador entre los 5 más parecidos). Con aplicar=False solo lo cuenta."""
    import shutil
    import eldoblaje
    rec = Reconocedor()
    conocidos = ids_conocidos()
    # los que solo tienen voz de estos mismos programas no sirven para comprobar
    solo_programas = {i for i, s in conocidos.items()
                      if not [c for c in (ACTORES / s / "clips").iterdir() if not c.name.startswith("voz_")]}
    lista = json.loads((RAIZ / "data" / "eldoblaje_lista.json").read_text(encoding="utf-8"))
    sin_audio = json.loads((RAIZ / "data" / "eldoblaje_sin_audio.json").read_text(encoding="utf-8"))
    episodios = json.loads((PROGRAMAS / "episodios.json").read_text(encoding="utf-8"))
    centro = lambda v: v / np.linalg.norm(v)
    proy = lambda x: centro(np.mean([centro(v) for v in (x - rec.mu) @ rec.W], 0))
    analizados = []
    for ep in episodios:
        clave = hashlib.sha1(ep["mp3"].encode()).hexdigest()[:12]
        f = PROGRAMAS / f"{clave}.npz"
        if f.exists():
            d = np.load(f)
            voces_ep = [{"k": k, "r100": d[f"r100_{k}"], "p": proy(d[f"r100_{k}"]), "seg": int(d[f"seg{k}"])}
                        for k in range(int(d["n"]))]
            analizados.append((ep, clave, voces_ep))
    decisiones, prueba = [], {}   # prueba[programa] = [bien, mal]
    for ep, clave, vs in analizados:
        otros = [vs2 for e2, c2, vs2 in analizados if e2["programa"] == ep["programa"] and c2 != clave]
        for v in vs:   # presentador, sintonía...: está en 3 o más episodios más del programa
            v["repite"] = sum(any(float(v["p"] @ o["p"]) >= 0.6 for o in vs2) for vs2 in otros)
        candidatas = [v for v in vs if v["repite"] < 3 and v["seg"] >= 60]
        if not candidatas:
            continue
        inv = max(candidatas, key=lambda v: v["seg"])
        punt = rec.puntuar(inv["r100"])
        if ep["id"] in conocidos:
            i = rec.ids.index(conocidos[ep["id"]])
            puesto = int((punt > punt[i]).sum()) + 1
            if ep["id"] not in solo_programas:
                # ¿alguna otra voz del episodio se parecía más a su muestra? entonces se eligió mal
                otras = [int((q > q[i]).sum()) + 1 for q in (rec.puntuar(v["r100"]) for v in vs if v is not inv)]
                r = prueba.setdefault(ep["programa"], [0, 0])
                r[0 if puesto <= min(otras, default=9999) else 1] += 1
            if puesto <= 5:
                decisiones.append((ep, clave, inv, conocidos[ep["id"]]))
            continue
        if punt.max() >= 0.45:   # nuevo: que no sea la voz de un doblador conocido
            continue
        if doblajes_aprox(sin_audio.get(ep["id"], {"paginas": 1, "filas": 0})) < 50:   # actores, youtubers...
            continue
        decisiones.append((ep, clave, inv, None))
    fiables = {prog for prog, (b, m) in prueba.items() if b >= 2 and b >= 0.7 * (b + m)}
    for prog, (b, m) in sorted(prueba.items(), key=lambda x: -sum(x[1])):
        print(f"{prog}: bien {b}, mal {m} -> {'de fiar' if prog in fiables else 'no'}")
    nuevos = extra = 0
    for ep, clave, inv, carpeta in decisiones:
        if ep["programa"] not in fiables:
            continue
        if carpeta is None:
            nuevos += 1
            print(f"  nuevo: {lista[ep['id']]} ({ep['programa']}), {inv['seg']} s")
            if not aplicar or eldoblaje.ficha(ep["id"], lista[ep["id"]], sin_audio=True) != "ok":
                continue
            carpeta = next(ACTORES.glob(f"{ep['id']}-*")).name
        else:
            extra += 1
        destino = ACTORES / carpeta / "clips" / f"voz_{clave}.flac"
        if aplicar and not destino.exists():
            shutil.copy(PROGRAMAS / f"{clave}_{inv['k']}.flac", destino)
    print(f"{nuevos} dobladores nuevos y {extra} voces más de dobladores conocidos")


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
    # dobladores que solo tienen voz de entrevistas (procesar_nuevos): su ficha entra en fichas.json
    fs = fichas()
    altas = [x for x in sorted(set(ny)) if x not in fs]
    for slug in altas:
        fs[slug] = json.loads((ACTORES / slug / "ficha.json").read_text(encoding="utf-8"))
    if altas:
        (RAIZ / "data" / "fichas.json").write_text(json.dumps(fs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(altas)} dobladores nuevos: {', '.join(fs[x]['nombre'] for x in altas)}")
    nuevos = sum("/voz_" in x and "#" not in x for x in nc)
    print(f"Índice: {len(ny)} trozos, {nuevos} limpios de voz nueva, de {len({s for s, x in zip(ny, nc) if '/voz_' in x})} dobladores")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["buscar", "procesar", "indexar", "juntar", "buscar_nuevos", "procesar_nuevos",
                                       "buscar_programas", "analizar_programas", "elegir_invitados"])
    p.add_argument("--hilos", type=int, default=1)
    p.add_argument("--max", type=int, default=300, help="buscar_nuevos: cuántos dobladores sin muestra mirar")
    p.add_argument("--minimo", type=int, default=100, help="buscar_nuevos: doblajes mínimos")
    p.add_argument("--esperar", action="store_true", help="procesar_nuevos: seguir esperando candidatos nuevos")
    p.add_argument("--aplicar", action="store_true", help="elegir_invitados: guardar la voz de los nuevos")
    a = p.parse_args()
    if a.paso == "buscar":
        buscar()
    elif a.paso == "procesar":
        procesar(a.hilos)
    elif a.paso == "indexar":
        indexar(a.hilos)
    elif a.paso == "buscar_programas":
        buscar_programas()
    elif a.paso == "analizar_programas":
        analizar_programas(a.hilos)
    elif a.paso == "elegir_invitados":
        elegir_invitados(a.aplicar)
    elif a.paso == "buscar_nuevos":
        buscar_nuevos(a.max, a.minimo)
    elif a.paso == "procesar_nuevos":
        procesar_nuevos(a.hilos, a.esperar)
    else:
        juntar()
