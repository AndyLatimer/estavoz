"""Descarga fichas y muestras de voz de los actores de doblaje de eldoblaje.com.

Pasos (se pueden repetir; lo ya descargado no se vuelve a pedir):
  .venv/bin/python scraper/eldoblaje.py listar             # lista de todos los actores de doblaje
  .venv/bin/python scraper/eldoblaje.py fichas [--max N]   # fichas + clips de voz de cada actor
  .venv/bin/python scraper/eldoblaje.py fotos              # fotos de los actores ya descargados
  .venv/bin/python scraper/eldoblaje.py trabajos           # lista completa de doblajes de cada actor con voz
  .venv/bin/python scraper/eldoblaje.py contar             # cuántos doblajes tiene cada actor SIN voz (para
                                                           # buscar su voz en otra parte, scraper/voces.py)

Resultado en data/actores/<id>-<nombre>/ (ficha.json y clips/), listo para backend/indexar.py.
Va despacio a propósito (PAUSA_SEG entre peticiones) para no cargar su servidor.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.eldoblaje.com/datos/"
PAUSA_SEG = 0.7
LETRAS = list("abcdefghijklmnopqrstuvwxyz") + ["ñ", "num"]
CARPETAS_AUDIO = ["Muestras", "Saludos", "Muestras3", "Muestras4"]
MAX_TRABAJOS = 15

RAIZ = Path(__file__).resolve().parent.parent
ACTORES = RAIZ / "data" / "actores"
LISTA = RAIZ / "data" / "eldoblaje_lista.json"

sesion = requests.Session()
sesion.headers["User-Agent"] = "EstavozBot/0.1 (+https://github.com/AndyLatimer/estavoz)"


def pedir(url: str) -> requests.Response | None:
    """GET con pausa y reintentos. La web redirige a http://, así que pedimos https sin seguir redirecciones."""
    for intento in range(3):
        time.sleep(PAUSA_SEG)
        try:
            r = sesion.get(url, timeout=60, allow_redirects=False)
        except requests.RequestException as e:
            print(f"  ! {url}: {e}")
            time.sleep(5 * (intento + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        print(f"  ! {url}: HTTP {r.status_code}")
        time.sleep(5 * (intento + 1))
    return None


def html(url: str) -> BeautifulSoup | None:
    r = pedir(url)
    if r is None:
        return None
    return BeautifulSoup(r.content.decode("latin-1"), "html.parser")


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:60]


def listar():
    actores = json.loads(LISTA.read_text(encoding="utf-8")) if LISTA.exists() else {}
    for letra in LETRAS:
        pag, total = 1, 1
        while pag <= total:
            sopa = html(f"{BASE}ListedResults.asp?letter={letra}&ocupacion=Doblaje&pag={pag}")
            if sopa is None:
                break
            for a in sopa.select('a[href^="FichaActorDoblaje.asp?id="]'):
                actores[a["href"].split("id=")[1]] = a.get_text(strip=True)
            paginas = [int(m) for m in re.findall(r"pag=(\d+)", str(sopa))]
            total = max(paginas, default=1)
            print(f"{letra} pág {pag}/{total}: {len(actores)} actores en total")
            pag += 1
        LISTA.write_text(json.dumps(actores, ensure_ascii=False, indent=0), encoding="utf-8")


def nombre_de(sopa: BeautifulSoup, respaldo: str) -> str:
    titulo = sopa.title.get_text(strip=True) if sopa.title else ""
    nombre = titulo.split(" - Ficha")[0].strip()
    return nombre or respaldo


def trabajos_de(sopa: BeautifulSoup) -> list[dict]:
    trabajos = []
    for fila in sopa.find_all("tr"):
        celdas = fila.find_all("td", recursive=False)
        if len(celdas) != 4:
            continue
        original = celdas[0].find("a", href=re.compile(r"^FichaActorOriginal"))
        obra = celdas[1].find("a", href=re.compile(r"^FichaPelicula"))
        if not (original and obra):
            continue
        anio = celdas[3].get_text(strip=True)
        trabajos.append({
            "obra": obra.get_text(" ", strip=True),
            "actor_original": original.get_text(" ", strip=True).title(),
            "personaje": celdas[2].get_text(" ", strip=True),
            "anio": int(anio) if anio.isdigit() else None,
        })
    trabajos.sort(key=lambda t: t["anio"] or 0, reverse=True)
    return trabajos


def muestras_de(sopa: BeautifulSoup) -> list[dict]:
    muestras = {}
    for a in sopa.find_all("a", href=re.compile(r"^(%s)/\d+\.mp3$" % "|".join(CARPETAS_AUDIO))):
        texto = a.get_text(" ", strip=True)
        if texto or a["href"] not in muestras:
            muestras[a["href"]] = texto
    return [{"ruta": ruta, "titulo": titulo} for ruta, titulo in muestras.items()]


def ficha(id_: str, nombre_lista: str, sin_audio: bool = False) -> str:
    """Descarga un actor. Devuelve 'ok', 'sin_audio' o 'error'.
    Con sin_audio=True guarda la ficha aunque no tenga muestras (su voz sale de scraper/voces.py)."""
    sopa = html(f"{BASE}FichaActorDoblaje.asp?id={id_}")
    if sopa is None:
        return "error"
    muestras = muestras_de(sopa)
    if not muestras and not sin_audio:
        return "sin_audio"

    nombre = nombre_de(sopa, nombre_lista)
    carpeta = ACTORES / f"{id_}-{slug(nombre)}"
    (carpeta / "clips").mkdir(parents=True, exist_ok=True)
    for m in muestras:
        destino = carpeta / "clips" / f"{m['ruta'].split('/')[0]}.mp3"
        if not destino.exists():
            r = pedir(BASE + m["ruta"])
            if r is not None and r.headers.get("content-type", "").startswith("audio"):
                destino.write_bytes(r.content)

    tiene_foto = sopa.find("img", src=f"FotosActores/{id_}.jpg") is not None
    if tiene_foto:
        descargar_foto(id_, carpeta)
    trabajos = trabajos_de(sopa)
    datos = {
        "nombre": nombre,
        "id_eldoblaje": int(id_),
        "url": f"https://www.eldoblaje.com/datos/FichaActorDoblaje.asp?id={id_}",
        "foto": f"https://www.eldoblaje.com/datos/FotosActores/{id_}.jpg" if tiene_foto else "",
        "muestras": [m["titulo"] for m in muestras],
        "trabajos": [
            f"{t['personaje'] or t['actor_original']} en {t['obra']}" + (f" ({t['anio']})" if t["anio"] else "")
            for t in trabajos[:MAX_TRABAJOS]
        ],
        "num_trabajos": len(trabajos),
    }
    (carpeta / "ficha.json").write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    return "ok"


def descargar_foto(id_: str, carpeta: Path):
    destino = carpeta / "foto.jpg"
    if destino.exists():
        return
    r = pedir(f"{BASE}FotosActores/{id_}.jpg")
    if r is not None and r.headers.get("content-type", "").startswith("image"):
        destino.write_bytes(r.content)


def fotos():
    """Descarga la foto de los actores ya bajados que la tengan y aún no esté."""
    pendientes = []
    for ficha_path in sorted(ACTORES.glob("*/ficha.json")):
        datos = json.loads(ficha_path.read_text(encoding="utf-8"))
        if datos.get("foto") and not (ficha_path.parent / "foto.jpg").exists():
            pendientes.append((str(datos["id_eldoblaje"]), ficha_path.parent))
    for n, (id_, carpeta) in enumerate(pendientes, 1):
        descargar_foto(id_, carpeta)
        print(f"[{n}/{len(pendientes)}] {carpeta.name}")


def trabajos_de_obra(sopa: BeautifulSoup) -> list[dict]:
    """Como trabajos_de, pero guardando también el id de la obra en eldoblaje."""
    out = []
    for fila in sopa.find_all("tr"):
        celdas = fila.find_all("td", recursive=False)
        if len(celdas) != 4:
            continue
        original = celdas[0].find("a", href=re.compile(r"^FichaActorOriginal"))
        obra = celdas[1].find("a", href=re.compile(r"^FichaPelicula"))
        if not (original and obra):
            continue
        anio = celdas[3].get_text(strip=True)
        out.append({
            "id": int(obra["href"].split("id=")[1].split("&")[0]),
            "obra": obra.get_text(" ", strip=True),
            "actor_original": original.get_text(" ", strip=True).title(),
            "personaje": celdas[2].get_text(" ", strip=True),
            "anio": int(anio) if anio.isdigit() else None,
        })
    return out


def trabajos(hilos: int = 2):
    """Todas las páginas de trabajos de cada actor que tiene voz (data/actores/*/trabajos.json).

    Dos descargas a la vez (cada una con su pausa entre páginas): unas 2-3 páginas por segundo.
    """
    from concurrent.futures import ThreadPoolExecutor
    carpetas = [f.parent for f in sorted(ACTORES.glob("*/ficha.json")) if not (f.parent / "trabajos.json").exists()]
    with ThreadPoolExecutor(hilos) as ex:
        list(ex.map(lambda nc: trabajos_actor(*nc, len(carpetas)), enumerate(carpetas, 1)))


def trabajos_actor(n: int, carpeta: Path, total_actores: int):
    destino = carpeta / "trabajos.json"
    id_ = carpeta.name.split("-")[0]
    todos, pag, total = [], 1, 1
    while pag <= total:
        sopa = html(f"{BASE}FichaActorDoblaje.asp?id={id_}&pag={pag}&orden=")
        if sopa is None:
            break
        todos += trabajos_de_obra(sopa)
        paginas = [int(m) for m in re.findall(rf"FichaActorDoblaje\.asp\?id={id_}&(?:amp;)?pag=(\d+)", str(sopa))]
        total = max(paginas, default=1)
        pag += 1
    else:
        destino.write_text(json.dumps(todos, ensure_ascii=False), encoding="utf-8")
    print(f"[{n}/{total_actores}] {carpeta.name}: {len(todos)} trabajos en {total} páginas", flush=True)


def contar():
    """De los actores sin muestra de voz: nombre, número aproximado de doblajes y si tiene foto."""
    from concurrent.futures import ThreadPoolExecutor
    actores = json.loads(LISTA.read_text(encoding="utf-8"))
    hechos = json.loads((RAIZ / "data" / "eldoblaje_hechos.json").read_text())
    destino = RAIZ / "data" / "eldoblaje_sin_audio.json"
    hecho = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
    pendientes = [i for i, v in hechos.items() if v == "sin_audio" and i not in hecho]

    def uno(id_):
        sopa = html(f"{BASE}FichaActorDoblaje.asp?id={id_}")
        if sopa is None:
            return id_, None
        filas = len(trabajos_de(sopa))
        paginas = max([int(m) for m in re.findall(rf"FichaActorDoblaje\.asp\?id={id_}&(?:amp;)?pag=(\d+)", str(sopa))], default=1)
        foto = sopa.find("img", src=f"FotosActores/{id_}.jpg") is not None
        return id_, {"nombre": nombre_de(sopa, actores.get(id_, "")), "paginas": paginas, "filas": filas, "foto": foto}

    with ThreadPoolExecutor(2) as ex:
        for n, (id_, datos) in enumerate(ex.map(uno, pendientes), 1):
            if datos:
                hecho[id_] = datos
            if n % 50 == 0 or n == len(pendientes):
                destino.write_text(json.dumps(hecho, ensure_ascii=False), encoding="utf-8")
                print(f"[{n}/{len(pendientes)}]", flush=True)


def fichas(maximo: int | None):
    if not LISTA.exists():
        sys.exit("Primero: scraper/eldoblaje.py listar")
    actores = json.loads(LISTA.read_text(encoding="utf-8"))
    hechos_path = RAIZ / "data" / "eldoblaje_hechos.json"
    hechos = json.loads(hechos_path.read_text()) if hechos_path.exists() else {}
    pendientes = [i for i in actores if hechos.get(i) not in ("ok", "sin_audio")]
    if maximo:
        pendientes = pendientes[:maximo]
    for n, id_ in enumerate(pendientes, 1):
        hechos[id_] = ficha(id_, actores[id_])
        print(f"[{n}/{len(pendientes)}] {actores[id_]}: {hechos[id_]}")
        if n % 20 == 0 or n == len(pendientes):
            hechos_path.write_text(json.dumps(hechos))
    con_audio = sum(v == "ok" for v in hechos.values())
    print(f"Hecho. {con_audio} actores con muestras de voz.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["listar", "fichas", "fotos", "trabajos", "contar"])
    p.add_argument("--max", type=int, help="procesar como mucho N actores (para probar)")
    a = p.parse_args()
    {"listar": listar, "fotos": fotos, "trabajos": trabajos, "contar": contar}.get(a.paso, lambda: fichas(a.max))()
