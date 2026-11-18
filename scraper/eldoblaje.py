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
from collections import Counter
from html import unescape
from html.parser import HTMLParser
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # respaldo con parser estándar de Python

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
from filmografia import (  # noqa: E402
    clasificar_tipo_obra,
    elegir_trabajos_destacados,
    nombre_original,
    titulo_base as _limpiar_titulo_obra,
)

BASE = "https://www.eldoblaje.com/datos/"
PAUSA_SEG = 0.7
LETRAS = list("abcdefghijklmnopqrstuvwxyz") + ["ñ", "num"]
CARPETAS_AUDIO = ["Muestras", "Saludos", "Muestras3", "Muestras4"]
MAX_TRABAJOS = 15
_MARCAS_SIN_ORIGINAL = {"(voz)", "(desconocido)", "desconocido", "sin acreditar", "-", "n/a"}

RAIZ = Path(__file__).resolve().parent.parent

def _limpiar_html(fragmento: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", fragmento or "")
    return re.sub(r"\s+", " ", unescape(txt)).strip()


def html_texto(url: str) -> str | None:
    r = pedir(url)
    if r is None:
        return None
    return r.content.decode("latin-1", errors="replace")


class _TablaParser(HTMLParser):
    """Extractor ligero de filas <tr><td>...</td></tr> sin depender de bs4."""

    def __init__(self):
        super().__init__()
        self.filas: list[list[str]] = []
        self._en_tr = False
        self._en_td = False
        self._fila: list[str] = []
        self._celda: list[str] = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "tr":
            self._en_tr = True
            self._fila = []
        elif t == "td" and self._en_tr:
            self._en_td = True
            self._celda = []
        elif self._en_td:
            attr_txt = " ".join(f'{k}="{v}"' for k, v in attrs if v is not None)
            self._celda.append(f"<{tag} {attr_txt}>" if attr_txt else f"<{tag}>")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t == "td" and self._en_td:
            self._en_td = False
            self._fila.append("".join(self._celda))
        elif t == "tr" and self._en_tr:
            self._en_tr = False
            if self._fila:
                self.filas.append(self._fila)
        elif self._en_td:
            self._celda.append(f"</{tag}>")

    def handle_data(self, data):
        if self._en_td:
            self._celda.append(data)

    def handle_entityref(self, name):
        if self._en_td:
            self._celda.append(f"&{name};")

    def handle_charref(self, name):
        if self._en_td:
            self._celda.append(f"&#{name};")


def _extraer_filas_html(doc_html: str) -> list[list[str]]:
    p = _TablaParser()
    p.feed(doc_html)
    return p.filas


def _parsear_trabajos_html(doc_html: str, con_id: bool = False) -> list[dict]:
    out = []
    for celdas in _extraer_filas_html(doc_html):
        if len(celdas) != 4:
            continue
        m_obra = re.search(r'<a[^>]+href=["\']FichaPelicula\.asp\?id=(\d+)[^"\']*["\'][^>]*>(.*?)</a>', celdas[1], re.I | re.S)
        if not m_obra:
            continue
        m_orig = re.search(r'<a[^>]+href=["\']FichaActorOriginal[^"\']*["\'][^>]*>(.*?)</a>', celdas[0], re.I | re.S)
        if m_orig:
            orig = _limpiar_html(m_orig.group(1)).title()
        else:
            txt0 = _limpiar_html(celdas[0])
            if not txt0 or txt0.lower() in _MARCAS_SIN_ORIGINAL or (txt0.startswith("(") and txt0.endswith(")")):
                orig = ""
            else:
                orig = txt0.title()
        anio_txt = _limpiar_html(celdas[3])
        item = {
            "obra": _limpiar_html(m_obra.group(2)),
            "actor_original": orig,
            "personaje": _limpiar_html(celdas[2]),
            "anio": int(anio_txt) if anio_txt.isdigit() else None,
        }
        if con_id:
            item["id"] = int(m_obra.group(1))
        out.append(item)
    return out


def _muestras_html(doc_html: str) -> list[dict]:
    muestras: dict[str, str] = {}
    patron = re.compile(r'<a[^>]+href=["\']((?:%s)/\d+\.mp3)["\'][^>]*>(.*?)</a>' % "|".join(CARPETAS_AUDIO), re.I | re.S)
    for ruta, cuerpo in patron.findall(doc_html):
        texto = _limpiar_html(cuerpo)
        if texto or ruta not in muestras:
            muestras[ruta] = texto
    return [{"ruta": r, "titulo": t} for r, t in muestras.items()]


def _nombre_html(doc_html: str, respaldo: str) -> str:
    m = re.search(r"<title>(.*?)</title>", doc_html, re.I | re.S)
    if m:
        nombre = _limpiar_html(m.group(1)).split(" - Ficha")[0].strip()
        if nombre:
            return nombre
    return respaldo

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


def html(url: str):
    txt = html_texto(url)
    if txt is None:
        return None
    if BeautifulSoup is None:
        return txt
    return BeautifulSoup(txt, "html.parser")


def nombre_de(sopa, respaldo: str) -> str:
    if isinstance(sopa, str):
        return _nombre_html(sopa, respaldo)
    titulo = sopa.title.get_text(strip=True) if sopa.title else ""
    nombre = titulo.split(" - Ficha")[0].strip()
    return nombre or respaldo


def _extraer_original(celda) -> str:
    original = celda.find("a", href=re.compile(r"^FichaActorOriginal"))
    if original is not None:
        return original.get_text(" ", strip=True).title()
    texto = re.sub(r"\s+", " ", celda.get_text(" ", strip=True)).strip()
    if not texto or texto.lower() in _MARCAS_SIN_ORIGINAL or (texto.startswith("(") and texto.endswith(")")):
        return ""
    return texto.title()


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:60]


def _extraer_enlaces_actores(sopa) -> list[tuple[str, str]]:
    if isinstance(sopa, str):
        return [
            (m.group(1), _limpiar_html(m.group(2)))
            for m in re.finditer(r'<a[^>]+href=["\'](?:[^"\']*?)FichaActorDoblaje\.asp\?id=(\d+)[^"\']*["\'][^>]*>(.*?)</a>', sopa, re.I | re.S)
        ]
    return [
        (re.search(r"id=(\d+)", a["href"]).group(1), a.get_text(" ", strip=True))
        for a in sopa.select('a[href*="FichaActorDoblaje.asp?id="]')
        if re.search(r"id=(\d+)", a["href"])
    ]


def listar():
    actores = json.loads(LISTA.read_text(encoding="utf-8")) if LISTA.exists() else {}
    for letra in LETRAS:
        pag, total = 1, 1
        while pag <= total:
            sopa = html(f"{BASE}ListedResults.asp?letter={letra}&ocupacion=Doblaje&pag={pag}")
            if sopa is None:
                break
            for id_a, txt in _extraer_enlaces_actores(sopa):
                actores[id_a] = txt
            paginas = [int(m) for m in re.findall(r"pag=(\d+)", str(sopa))]
            total = max(paginas, default=1)
            print(f"{letra} pág {pag}/{total}: {len(actores)} actores en total")
            pag += 1
        LISTA.write_text(json.dumps(actores, ensure_ascii=False, indent=0), encoding="utf-8")


def trabajos_de(sopa) -> list[dict]:
    if isinstance(sopa, str):
        trabajos = _parsear_trabajos_html(sopa, con_id=False)
        trabajos.sort(key=lambda t: t["anio"] or 0, reverse=True)
        return trabajos
    trabajos = []
    for fila in sopa.find_all("tr"):
        celdas = fila.find_all("td", recursive=False)
        if len(celdas) != 4:
            continue
        obra = celdas[1].find("a", href=re.compile(r"^FichaPelicula"))
        if not obra:
            continue
        anio = celdas[3].get_text(strip=True)
        trabajos.append({
            "obra": obra.get_text(" ", strip=True),
            "actor_original": _extraer_original(celdas[0]),
            "personaje": celdas[2].get_text(" ", strip=True),
            "anio": int(anio) if anio.isdigit() else None,
        })
    trabajos.sort(key=lambda t: t["anio"] or 0, reverse=True)
    return trabajos


def muestras_de(sopa) -> list[dict]:
    if isinstance(sopa, str):
        return _muestras_html(sopa)
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

    tiene_foto = f"FotosActores/{id_}.jpg" in sopa if isinstance(sopa, str) else (sopa.find("img", src=f"FotosActores/{id_}.jpg") is not None)
    if tiene_foto:
        descargar_foto(id_, carpeta)
    trabajos = trabajos_de(sopa)
    datos = {
        "nombre": nombre,
        "id_eldoblaje": int(id_),
        "url": f"https://www.eldoblaje.com/datos/FichaActorDoblaje.asp?id={id_}",
        "foto": f"https://www.eldoblaje.com/datos/FotosActores/{id_}.jpg" if tiene_foto else "",
        "muestras": [m["titulo"] for m in muestras],
        "muestra": muestras[0]["ruta"] if muestras else "",
        "trabajos": elegir_trabajos_destacados(trabajos, max_trabajos=MAX_TRABAJOS),
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


def trabajos_de_obra(sopa) -> list[dict]:
    """Como trabajos_de, pero guardando también el id de la obra en eldoblaje."""
    if isinstance(sopa, str):
        return _parsear_trabajos_html(sopa, con_id=True)
    out = []
    for fila in sopa.find_all("tr"):
        celdas = fila.find_all("td", recursive=False)
        if len(celdas) != 4:
            continue
        obra = celdas[1].find("a", href=re.compile(r"^FichaPelicula"))
        if not obra:
            continue
        anio = celdas[3].get_text(strip=True)
        out.append({
            "id": int(obra["href"].split("id=")[1].split("&")[0]),
            "obra": obra.get_text(" ", strip=True),
            "actor_original": _extraer_original(celdas[0]),
            "personaje": celdas[2].get_text(" ", strip=True),
            "anio": int(anio) if anio.isdigit() else None,
        })
    return out


def descargar_trabajos_actor_id(id_: str, max_paginas: int = 120) -> list[dict]:
    """Descarga todas las páginas de filmografía de un actor por su id de eldoblaje."""
    todos, pag, total = [], 1, 1
    while pag <= total and pag <= max_paginas:
        sopa = html(f"{BASE}FichaActorDoblaje.asp?id={id_}&pag={pag}&orden=")
        if sopa is None:
            break
        todos += trabajos_de_obra(sopa)
        paginas = [int(m) for m in re.findall(rf"FichaActorDoblaje\.asp\?id={id_}&(?:amp;)?pag=(\d+)", str(sopa))]
        total = max(paginas, default=1)
        pag += 1
    return todos


def obtener_ficha_web(id_: str, nombre_respaldo: str = "", descargar_todos_trabajos: bool = True) -> dict | None:
    """Obtiene ficha completa (nombre, foto, muestras, trabajos completos) de eldoblaje.com sin guardar clips."""
    doc = html_texto(f"{BASE}FichaActorDoblaje.asp?id={id_}")
    if doc is None:
        return None
    nombre = _nombre_html(doc, nombre_respaldo)
    muestras = _muestras_html(doc)
    tiene_foto = f"FotosActores/{id_}.jpg" in doc
    trabajos = _parsear_trabajos_html(doc, con_id=True)
    paginas = [int(m) for m in re.findall(rf"FichaActorDoblaje\.asp\?id={id_}&(?:amp;)?pag=(\d+)", doc)]
    total_pags = max(paginas, default=1)
    if descargar_todos_trabajos and total_pags > 1:
        for pag in range(2, min(total_pags, 120) + 1):
            doc_p = html_texto(f"{BASE}FichaActorDoblaje.asp?id={id_}&pag={pag}&orden=")
            if doc_p is None:
                break
            trabajos += _parsear_trabajos_html(doc_p, con_id=True)
    return {
        "id": f"{id_}-{slug(nombre)}",
        "id_eldoblaje": int(id_),
        "nombre": nombre,
        "foto_url": f"{BASE}FotosActores/{id_}.jpg" if tiene_foto else "",
        "muestras": muestras,
        "trabajos_completos": trabajos,
    }


def descubrir_buscavoces(max_paginas: int = 10) -> list[str]:
    """Devuelve los IDs de actores con Ficha Extendida en buscavoces_results.asp."""
    ids, vistos = [], set()
    for pag in range(1, max_paginas + 1):
        sopa = html(f"{BASE}buscavoces_results.asp?pag={pag}")
        if sopa is None:
            break
        nuevos_en_pag = 0
        for id_a, _ in _extraer_enlaces_actores(sopa):
            if id_a not in vistos:
                vistos.add(id_a)
                ids.append(id_a)
                nuevos_en_pag += 1
        pags = [int(x) for x in re.findall(r"pag=(\d+)", str(sopa))]
        if pag >= max(pags, default=1) or nuevos_en_pag == 0:
            break
    return ids


_CACHE_LISTED_PAGS: dict[tuple[str, int], tuple[list[tuple[str, str]], int]] = {}


def _pagina_listed(letra: str, pag: int) -> tuple[list[tuple[str, str]], int] | None:
    """Descarga y cachea en memoria una página de `ListedResults.asp`."""
    clave = (letra, pag)
    if clave in _CACHE_LISTED_PAGS:
        return _CACHE_LISTED_PAGS[clave]
    sopa = html(f"{BASE}ListedResults.asp?letter={quote_plus(letra)}&ocupacion=Doblaje&pag={pag}")
    if sopa is None:
        return None
    enlaces = _extraer_enlaces_actores(sopa)
    paginas = [int(m) for m in re.findall(r"pag=(\d+)", str(sopa))]
    total = max(paginas, default=1)
    _CACHE_LISTED_PAGS[clave] = (enlaces, total)
    return enlaces, total


def buscar_actor_por_nombre(nombre: str) -> tuple[str, str] | None:
    """Busca un actor en eldoblaje.com por nombre completo y devuelve `(id, nombre_oficial)`."""
    def _plano(t: str) -> str:
        s = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    objetivo = _plano(nombre)
    partes = objetivo.split()
    if not partes:
        return None
    tokens_orig = [t for t in re.split(r"\s+", (nombre or "").strip()) if t]
    # En eldoblaje se lista por primer apellido ("APELLIDO, NOMBRE")
    letras_prueba: list[str] = []
    if len(tokens_orig) >= 2 and "-" in tokens_orig[-1]:
        prim_ap = _plano(tokens_orig[-1].split("-")[0])
        if prim_ap:
            letras_prueba.append(prim_ap[0])
    letras_prueba.append(partes[-1][0])
    for tok in reversed(partes[1:-1]):
        letras_prueba.append(tok[0])
    letras_prueba.append(partes[0][0])

    vistas = set()
    for letra in letras_prueba:
        if letra in vistas or letra not in LETRAS:
            continue
        vistas.add(letra)
        pag, total = 1, 1
        while pag <= total:
            res = _pagina_listed(letra, pag)
            if res is None:
                break
            enlaces, total = res
            for id_a, txt in enlaces:
                ap, _, no = txt.partition(",")
                nombre_fmt = f"{no.strip()} {ap.strip()}".strip() if "," in txt else txt
                sin_par = _plano(re.sub(r"\([^)]*\)", " ", nombre_fmt))
                if objetivo in (_plano(nombre_fmt), _plano(txt), sin_par):
                    return id_a, nombre_fmt
            pag += 1
    return None


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
        foto = f"FotosActores/{id_}.jpg" in sopa if isinstance(sopa, str) else (sopa.find("img", src=f"FotosActores/{id_}.jpg") is not None)
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
