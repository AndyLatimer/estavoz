"""Utilidades compartidas de filmografía, categorías y papeles destacados de dobladores."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any

_MINUSCULAS = set("de del la las el los lo un una unos unas y e o u en a al con por para sin su sus".split())
_PERSONAJES_GENERICOS = {
    "voces adicionales", "voces", "ambiente", "narrador", "narración", "locutor", "presentador",
    "policía", "agente", "doctor", "médico", "enfermera", "camarero", "camarera", "soldado",
    "guardia", "hombre", "mujer", "chico", "chica", "niño", "niña", "cliente", "periodista",
    "reportero", "juez", "abogado", "taxista", "recepcionista", "vendedor", "conductor",
    "voz", "desconocido",
}
_MARCAS_SIN_ORIGINAL = {"(voz)", "(desconocido)", "desconocido", "sin acreditar", "-", "n/a"}


def normalizar_texto(t: str) -> str:
    s = unicodedata.normalize("NFKD", t or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def titulo_base(t: str) -> str:
    """'LA JAULA [1ª temporada]' y 'LA JAULA (1x05)' -> 'LA JAULA'."""
    t = re.sub(r"\[[^\]]*\]|\(\d+x\d+\)", "", t or "")
    return re.sub(r"\s+", " ", t).strip(" :-")


def personaje_base(p: str) -> str:
    """'Buzz Lightyear (voz)' / 'Sofiane (1x05)' -> 'Buzz Lightyear'."""
    p = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", p or "")
    return re.sub(r"\s+", " ", p).strip(" .,:-")


def titulo_bonito(t: str) -> str:
    """'TOY STORY 2: LOS JUGUETES VUELVEN A LA CARGA' -> 'Toy Story 2: los Juguetes Vuelven a la Carga'."""
    if not t or not t.isupper():
        return t
    palabras = t.lower().split(" ")
    return " ".join(w if (k and w in _MINUSCULAS) else w[:1].upper() + w[1:] for k, w in enumerate(palabras))


def nombre_original(a: str) -> str:
    """'Bale, Christian' -> 'Christian Bale'."""
    a = (a or "").strip()
    if not a or a.lower() in _MARCAS_SIN_ORIGINAL:
        return ""
    if a.count(",") == 1:
        apellidos, nombre = (x.strip() for x in a.split(","))
        a = f"{nombre} {apellidos}".strip()
    return a.title() if a.isupper() else a


def clasificar_tipo_obra(obra: str, personaje: str = "") -> str:
    """Clasifica una obra en pelicula, serie, animacion, videojuego, documental u otros."""
    t = f" {obra or ''} ".lower()
    p = (personaje or "").lower()
    if "[videojuego" in t or "dlc]" in t:
        return "videojuego"
    if "[documental" in t or "[docureality" in t or "[programa tv" in t:
        return "documental"
    if "[cortometraje" in t or "[audiocuento" in t or "[audiolibro" in t or "[musical" in t:
        return "otros"
    if "[serie de animación" in t or "[animación" in t or "[anime" in t:
        return "animacion"
    if (
        "[serie tv" in t
        or "[miniserie" in t
        or re.search(r"\[\d+ª\s*temporada\]", t)
        or re.search(r"\(\d+x\d+", p)
    ):
        return "serie"
    return "pelicula"


def _normalizar_fila(fila: dict | list | tuple) -> tuple[str, str, int | None, str]:
    if isinstance(fila, dict):
        return (
            str(fila.get("obra") or ""),
            str(fila.get("personaje") or ""),
            int(fila["anio"]) if str(fila.get("anio") or "").isdigit() else None,
            str(fila.get("actor_original") or ""),
        )
    obra = str(fila[0] if len(fila) > 0 and fila[0] is not None else "")
    pers = str(fila[1] if len(fila) > 1 and fila[1] is not None else "")
    anio_raw = fila[2] if len(fila) > 2 else None
    anio = int(anio_raw) if isinstance(anio_raw, int) or (isinstance(anio_raw, str) and anio_raw.isdigit()) else None
    orig = str(fila[3] if len(fila) > 3 and fila[3] is not None else "")
    return obra, pers, anio, orig


def elegir_trabajos_destacados(
    filas: list[dict | list | tuple],
    popularidad_obras: dict[str, int] | None = None,
    max_trabajos: int = 15,
) -> list[str]:
    """Selecciona los papeles más reconocibles y variados (pelis, series, animación, videojuegos)."""
    norm = [_normalizar_fila(f) for f in filas]
    if not norm:
        return []
    conteo_orig = Counter(nombre_original(orig) for _, _, _, orig in norm if nombre_original(orig))
    pop = popularidad_obras or {}

    def puntuar(item: tuple[str, str, int | None, str]) -> tuple[float, int]:
        obra, pers_raw, anio, orig_raw = item
        pers = personaje_base(pers_raw)
        orig = nombre_original(orig_raw)
        base = titulo_base(obra).lower()
        n_reparto = pop.get(base, 0)
        es_generico = pers.lower() in _PERSONAJES_GENERICOS or len(pers) < 2
        es_episodico = bool(re.search(r"\(\d+x\d+", pers_raw))
        score = 0.0
        if orig:
            score += min(conteo_orig[orig], 25) * 2.5
        if n_reparto:
            score += min(n_reparto, 180) * 0.45
        if not es_generico:
            score += 18.0
        if not es_episodico:
            score += 8.0
        if pers and not es_generico:
            score += 4.0
        if anio and anio >= 1990:
            score += 3.0
        return score, anio or 0

    ordenados = sorted(norm, key=puntuar, reverse=True)
    vistos: set[str] = set()
    por_tipo: dict[str, list[tuple[str, str, int | None, str]]] = {
        "pelicula": [], "serie": [], "animacion": [], "videojuego": [], "documental": [], "otros": [],
    }
    unicos: list[tuple[str, str, int | None, str]] = []
    for item in ordenados:
        base = titulo_base(item[0]).lower()
        if not base or base in vistos:
            continue
        vistos.add(base)
        unicos.append(item)
        por_tipo[clasificar_tipo_obra(item[0], item[1])].append(item)

    elegidos: list[tuple[str, str, int | None, str]] = []
    elegidos_bases: set[str] = set()
    for tipo in ("pelicula", "serie", "animacion", "videojuego"):
        for item in por_tipo[tipo][:2]:
            b = titulo_base(item[0]).lower()
            if b not in elegidos_bases:
                elegidos.append(item)
                elegidos_bases.add(b)
    for item in unicos:
        if len(elegidos) >= max_trabajos:
            break
        b = titulo_base(item[0]).lower()
        if b not in elegidos_bases:
            elegidos.append(item)
            elegidos_bases.add(b)
    elegidos = elegidos[:max_trabajos]

    out = []
    for obra, pers, anio, orig in elegidos:
        quien = pers or nombre_original(orig) or "Voz"
        anio_txt = f" ({anio})" if anio else ""
        out.append(f"{quien} en {obra}{anio_txt}")
    return out


def resumen_actor_desde_papeles(
    filas: list[dict | list | tuple],
    popularidad_obras: dict[str, int] | None = None,
    max_trabajos: int = 15,
    max_vozh: int = 6,
) -> dict[str, Any]:
    """Calcula destacados, conteos por formato y voces originales habituales de un doblador."""
    norm = [_normalizar_fila(f) for f in filas]
    resumen_tipos: dict[str, int] = {
        "pelicula": 0, "serie": 0, "animacion": 0, "videojuego": 0, "documental": 0, "otros": 0,
    }
    orig_counts: Counter[str] = Counter()
    for obra, pers, _anio, orig in norm:
        resumen_tipos[clasificar_tipo_obra(obra, pers)] += 1
        nom = nombre_original(orig)
        if nom:
            orig_counts[nom] += 1

    vozh = [{"nombre": nom, "n": cnt} for nom, cnt in orig_counts.most_common(max_vozh) if cnt >= 2]
    return {
        "trabajos": elegir_trabajos_destacados(norm, popularidad_obras=popularidad_obras, max_trabajos=max_trabajos),
        "num_trabajos": len(norm),
        "resumen_tipos": {k: v for k, v in resumen_tipos.items() if v > 0},
        "vozh": vozh,
    }


def filtrar_papeles_actor(
    filas: list[dict | list | tuple],
    tipo: str = "",
    q: str = "",
    limite: int = 60,
    offset: int = 0,
) -> dict[str, Any]:
    """Filtra y pagina la filmografía completa de un actor por formato y consulta textual."""
    norm = [_normalizar_fila(f) for f in filas]
    q_norm = normalizar_texto(q)
    palabras = q_norm.split() if q_norm else []
    conteo_tipos = Counter()
    items = []
    for obra, pers, anio, orig in norm:
        cat = clasificar_tipo_obra(obra, pers)
        orig_fmt = nombre_original(orig)
        if palabras:
            texto = normalizar_texto(f"{obra} {pers} {orig_fmt}")
            if not all(p in texto for p in palabras):
                continue
        conteo_tipos[cat] += 1
        if tipo and cat != tipo:
            continue
        items.append({
            "obra": obra,
            "obra_bonita": titulo_bonito(titulo_base(obra)) or obra,
            "personaje": pers or orig_fmt or "Voz",
            "anio": anio,
            "actor_original": orig_fmt,
            "tipo": cat,
        })
    items.sort(key=lambda x: (x["anio"] or 0, x["obra"]), reverse=True)
    return {
        "total": len(items),
        "conteo_tipos": dict(conteo_tipos),
        "items": items[offset : offset + limite],
    }

