"""Herramienta de diagnóstico y limpieza de calidad del índice de voces de EstaVoz.

Permite:
1. Detectar y podar trozos intrusos (`filtrar_trozos_intrusos`): segmentos de 6 s donde habla
   otro interlocutor o predomina música/ruido (baja similitud con el centroide del propio actor).
2. Identificar parejas de «gemelos vocales» (máxima similitud entre centroides de actores distintos).
3. Detectar actores «imán» (*hubs* con alta similitud media a sus 30 vecinos más cercanos) y actores con baja cobertura.

Uso:
  .venv/bin/python backend/auditar_indice.py [--datos-web frontend/datos] [--salida data/evaluaciones/auditoria_indice.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from lda import normaliza  # noqa: E402


def filtrar_trozos_intrusos(
    Xp: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    slugs: list[str],
    umbral_outlier: float = 0.10,
    min_trozos_por_actor: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Devuelve una máscara booleana `conservar` eliminando trozos intrusos (`sim < umbral_outlier`).

    Garantiza que cada actor conserve siempre al menos `min_trozos_por_actor` (o todos si tenía menos).
    """
    n_total = len(Xp)
    conservar = np.ones(n_total, dtype=bool)
    intrusos: list[dict[str, Any]] = []

    Xp_norm = normaliza(Xp)
    C_norm = normaliza(C)
    sims_todos = Xp_norm @ C_norm.T  # (n_total, n_actores)

    for i_a, slug in enumerate(slugs):
        ix = np.flatnonzero(y == slug)
        if len(ix) == 0:
            continue
        sims_propio = sims_todos[ix, i_a]
        orden = np.argsort(-sims_propio)
        protegidos = set(ix[orden[:min_trozos_por_actor]].tolist())
        for pos_local, fila_global in enumerate(ix):
            sim_p = float(sims_propio[pos_local])
            if sim_p < umbral_outlier and int(fila_global) not in protegidos:
                conservar[fila_global] = False
                mejor_otro_idx = int(np.argmax(sims_todos[fila_global]))
                intrusos.append(
                    {
                        "actor_asignado": slug,
                        "fila_global": int(fila_global),
                        "similitud_propio": round(sim_p, 4),
                        "actor_mas_parecido": slugs[mejor_otro_idx],
                        "similitud_mas_parecido": round(float(sims_todos[fila_global, mejor_otro_idx]), 4),
                    }
                )

    intrusos.sort(key=lambda d: d["similitud_propio"])
    return conservar, intrusos


def auditar_datos_web(
    carpeta_datos: str | Path = RAIZ / "frontend" / "datos",
    umbral_outlier: float = 0.10,
    top_n: int = 15,
) -> dict[str, Any]:
    """Ejecuta el diagnóstico integral sobre los artefactos web (`actores.json`, `centroides.bin`, `trozos.bin`)."""
    carpeta = Path(carpeta_datos)
    meta = json.loads((carpeta / "actores.json").read_text(encoding="utf-8"))
    actores = meta["actores"]
    slugs = [a["id"] for a in actores]
    nombres = {a["id"]: a.get("nombre", a["id"]) for a in actores}
    dim = int(meta["dim"])
    nA = len(slugs)

    C = normaliza(np.fromfile(carpeta / "centroides.bin", dtype="<f4").reshape(nA, dim).astype(np.float32))
    raw_troz = (carpeta / "trozos.bin").read_bytes()
    inicio = np.frombuffer(raw_troz, dtype="<u4", count=nA + 1)
    n_total = int(inicio[-1])
    escala = np.frombuffer(raw_troz, dtype="<f4", count=n_total, offset=(nA + 1) * 4)
    q8 = np.frombuffer(raw_troz, dtype=np.int8, count=n_total * dim, offset=(nA + 1 + n_total) * 4).reshape(n_total, dim)
    V = normaliza((q8.astype(np.float32) * escala[:, None]).astype(np.float32))
    y = np.concatenate([[slugs[i]] * int(inicio[i + 1] - inicio[i]) for i in range(nA)])

    conservar, intrusos = filtrar_trozos_intrusos(V, y, C, slugs, umbral_outlier=umbral_outlier)

    # 2. Parejas de gemelos vocales (mayor similitud coseno entre centroides de actores distintos)
    S_cent = (C @ C.T).astype(np.float64)
    np.fill_diagonal(S_cent, -np.inf)
    parejas: list[dict[str, Any]] = []
    for i in range(nA):
        j = int(np.argmax(S_cent[i]))
        if i < j or S_cent[j, i] < S_cent[i, j]:
            continue
    # Extraer los top_n pares únicos (i < j)
    iu, ju = np.triu_indices(nA, k=1)
    vals = S_cent[iu, ju]
    top_idx = np.argsort(-vals)[:top_n]
    for idx_par in top_idx:
        i, j = int(iu[idx_par]), int(ju[idx_par])
        parejas.append(
            {
                "actor_1": slugs[i],
                "nombre_1": nombres[slugs[i]],
                "actor_2": slugs[j],
                "nombre_2": nombres[slugs[j]],
                "similitud_centroides": round(float(S_cent[i, j]), 4),
            }
        )

    # 3. Actores imán (hubness: media de similitud con sus 30 vecinos más cercanos)
    k_vec = min(30, max(1, nA - 1))
    top_vecinos = -np.partition(-S_cent, kth=k_vec - 1, axis=1)[:, :k_vec]
    media_vecinos = top_vecinos.mean(axis=1)
    n_trozos_actor = np.diff(inicio).astype(int)
    orden_hubs = np.argsort(-media_vecinos)[:top_n]
    actores_iman = [
        {
            "actor": slugs[int(i)],
            "nombre": nombres[slugs[int(i)]],
            "similitud_media_30_vecinos": round(float(media_vecinos[int(i)]), 4),
            "num_trozos": int(n_trozos_actor[int(i)]),
        }
        for i in orden_hubs
    ]

    # 4. Distribución de cobertura de trozos por actor (priorizando dobladores con más trabajos)
    actores_pocos_trozos = sorted(
        [
            {
                "actor": slugs[i],
                "nombre": nombres[slugs[i]],
                "num_trozos": int(n_trozos_actor[i]),
                "num_trabajos": int(actores[i].get("num_trabajos") or len(actores[i].get("trabajos") or [])),
            }
            for i in range(nA)
            if n_trozos_actor[i] <= 4
        ],
        key=lambda d: (d["num_trozos"], -d["num_trabajos"], d["actor"]),
    )

    return {
        "resumen": {
            "num_actores": nA,
            "num_trozos_totales": n_total,
            "mediana_trozos_por_actor": float(np.median(n_trozos_actor)),
            "media_trozos_por_actor": round(float(np.mean(n_trozos_actor)), 2),
            "umbral_outlier": umbral_outlier,
            "num_trozos_intrusos_detectados": int((~conservar).sum()),
            "pct_trozos_intrusos": round(100.0 * float((~conservar).mean()), 2),
            "num_actores_con_4_o_menos_trozos": len(actores_pocos_trozos),
        },
        "gemelos_vocales_top": parejas,
        "actores_iman_top": actores_iman,
        "trozos_intrusos_peores": intrusos[:top_n * 2],
        "actores_baja_cobertura": actores_pocos_trozos[:top_n * 2],
    }


_NOMBRES_FEM_INEQUIVOCOS = frozenset({
    "alba", "alicia", "amparo", "ana", "angela", "antonia", "araceli", "ascension", "aurora",
    "azucena", "beatriz", "begona", "belen", "blanca", "camila", "carla", "carmen", "carolina",
    "catalina", "cecilia", "celia", "clara", "claudia", "concepcion", "concha", "consuelo",
    "coral", "covadonga", "cristina", "daniela", "diana", "dolores", "elena", "elisa", "elvira",
    "emilia", "encarna", "esther", "estrella", "eugenia", "eva", "fatima", "francisca", "gema",
    "gloria", "graciela", "guadalupe", "helena", "ines", "irene", "isabel", "juana", "julia",
    "laura", "leticia", "lidia", "lorena", "lourdes", "lucia", "luisa", "luz", "magdalena",
    "maite", "manuela", "margarita", "maria", "marina", "marisol", "marta", "matilde", "mayte",
    "mercedes", "milagros", "miriam", "monica", "montserrat", "natalia", "nerea", "nieves",
    "noelia", "noemi", "nuria", "olga", "paloma", "patricia", "paula", "paz", "pepa", "pilar",
    "purificacion", "raquel", "rebeca", "remedios", "rocio", "rosa", "rosario", "ruth",
    "salome", "sandra", "sara", "silvia", "soledad", "sonia", "susana", "tamara", "teresa",
    "trinidad", "vanesa", "vanessa", "veronica", "victoria", "virginia", "yolanda",
})

_NOMBRES_MASC_INEQUIVOCOS = frozenset({
    "alberto", "alejandro", "alfonso", "alfredo", "alonso", "alvaro", "andres", "angel",
    "antonio", "arturo", "benjamin", "carlos", "cesar", "claudio", "constantino", "cristobal",
    "daniel", "david", "diego", "eduardo", "emilio", "enrique", "ernesto", "esteban",
    "eugenio", "federico", "felipe", "felix", "fernando", "francisco", "gabriel", "gerardo",
    "german", "gonzalo", "guillermo", "gustavo", "hector", "hugo", "ignacio", "iker", "isaac",
    "ismael", "ivan", "jacinto", "jacob", "jaime", "javier", "jesus", "joaquin", "jordi",
    "jorge", "jose", "juan", "julian", "julio", "leandro", "leonardo", "lorenzo", "lucas",
    "luis", "manolo", "manuel", "marc", "marcos", "mariano", "mario", "martin", "mateo",
    "matias", "mauricio", "miguel", "nicolas", "oscar", "pablo", "paco", "patricio", "pedro",
    "pepe", "rafael", "ramon", "raul", "ricardo", "roberto", "rodrigo", "ruben", "salvador",
    "samuel", "santiago", "sebastian", "sergio", "simon", "tomas", "vicente", "victor", "xavier",
})


def _primer_nombre_norm(nombre: str) -> str:
    s = unicodedata.normalize("NFKD", nombre or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    return s.split()[0] if s else ""


def hidratar_f0_catalogo_web(carpeta_datos: str | Path = RAIZ / "frontend" / "datos") -> dict[str, int]:
    """Hidrata `f0` aproximado por registro vocal en `actores.json` usando el eje LDA grave/agudo."""
    carpeta = Path(carpeta_datos)
    ruta_meta = carpeta / "actores.json"
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    actores = meta["actores"]
    dim = int(meta["dim"])
    nA = len(actores)
    C = normaliza(np.fromfile(carpeta / "centroides.bin", dtype="<f4").reshape(nA, dim).astype(np.float32))

    nombres_pila = [_primer_nombre_norm(a.get("nombre", "")) for a in actores]
    idx_fem = [i for i, nom in enumerate(nombres_pila) if nom in _NOMBRES_FEM_INEQUIVOCOS]
    idx_masc = [i for i, nom in enumerate(nombres_pila) if nom in _NOMBRES_MASC_INEQUIVOCOS]
    if not idx_fem or not idx_masc:
        return {"total": nA, "graves": 0, "agudos": 0, "conservados": nA, "sin_cambio": nA}

    w_reg = normaliza(C[idx_fem].mean(axis=0) - C[idx_masc].mean(axis=0))
    proy = C @ w_reg
    mu_z = float(proy.mean())
    sd_z = max(float(proy.std()), 1e-6)
    z = (proy - mu_z) / sd_z

    graves = agudos = conservados = sin_cambio = 0
    for i, a in enumerate(actores):
        if a.get("f0") is not None:
            conservados += 1
            continue
        nom = nombres_pila[i]
        if float(z[i]) <= -0.65 and nom not in _NOMBRES_FEM_INEQUIVOCOS:
            a["f0"] = 105.0
            graves += 1
        elif float(z[i]) >= 0.65 and nom not in _NOMBRES_MASC_INEQUIVOCOS:
            a["f0"] = 225.0
            agudos += 1
        else:
            sin_cambio += 1

    ruta_meta.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"total": nA, "graves": graves, "agudos": agudos, "conservados": conservados, "sin_cambio": sin_cambio}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría de calidad del índice de voces de EstaVoz")
    parser.add_argument("--datos-web", type=Path, default=RAIZ / "frontend" / "datos")
    parser.add_argument("--umbral-outlier", type=float, default=0.10)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--hidratar-f0", action="store_true", help="Hidrata `f0` por registro vocal en actores.json")
    args = parser.parse_args(argv)

    if args.hidratar_f0:
        res_f0 = hidratar_f0_catalogo_web(args.datos_web)
        print(
            f"F0 hidratado ({res_f0['total']} actores): graves={res_f0['graves']} | "
            f"agudos={res_f0['agudos']} | conservados={res_f0['conservados']} | medios/neutros={res_f0['sin_cambio']}"
        )

    informe = auditar_datos_web(args.datos_web, umbral_outlier=args.umbral_outlier)
    res = informe["resumen"]
    print(f"=== AUDITORÍA DE ÍNDICE ({res['num_actores']} actores, {res['num_trozos_totales']} trozos) ===")
    print(f"  Trozos por actor -> media: {res['media_trozos_por_actor']} | mediana: {res['mediana_trozos_por_actor']}")
    print(f"  Trozos intrusos (sim < {res['umbral_outlier']:.2f}): {res['num_trozos_intrusos_detectados']} ({res['pct_trozos_intrusos']}%)")
    print(f"  Actores con <= 4 trozos: {res['num_actores_con_4_o_menos_trozos']}")
    print("  Top 5 parejas de gemelos vocales:")
    for p in informe["gemelos_vocales_top"][:5]:
        print(f"    - {p['nombre_1']} <-> {p['nombre_2']}: sim = {p['similitud_centroides']:.4f}")
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Informe guardado en: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
