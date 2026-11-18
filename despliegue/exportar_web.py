"""Pasa el índice (data/indice.npz + data/fichas.json) al formato que lee la web.

Genera en frontend/:
  datos/lda.bin         media (256) + matriz W (256 x 256) de la LDA, float32
  datos/centroides.bin  huella media de cada actor ya proyectada, float32 (actores x 256)
  datos/trozos.bin      todas las huellas proyectadas, por actor: inicio de cada actor (uint32, actores + 1),
                        escala de cada huella (float32) y la huella en int8. La web mezcla el parecido con
                        la huella media y con los 3 trozos más parecidos de cada actor (+3 puntos de acierto)
  datos/actores.json    fichas, en el mismo orden que los centroides
  datos/busqueda.json   obras, personajes y actores originales con su reparto ("¿Qué estás viendo?")
  datos/papeles/<n>.json  trabajos completos de cada doblador
  fotos/<n>.jpg
La web no enlaza ni carga nada de fuera.
Si existe data/respuestas.json (backend/aprender.py), añade las respuestas "¿Acertamos?" que pasen los filtros.
El índice tiene que estar hecho con ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx (VoxBlink2 ResNet100, v3).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
from auditar_indice import filtrar_trozos_intrusos, hidratar_f0_catalogo_web  # noqa: E402
from filmografia import (  # noqa: E402
    construir_entradas_busqueda,
    exportar_indices_busqueda,
    resumen_actor_desde_papeles,
)
from lda import entrenar_lda, estimar_covarianza_canal, proyectar  # noqa: E402
from metadatos_modelo import (  # noqa: E402
    VERSION_LDA_ACTUAL,
    info_por_version,
    inspeccionar_onnx,
    validar_artefactos_web,
    validar_compatibilidad_indice,
)
from rechazo import UMBRALES_VOXBLINK_R100_CALIBRADOS  # noqa: E402


def entradas_busqueda(slugs):
    """Sugerencias de "¿Qué estás viendo?": obras, personajes y actores originales con su reparto."""
    filas_por_actor = []
    for s in slugs:
        f = RAIZ / "data" / "actores" / s / "trabajos.json"
        filas_por_actor.append(json.loads(f.read_text(encoding="utf-8")) if f.exists() else [])
    return construir_entradas_busqueda(filas_por_actor)


# versión del modelo de voz que usa la web; las respuestas guardan la suya y solo se aprovechan
# las del modelo en uso (1 = ECAPA, 2 = VoxBlink2 ResNet34, 3 = VoxBlink2 ResNet100)
VERSION_MODELO = "3"


def respuestas_validas(mu, W, C, slugs, ruta_respuestas=None):
    """Huellas de las respuestas de los usuarios que se pueden usar para aprender.

    Filtros contra respuestas falsas, despistes o envenenamiento:
      - solo respuestas con un actor válido y de la versión actual del modelo;
      - vectores finitos con norma no degenerada y sin duplicados exactos;
      - confirmación (`acierto=True`): similitud >= 0.28 y actor en Top-3;
      - corrección (`acierto=False`): similitud >= 0.32, actor en Top-3 y consenso >= 2 si < 0.42.
    """
    f = Path(ruta_respuestas) if ruta_respuestas is not None else (RAIZ / "data" / "respuestas.json")
    if not f.exists():
        return [], []
    pos = {s: i for i, s in enumerate(slugs)}
    hashes_vistos: set[str] = set()
    confirmadas: list[tuple[str, np.ndarray]] = []
    pendientes_correccion: dict[str, list[np.ndarray]] = {}
    for r in json.loads(f.read_text(encoding="utf-8")):
        if str(r.get("version")) != VERSION_MODELO:
            continue
        es_acierto = bool(r.get("acierto"))
        actor = r.get("propuesto") if es_acierto else r.get("real")
        if actor not in pos:
            continue
        h = np.asarray(r.get("huellas", []), dtype=np.float32)
        if h.ndim != 2 or len(h) == 0 or h.shape[1] != C.shape[1] or not np.all(np.isfinite(h)):
            continue
        normas = np.linalg.norm(h, axis=1, keepdims=True)
        if np.any(normas < 1e-5):
            continue
        clave_hash = hashlib.sha1(np.round(h, 4).tobytes()).hexdigest()
        if clave_hash in hashes_vistos:
            continue
        h = h / normas
        punt = (proyectar(h, mu, W) @ C.T).mean(0)
        idx_a = pos[actor]
        en_top3 = idx_a in np.argsort(-punt)[:3]
        sim_a = float(punt[idx_a])
        if es_acierto:
            if sim_a >= 0.28 and en_top3:
                hashes_vistos.add(clave_hash)
                confirmadas.append((actor, h))
        elif sim_a >= 0.32 and en_top3:
            hashes_vistos.add(clave_hash)
            if sim_a >= 0.42:
                confirmadas.append((actor, h))
            else:
                pendientes_correccion.setdefault(actor, []).append(h)

    for actor, lista_h in pendientes_correccion.items():
        if len(lista_h) >= 2:
            for h in lista_h:
                confirmadas.append((actor, h))

    X = [h for _, h in confirmadas]
    y = [a for a, _ in confirmadas for _ in range(len(_))]
    return X, y


def centroides_mezcla(Xp, y, clips, slugs, peso_entrevistas=0.5, peso_respuestas=0.35):
    """Huella media de cada actor, con la voz de entrevistas (`voz_`) a medio peso (0.5)
    y las respuestas voluntarias de usuarios (`respuesta`) a peso moderado (0.35) para no desplazar
    las muestras verificadas de estudio."""
    clips_str = np.asarray(clips, dtype=str)
    entrevista = np.array(["/voz_" in c for c in clips_str], dtype=bool)
    respuesta = clips_str == "respuesta"
    C = np.zeros((len(slugs), Xp.shape[1]))
    for i, a in enumerate(slugs):
        es_actor = (y == a)
        propia = (~entrevista) & (~respuesta) & es_actor
        m_ent = entrevista & es_actor
        m_resp = respuesta & es_actor
        if propia.any():
            v = Xp[propia].mean(0)
            C[i] = v / np.linalg.norm(v)
        if m_ent.any():   # los que solo tienen voz de entrevistas (sin muestra en eldoblaje) se quedan con esa
            v = Xp[m_ent].mean(0)
            C[i] += (peso_entrevistas if propia.any() else 1.0) * v / np.linalg.norm(v)
        if m_resp.any():
            v = Xp[m_resp].mean(0)
            tiene_base = propia.any() or m_ent.any()
            C[i] += (peso_respuestas if tiene_base else 1.0) * v / np.linalg.norm(v)
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def main():
    datos = np.load(RAIZ / "data" / "indice.npz")
    ruta_onnx = RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"
    info_onnx = inspeccionar_onnx(ruta_onnx) if ruta_onnx.exists() else info_por_version(VERSION_MODELO)
    validar_compatibilidad_indice(datos, info_modelo=info_onnx, version_esperada=VERSION_MODELO)
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]
    fichas = json.loads((RAIZ / "data" / "fichas.json").read_text(encoding="utf-8"))
    slugs = sorted(fichas)

    # la LDA aprende de los actores con 2+ clips (misma voz en grabaciones distintas)
    # e incorpora regularización acústica de canal a partir de pares limpio / #movil
    base = np.array([c.split("#")[0] for c in clips])  # "x.mp3#movil" es el mismo clip que "x.mp3"
    n_clips = {a: len(set(base[y == a])) for a in slugs}
    actores_lda = {a for a in slugs if n_clips[a] >= 2}
    sel = np.isin(y, list(actores_lda))
    pares_limp, pares_mov = [], []
    por_clip_id: dict[str, list[int]] = {}
    for idx_v, c in enumerate(clips):
        por_clip_id.setdefault(str(c), []).append(idx_v)
    for c_str, idxs_mov in por_clip_id.items():
        if c_str.endswith("#movil"):
            base_c = c_str[:-6]
            idxs_limp = por_clip_id.get(base_c)
            if idxs_limp and str(y[idxs_mov[0]]) in actores_lda:
                k_par = min(len(idxs_limp), len(idxs_mov))
                pares_limp.append(X[idxs_limp[:k_par]])
                pares_mov.append(X[idxs_mov[:k_par]])
    cov_ac = estimar_covarianza_canal(np.vstack(pares_limp), np.vstack(pares_mov)) if pares_limp else None
    mu, W = entrenar_lda(X[sel], y[sel], cov_acustica=cov_ac, peso_cov_acustica=0.10)
    C = centroides_mezcla(proyectar(X, mu, W), y, clips, slugs)
    Xt, yt, clips_t = X, y, np.asarray(clips, dtype=str)

    # Poda automática de trozos intrusos (interlocutores secundarios o música pura con sim < 0.20)
    conservar, intrusos = filtrar_trozos_intrusos(proyectar(Xt, mu, W), yt, C, slugs, umbral_outlier=0.20)
    if intrusos:
        Xt, yt, clips_t = Xt[conservar], yt[conservar], clips_t[conservar]
        C = centroides_mezcla(proyectar(Xt, mu, W), yt, clips_t, slugs)
        print(f"Trozos intrusos filtrados: {len(intrusos)} segmentos con sim < 0.20 al centroide del actor")

    # Las respuestas ciudadanas refuerzan solo los centroides (sin contaminar trozos.bin de 6 s)
    Xr, yr = respuestas_validas(mu, W, C, slugs)
    if Xr:
        Xr, yr = np.concatenate(Xr), np.array(yr)
        propios = {a: int((yt == a).sum()) for a in slugs}
        vistos, tope = {}, np.zeros(len(yr), dtype=bool)
        for i, a in enumerate(yr):
            vistos[a] = vistos.get(a, 0) + 1
            tope[i] = vistos[a] <= propios[a]
        Xr, yr = Xr[tope], yr[tope]
        C = centroides_mezcla(
            proyectar(np.concatenate([Xt, Xr]), mu, W),
            np.concatenate([yt, yr]),
            np.concatenate([clips_t, ["respuesta"] * len(yr)]),
            slugs,
            peso_respuestas=0.25,
        )
        print(f"Respuestas de usuarios usadas en centroides: {len(yr)} huellas de {len(set(yr))} actores")

    salida = RAIZ / "frontend" / "datos"
    shutil.rmtree(salida, ignore_errors=True)
    salida.mkdir(parents=True)
    np.concatenate([mu, W.ravel()]).astype("<f4").tofile(salida / "lda.bin")
    C.astype("<f4").tofile(salida / "centroides.bin")
    orden = np.argsort(np.searchsorted(slugs, yt), kind="stable")
    V = proyectar(Xt[orden], mu, W)
    escala = np.abs(V).max(1) / 127
    inicio = np.searchsorted(np.searchsorted(slugs, yt[orden]), np.arange(len(slugs) + 1))
    with open(salida / "trozos.bin", "wb") as f:
        f.write(inicio.astype("<u4").tobytes())
        f.write(escala.astype("<f4").tobytes())
        f.write(np.round(V / escala[:, None]).astype(np.int8).tobytes())

    entradas = entradas_busqueda(slugs)
    popularidad_obras = {e[1].lower(): len(e[2]) for e in entradas if e[0] == "obra"}
    exportar_indices_busqueda(entradas, salida)

    fotos = RAIZ / "frontend" / "fotos"
    shutil.rmtree(fotos, ignore_errors=True)
    fotos.mkdir()
    papeles = salida / "papeles"
    papeles.mkdir()
    actores = []
    for i, s in enumerate(slugs):
        foto = RAIZ / "data" / "actores" / s / "foto.jpg"
        if foto.exists():
            shutil.copy(foto, fotos / f"{i}.jpg")
        f_trab = RAIZ / "data" / "actores" / s / "trabajos.json"
        filas = json.loads(f_trab.read_text(encoding="utf-8")) if f_trab.exists() else []
        (papeles / f"{i}.json").write_text(
            json.dumps(
                [[t["obra"], t["personaje"], t["anio"], t["actor_original"]] for t in filas],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        resumen = resumen_actor_desde_papeles(filas, popularidad_obras=popularidad_obras) if filas else {
            "trabajos": fichas[s].get("trabajos", []),
            "num_trabajos": fichas[s].get("num_trabajos", 0),
            "resumen_tipos": {},
            "vozh": [],
        }
        info_actor = {
            "id": s,
            "nombre": fichas[s]["nombre"],
            "foto": f"fotos/{i}.jpg" if foto.exists() else "",
            "trabajos": resumen["trabajos"] or fichas[s].get("trabajos", []),
            "num_trabajos": resumen["num_trabajos"] or fichas[s].get("num_trabajos", 0),
        }
        if fichas[s].get("muestra"):
            info_actor["muestra"] = fichas[s]["muestra"]
        f0_val = fichas[s].get("f0")
        if f0_val is not None:
            try:
                f0_num = float(f0_val)
                if np.isfinite(f0_num) and 60.0 <= f0_num <= 380.0:
                    info_actor["f0"] = round(f0_num, 1)
            except (TypeError, ValueError):
                pass
        if resumen["resumen_tipos"]:
            info_actor["resumen_tipos"] = resumen["resumen_tipos"]
        if resumen["vozh"]:
            info_actor["vozh"] = resumen["vozh"]
        actores.append(info_actor)
    web = {
        "dim": int(X.shape[1]),
        "version_modelo": VERSION_MODELO,
        "arquitectura_modelo": info_onnx.arquitectura,
        "variante_modelo": info_onnx.variante,
        "version_lda": VERSION_LDA_ACTUAL,
        "umbrales_rechazo": UMBRALES_VOXBLINK_R100_CALIBRADOS.a_dict(),
        "actores": actores,
    }
    (salida / "actores.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    res_f0 = hidratar_f0_catalogo_web(salida)
    validar_artefactos_web(salida, ruta_onnx=ruta_onnx if ruta_onnx.exists() else None, version_esperada=VERSION_MODELO)
    print(f"F0 catálogo: graves={res_f0['graves']}, agudos={res_f0['agudos']}, conservados={res_f0['conservados']}")
    print(f"Búsqueda: {len(entradas)} obras/personajes/actores, {(salida / 'busqueda.json').stat().st_size >> 10} KB")
    print(f"{len(slugs)} actores (LDA entrenada con {sel.sum()} trozos de "
          f"{sum(v >= 2 for v in n_clips.values())} actores) -> {salida}")


if __name__ == "__main__":
    main()
