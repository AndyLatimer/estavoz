"""Sistema reproducible de evaluación técnica para EstaVoz (Fase 2 y Fase 3).

Reproduce fielmente el pipeline de producción del navegador (`frontend/index.html` + `despliegue/exportar_web.py`):
  - Proyección LDA entrenada sin fugas de información (solo sobre actores de referencia permitidos).
  - Centroides ponderados (`centroides_mezcla`: clips `/voz_` a peso 0.5).
  - Cuantización `int8` con escala por vector de los trozos proyectados (`trozos.bin`).
  - Puntuación híbrida: 50 % similitud al centroide + 50 % media de los 3 mejores trozos.

Capacidades:
  - Métricas Top-1 y Top-3 globales y desglosadas por actor, grabación, fuente y duración.
  - Separación estricta entre muestras de referencia (galería + LDA) y muestras de prueba, con agrupación
    por grabación de origen (`clip` y `clip#movil` juntos) y modo `solo_galeria_independiente` que evita
    la contaminación circular de clips de entrevista/podcast seleccionados usando el clip evaluado.
  - Simulación de voces ausentes de la base de datos (open-set) reteniendo actores fuera de la galería.
  - Evaluación de falsos positivos, rechazos correctos y calibración de porcentajes (`backend/rechazo.py`).
  - Comparación con y sin búsqueda contextual ("¿Qué estás viendo?").
  - Soporte para grabaciones reales etiquetadas con permiso verificado (`permiso_verificado=True`).
  - Exportación de informes en JSON y CSV, comparación entre versiones de modelo e historial de regresiones.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lda import entrenar_lda, estimar_covarianza_canal, normaliza, proyectar  # noqa: E402
from metadatos_modelo import InfoModelo, extraer_info_indice  # noqa: E402
from rechazo import (  # noqa: E402
    CalibradorConfianza,
    EstadoIdentificacion,
    MuestraValidacionRechazo,
    UmbralesRechazo,
    calibrar_umbrales,
    clasificar_consulta,
    evaluar_calibracion_porcentajes,
    evaluar_rechazo,
)


def deducir_fuente_clip(clip_id: str) -> str:
    """Identifica la fuente de un clip a partir de su identificador (`actor/archivo#sufijo`)."""
    base = clip_id.split("#")[0]
    nombre = base.split("/")[-1].lower()
    if nombre.startswith("voz_"):
        return "voces"
    if nombre.startswith("podcast_"):
        return "podcast"
    if nombre.startswith("entrevista_"):
        return "entrevista"
    if nombre.startswith("real_"):
        return "real"
    return "eldoblaje"


def id_grabacion_origen(clip_id: str) -> str:
    """Agrupa variantes de una misma grabación original para evitar fugas entre referencia y prueba.

    - `actor/Muestras.mp3` y `actor/Muestras.mp3#movil` -> `actor/Muestras.mp3`
    - `actor/voz_abc123_0.flac` y `actor/voz_abc123_1.flac` -> `actor/voz_abc123`
    """
    base = clip_id.split("#")[0]
    actor, _, archivo = base.partition("/")
    stem = Path(archivo).stem
    # Si un episodio produjo varios cortes con sufijo numérico (voz_<hash>_0, entrevista_0...)
    m = re.match(r"^(voz_[0-9a-f]{8,16})_\d+$", stem)
    if m:
        return f"{actor}/{m.group(1)}"
    return f"{actor}/{stem}"


def cuantizar_descuantizar_int8(V: np.ndarray) -> np.ndarray:
    """Reproduce exactamente la cuantización int8 de `trozos.bin` en `exportar_web.py` y `frontend/index.html`."""
    if len(V) == 0:
        return V.astype(np.float32)
    escala = (np.abs(V).max(axis=1) / 127.0).astype(np.float32)
    escala = np.where(escala > 0, escala, 1.0).astype(np.float32)
    q8 = np.round(V / escala[:, None]).clip(-127, 127).astype(np.int8)
    return (q8.astype(np.float32) * escala[:, None]).astype(np.float32)


def centroides_mezcla_web(
    Xp: np.ndarray,
    y: np.ndarray,
    clips: np.ndarray,
    slugs: list[str],
    peso_entrevistas: float = 0.5,
) -> np.ndarray:
    """Reproduce `centroides_mezcla` de `despliegue/exportar_web.py` (líneas 127-141)."""
    entrevista = np.array(["/voz_" in str(c) for c in clips], dtype=bool)
    C = np.zeros((len(slugs), Xp.shape[1]), dtype=np.float32)
    for i, a in enumerate(slugs):
        propia = (~entrevista) & (y == a)
        m = entrevista & (y == a)
        if propia.any():
            v = Xp[propia].mean(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                C[i] = v / n
        if m.any():
            v = Xp[m].mean(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                C[i] += (peso_entrevistas if propia.any() else 1.0) * (v / n)
    normas = np.linalg.norm(C, axis=1, keepdims=True)
    return np.where(normas > 0, C / np.maximum(normas, 1e-12), 0.0).astype(np.float32)


def puntuar_navegador(
    Q_proy: np.ndarray,
    C_galeria: np.ndarray,
    V_galeria: np.ndarray,
    actor_idx_galeria: np.ndarray,
    indices_candidatos: np.ndarray | None = None,
    peso_centroide: float = 0.5,
    top_k_trozos: int = 3,
    beta_exceso_trozos: float = 0.0,
) -> np.ndarray:
    """Calcula la puntuación exacta de `puntuar()` en `frontend/index.html` (líneas 330-343).

    Para cada candidato `i`:
      - `media`: similitud media de los vectores de la consulta `Q_proy` con el centroide `C_galeria[i]`.
      - `mejores`: media de los `top_k_trozos` trozos más parecidos del actor `i` en `V_galeria`.
      - `parecido = peso_centroide * media + (1 - peso_centroide) * (mejores - penal)`.
    Devuelve un vector de longitud `len(C_galeria)` (con `-np.inf` fuera de `indices_candidatos`).
    """
    n_actores = len(C_galeria)
    if len(Q_proy) == 0 or n_actores == 0:
        return np.full(n_actores, -np.inf, dtype=np.float32)

    cand = np.arange(n_actores) if indices_candidatos is None else np.asarray(indices_candidatos, dtype=int)
    media_cent = (Q_proy @ C_galeria.T).mean(axis=0)
    sims_trozos = (Q_proy @ V_galeria.T).mean(axis=0) if len(V_galeria) else np.zeros(0, dtype=np.float32)

    salida = np.full(n_actores, -np.inf, dtype=np.float32)
    if len(V_galeria) > 0:
        orden = np.lexsort((-sims_trozos, actor_idx_galeria))
        fa_ord = actor_idx_galeria[orden]
        sv_ord = sims_trozos[orden]
        ini = np.searchsorted(fa_ord, np.arange(n_actores))
        fin = np.searchsorted(fa_ord, np.arange(n_actores), side="right")
    else:
        ini = fin = np.zeros(n_actores, dtype=int)
        sv_ord = np.zeros(0, dtype=np.float32)

    for i in cand:
        m = float(media_cent[i])
        a, b = int(ini[i]), int(fin[i])
        if b > a:
            n_disp = b - a
            mejores = float(sv_ord[a : min(b, a + top_k_trozos)].mean())
            penal = float(beta_exceso_trozos * max(0.0, np.log(max(n_disp, 1) / 12.0))) if beta_exceso_trozos > 0 else 0.0
        else:
            mejores = m
            penal = 0.0
        salida[i] = peso_centroide * m + (1.0 - peso_centroide) * (mejores - penal)
    return salida


@dataclass(frozen=True)
class MuestraReferencia:
    actor: str
    clip_id: str
    vectores: np.ndarray        # (n_trozos, dim)
    fuente: str = ""
    grupo_grabacion: str = ""

    def __post_init__(self) -> None:
        if not self.fuente:
            object.__setattr__(self, "fuente", deducir_fuente_clip(self.clip_id))
        if not self.grupo_grabacion:
            object.__setattr__(self, "grupo_grabacion", id_grabacion_origen(self.clip_id))


@dataclass(frozen=True)
class ConsultaEvaluacion:
    consulta_id: str
    actor_real: str | None      # None si es una grabación de voz anónima/no registrada
    clip_id: str
    vectores: np.ndarray        # (n_trozos_consulta, dim)
    duracion_seg: float = 10.0
    fuente: str = ""
    grupo_grabacion: str = ""
    candidatos_contexto: list[str] | None = None
    es_grabacion_real: bool = False
    permiso_verificado: bool = True

    def __post_init__(self) -> None:
        if not self.fuente:
            object.__setattr__(self, "fuente", deducir_fuente_clip(self.clip_id))
        if not self.grupo_grabacion:
            object.__setattr__(self, "grupo_grabacion", id_grabacion_origen(self.clip_id))


@dataclass
class ConfigEvaluacion:
    version_modelo: str = "3"
    nombre_experimento: str = "evaluacion_estandar"
    pliegues_lda: int = 5
    entrenar_lda_en_pliegues: bool = True
    semilla: int = 0
    reg_lda: float = 1e-2
    dims_lda: int | None = None
    peso_entrevistas: float = 0.5
    peso_centroide: float = 0.5
    cuantizar_trozos_int8: bool = True
    usar_cov_acustica: bool = False
    peso_cov_acustica: float = 0.10
    # Si es True, al evaluar un clip de eldoblaje excluye también de la galería del actor los clips
    # extraídos de podcasts/entrevistas/voces cuya selección dependió del índice previo.
    solo_galeria_independiente: bool = False
    # Proporción de actores retenidos fuera de la galería para simular voces desconocidas (open-set)
    proporcion_actores_desconocidos: float = 0.15
    # Tamaño medio del reparto al simular búsqueda contextual ("¿Qué estás viendo?")
    tamano_contexto_simulado: int = 12
    max_fpr_desconocidos: float = 0.05
    min_precision_conocidos: float = 0.90


def _calcular_top_metricas(filas: list[dict[str, Any]], prefijo: str = "") -> dict[str, Any]:
    """Calcula Top-1, Top-3 y similitud media para una lista de resultados de consultas conocidas."""
    conocidas = [r for r in filas if r["es_actor_conocido"]]
    n = len(conocidas)
    if n == 0:
        return {
            f"{prefijo}n_consultas": 0,
            f"{prefijo}top1_pct": 0.0,
            f"{prefijo}top3_pct": 0.0,
            f"{prefijo}similitud_media_top1": 0.0,
            f"{prefijo}margen_medio_12": 0.0,
        }
    t1 = sum(1 for r in conocidas if r["acierto_top1"])
    t3 = sum(1 for r in conocidas if r["acierto_top3"])
    s1 = float(np.mean([r["s1"] for r in conocidas]))
    m12 = float(np.mean([r["margen_12"] for r in conocidas]))
    return {
        f"{prefijo}n_consultas": n,
        f"{prefijo}top1_pct": round(100.0 * t1 / n, 2),
        f"{prefijo}top3_pct": round(100.0 * t3 / n, 2),
        f"{prefijo}similitud_media_top1": round(s1, 4),
        f"{prefijo}margen_medio_12": round(m12, 4),
    }


def cargar_grabaciones_reales(manifiesto_json: str | Path) -> list[ConsultaEvaluacion]:
    """Carga grabaciones reales etiquetadas desde un manifiesto JSON verificando permisos explícitos.

    Formato esperado del manifiesto JSON:
    [
      {
        "id": "toma_001",
        "actor_real": "108-jose-luis-gil",   // o null si es voz desconocida
        "clip_id": "108-jose-luis-gil/real_toma_001.wav",
        "duracion_seg": 10.0,
        "permiso_verificado": true,
        "candidatos_contexto": ["108-jose-luis-gil", "205-claudio-serrano"], // opcional
        "huellas": [[...], ...]               // vectores ya extraídos (o ruta_npz)
      }
    ]
    """
    ruta = Path(manifiesto_json)
    entradas = json.loads(ruta.read_text(encoding="utf-8"))
    consultas: list[ConsultaEvaluacion] = []
    for item in entradas:
        if not item.get("permiso_verificado", False):
            raise PermissionError(
                f"La grabación real {item.get('id')!r} no declara `permiso_verificado: true`. "
                "No se admiten grabaciones de terceros sin derechos o permisos verificados."
            )
        if "huellas" in item:
            vecs = np.asarray(item["huellas"], dtype=np.float32)
        elif "ruta_npz" in item:
            d = np.load(ruta.parent / item["ruta_npz"])
            vecs = d["vectores"].astype(np.float32)
        else:
            raise ValueError(f"La entrada {item.get('id')!r} debe incluir `huellas` o `ruta_npz`.")
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        vecs = normaliza(vecs)
        consultas.append(
            ConsultaEvaluacion(
                consulta_id=str(item["id"]),
                actor_real=item.get("actor_real"),
                clip_id=str(item.get("clip_id", f"{item.get('actor_real') or 'anonimo'}/real_{item['id']}.wav")),
                vectores=vecs,
                duracion_seg=float(item.get("duracion_seg", 10.0)),
                fuente="real",
                candidatos_contexto=item.get("candidatos_contexto"),
                es_grabacion_real=True,
                permiso_verificado=True,
            )
        )
    return consultas


def ejecutar_evaluacion_completa(
    referencias: list[MuestraReferencia],
    consultas: list[ConsultaEvaluacion],
    config: ConfigEvaluacion | None = None,
    repartos_contexto: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Ejecuta la evaluación completa separando entrenamiento LDA, galería, conocidos y desconocidos.

    Garantías contra fugas de información (anti-leakage):
      1. Los actores se dividen en `pliegues_lda` pliegues disjuntos: para cada consulta del pliegue `p`,
         la transformación LDA `(mu, W)` se entrena exclusivamente con actores de otros pliegues (`!= p`)
         que tampoco pertenezcan al conjunto de actores desconocidos retenidos.
      2. Al evaluar una consulta `q`, se excluyen de la galería todos los trozos cuyo `grupo_grabacion`
         coincida con `q.grupo_grabacion` (así se apartan tanto el clip limpio como su versión `#movil`
         y cortes hermanos del mismo episodio).
      3. Si `config.solo_galeria_independiente=True`, al evaluar un clip de `eldoblaje` también se
         excluyen de la galería del actor los clips de `podcast`, `entrevista` y `voces` (cuya extracción
         dependió del índice original).
      4. Para medir voces desconocidas (open-set), un subconjunto determinista de actores se retira por
         completo de la galería y del entrenamiento LDA; sus consultas se evalúan contra la galería
         restante para medir falsos positivos y rechazos correctos.
    """
    cfg = config or ConfigEvaluacion()
    rng = np.random.default_rng(cfg.semilla)

    for q in consultas:
        if q.es_grabacion_real and not q.permiso_verificado:
            raise PermissionError(f"Grabación real sin permiso verificado: {q.consulta_id}")

    actores_todos = sorted({r.actor for r in referencias})
    if len(actores_todos) < 2:
        raise ValueError("Se necesitan al menos 2 actores en `referencias` para evaluar.")

    # Selección determinista de actores retenidos como "voces desconocidas" (fuera de la base)
    n_desc = max(1, int(round(len(actores_todos) * cfg.proporcion_actores_desconocidos))) if cfg.proporcion_actores_desconocidos > 0 else 0
    n_desc = min(n_desc, max(0, len(actores_todos) - 2))
    actores_desconocidos = set(rng.choice(actores_todos, size=n_desc, replace=False).tolist()) if n_desc > 0 else set()
    actores_galeria = [a for a in actores_todos if a not in actores_desconocidos]
    idx_galeria = {a: i for i, a in enumerate(actores_galeria)}

    # Asignación de pliegues por actor para entrenar la LDA sin ver los actores evaluados
    n_pliegues = max(2, min(cfg.pliegues_lda, len(actores_galeria)))
    pliegue_actor = {a: i % n_pliegues for i, a in enumerate(rng.permutation(actores_galeria))}
    for a in actores_desconocidos:
        pliegue_actor[a] = int(rng.integers(0, n_pliegues))

    # Pre-agrupar referencias de actores de la galería
    refs_galeria = [r for r in referencias if r.actor in idx_galeria and len(r.vectores) > 0]
    grupos_por_actor: dict[str, set[str]] = defaultdict(set)
    for r in refs_galeria:
        grupos_por_actor[r.actor].add(r.grupo_grabacion)

    # Entrenar una LDA por pliegue (o usar identidad si los vectores ya están en el subespacio LDA publicado)
    dim_vec = refs_galeria[0].vectores.shape[1]
    modelos_lda: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if cfg.entrenar_lda_en_pliegues:
        for p in range(n_pliegues):
            actores_entreno = {
                a for a in actores_galeria if pliegue_actor[a] != p and len(grupos_por_actor[a]) >= 2
            }
            if len(actores_entreno) < 2:
                actores_entreno = {a for a in actores_galeria if pliegue_actor[a] != p}
            X_lda = np.concatenate([r.vectores for r in refs_galeria if r.actor in actores_entreno])
            y_lda = np.concatenate([[r.actor] * len(r.vectores) for r in refs_galeria if r.actor in actores_entreno])
            cov_ac = None
            if cfg.usar_cov_acustica:
                pares_limp, pares_mov = [], []
                mapa_refs = {(r.actor, r.clip_id): r.vectores for r in refs_galeria if r.actor in actores_entreno}
                for r in refs_galeria:
                    if r.actor in actores_entreno and r.clip_id.endswith("#movil"):
                        base_id = r.clip_id[:-6]
                        if (r.actor, base_id) in mapa_refs:
                            vb = mapa_refs[(r.actor, base_id)]
                            vm = r.vectores
                            L = min(len(vb), len(vm))
                            if L > 0:
                                pares_limp.append(vb[:L])
                                pares_mov.append(vm[:L])
                if pares_limp:
                    cov_ac = estimar_covarianza_canal(np.vstack(pares_limp), np.vstack(pares_mov))
            modelos_lda[p] = entrenar_lda(
                X_lda, y_lda, dims=cfg.dims_lda, reg=cfg.reg_lda, cov_acustica=cov_ac, peso_cov_acustica=cfg.peso_cov_acustica
            )
    else:
        mu_id = np.zeros(dim_vec, dtype=np.float32)
        W_id = np.eye(dim_vec, dtype=np.float32)
        for p in range(n_pliegues):
            modelos_lda[p] = (mu_id, W_id)

    # Preparar vectores planos de galería ordenados por actor para indexado matricial ultrarrápido
    refs_galeria.sort(key=lambda r: idx_galeria[r.actor])
    X_ref = np.concatenate([r.vectores for r in refs_galeria])
    y_ref = np.concatenate([[r.actor] * len(r.vectores) for r in refs_galeria])
    fa_ref = np.array([idx_galeria[a] for a in y_ref], dtype=int)
    clip_ref = np.concatenate([[r.clip_id] * len(r.vectores) for r in refs_galeria])
    grupo_ref = np.concatenate([[r.grupo_grabacion] * len(r.vectores) for r in refs_galeria])
    fuente_ref = np.concatenate([[r.fuente] * len(r.vectores) for r in refs_galeria])

    n_gal = len(actores_galeria)
    ini_actor = np.searchsorted(fa_ref, np.arange(n_gal))
    fin_actor = np.searchsorted(fa_ref, np.arange(n_gal), side="right")
    max_t = int(np.max(fin_actor - ini_actor)) if n_gal > 0 else 1
    k_top = min(3, max_t)
    # Matriz (n_gal, max_t) con los índices de trozos de cada actor (el resto apunta al centinela -inf)
    matriz_idx_actor = np.full((n_gal, max_t), len(X_ref), dtype=np.int32)
    for i in range(n_gal):
        a_i, b_i = int(ini_actor[i]), int(fin_actor[i])
        if b_i > a_i:
            matriz_idx_actor[i, : b_i - a_i] = np.arange(a_i, b_i, dtype=np.int32)

    filas_por_grupo_lista: dict[str, list[int]] = defaultdict(list)
    for idx_f, g in enumerate(grupo_ref.tolist()):
        filas_por_grupo_lista[str(g)].append(idx_f)
    filas_por_grupo: dict[str, np.ndarray] = {
        g: np.array(idxs, dtype=np.int32) for g, idxs in filas_por_grupo_lista.items()
    }

    filas_contaminadas_actor: dict[str, np.ndarray] = {}
    if cfg.solo_galeria_independiente:
        es_cont = np.isin(fuente_ref, ["podcast", "entrevista", "voces"])
        for a in actores_galeria:
            i_a = idx_galeria[a]
            rango = np.arange(int(ini_actor[i_a]), int(fin_actor[i_a]))
            filas_contaminadas_actor[a] = rango[es_cont[rango]]

    # Pre-proyectar la galería y los centroides base por cada pliegue (para actualizar solo el actor consultado)
    galeria_por_pliegue: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p, (mu, W) in modelos_lda.items():
        V_proy = proyectar(X_ref, mu, W)
        V_quant = cuantizar_descuantizar_int8(V_proy) if cfg.cuantizar_trozos_int8 else V_proy
        C_base = centroides_mezcla_web(
            V_proy,
            y_ref,
            clip_ref,
            actores_galeria,
            peso_entrevistas=cfg.peso_entrevistas,
        )
        galeria_por_pliegue[p] = (V_proy, V_quant, C_base)

    resultados_detalle: list[dict[str, Any]] = []
    muestras_rechazo_abierto: list[MuestraValidacionRechazo] = []
    muestras_rechazo_contexto: list[MuestraValidacionRechazo] = []

    for q in consultas:
        if len(q.vectores) == 0:
            continue
        es_conocido = (q.actor_real is not None) and (q.actor_real in idx_galeria)
        p_fold = pliegue_actor.get(q.actor_real or "", 0)
        mu, W = modelos_lda[p_fold]
        V_proy_all, V_quant_all, C_base_fold = galeria_por_pliegue[p_fold]

        excl = filas_por_grupo.get(q.grupo_grabacion, np.zeros(0, dtype=int))
        if cfg.solo_galeria_independiente and q.fuente == "eldoblaje" and q.actor_real in filas_contaminadas_actor:
            excl = np.union1d(excl, filas_contaminadas_actor[q.actor_real])

        C_fold = C_base_fold
        if es_conocido and q.actor_real is not None:
            i_real = idx_galeria[q.actor_real]
            rango_actor = np.arange(int(ini_actor[i_real]), int(fin_actor[i_real]))
            quedan = np.setdiff1d(rango_actor, excl, assume_unique=True)
            if len(quedan) == 0:
                continue
            C_fold = C_base_fold.copy()
            C_fold[i_real] = centroides_mezcla_web(
                V_proy_all[quedan],
                y_ref[quedan],
                clip_ref[quedan],
                [q.actor_real],
                peso_entrevistas=cfg.peso_entrevistas,
            )[0]

        Q_proy = proyectar(q.vectores, mu, W)
        media_cent = (Q_proy @ C_fold.T).mean(axis=0)
        sims_all = np.empty(len(V_quant_all) + 1, dtype=np.float32)
        sims_all[:-1] = (Q_proy @ V_quant_all.T).mean(axis=0)
        sims_all[-1] = -np.inf
        if len(excl):
            sims_all[excl] = -np.inf

        sims_actor = sims_all[matriz_idx_actor]
        top_k_vals = -np.partition(-sims_actor, kth=k_top - 1, axis=1)[:, :k_top]
        finitos = np.isfinite(top_k_vals)
        cuenta_f = finitos.sum(axis=1)
        suma_f = np.where(finitos, top_k_vals, 0.0).sum(axis=1)
        mejores = np.where(cuenta_f > 0, suma_f / np.maximum(cuenta_f, 1), media_cent)
        punt_abierto = (cfg.peso_centroide * media_cent + (1.0 - cfg.peso_centroide) * mejores).astype(np.float32)

        orden_ab = np.argsort(-punt_abierto)
        top1_ab = actores_galeria[int(orden_ab[0])]
        top3_ab = [actores_galeria[int(i)] for i in orden_ab[:3]]
        s1_ab = float(punt_abierto[orden_ab[0]])
        s2_ab = float(punt_abierto[orden_ab[1]]) if len(orden_ab) >= 2 else float("nan")

        # Búsqueda contextual ("¿Qué estás viendo?")
        if q.candidatos_contexto is not None:
            cands_ctx_slugs = [c for c in q.candidatos_contexto if c in idx_galeria]
        elif repartos_contexto and q.actor_real in repartos_contexto:
            cands_ctx_slugs = [c for c in repartos_contexto[q.actor_real] if c in idx_galeria]
        else:
            # Simulación reproducible de reparto de una película/serie (incluye al actor real si es conocido)
            k_ctx = min(cfg.tamano_contexto_simulado, len(actores_galeria))
            sem_q = abs(hash((q.consulta_id, cfg.semilla))) % (2**31 - 1)
            rng_q = np.random.default_rng(sem_q)
            otros = [a for a in actores_galeria if a != q.actor_real]
            elegidos = rng_q.choice(otros, size=max(0, k_ctx - (1 if es_conocido else 0)), replace=False).tolist()
            if es_conocido and q.actor_real is not None:
                elegidos.append(q.actor_real)
            cands_ctx_slugs = sorted(elegidos)

        idx_ctx = np.array([idx_galeria[c] for c in cands_ctx_slugs], dtype=int)
        punt_ctx = punt_abierto[idx_ctx]
        orden_ctx = np.argsort(-punt_ctx)
        top1_ctx = cands_ctx_slugs[int(orden_ctx[0])] if len(orden_ctx) else None
        top3_ctx = [cands_ctx_slugs[int(i)] for i in orden_ctx[:3]]
        s1_ctx = float(punt_ctx[orden_ctx[0]]) if len(orden_ctx) else -1.0
        s2_ctx = float(punt_ctx[orden_ctx[1]]) if len(orden_ctx) >= 2 else float("nan")

        muestras_rechazo_abierto.append(
            MuestraValidacionRechazo(
                puntuaciones=punt_abierto.copy(),
                ids_candidatos=actores_galeria,
                actor_real=q.actor_real if es_conocido else None,
                es_conocido=es_conocido,
                n_galeria_total=len(actores_galeria),
                con_contexto=False,
            )
        )
        if len(cands_ctx_slugs) > 0:
            muestras_rechazo_contexto.append(
                MuestraValidacionRechazo(
                    puntuaciones=punt_ctx.copy(),
                    ids_candidatos=cands_ctx_slugs,
                    actor_real=q.actor_real if (es_conocido and q.actor_real in cands_ctx_slugs) else None,
                    es_conocido=bool(es_conocido and q.actor_real in cands_ctx_slugs),
                    n_galeria_total=len(actores_galeria),
                    con_contexto=True,
                )
            )

        resultados_detalle.append(
            {
                "consulta_id": q.consulta_id,
                "actor_real": q.actor_real,
                "es_actor_conocido": es_conocido,
                "clip_id": q.clip_id,
                "grupo_grabacion": q.grupo_grabacion,
                "fuente": q.fuente,
                "duracion_seg": float(q.duracion_seg),
                "es_grabacion_real": q.es_grabacion_real,
                "top1_predicho": top1_ab,
                "top3_predichos": top3_ab,
                "acierto_top1": bool(es_conocido and top1_ab == q.actor_real),
                "acierto_top3": bool(es_conocido and q.actor_real in top3_ab),
                "s1": round(s1_ab, 4),
                "s2": round(s2_ab, 4) if np.isfinite(s2_ab) else None,
                "margen_12": round(s1_ab - s2_ab, 4) if np.isfinite(s2_ab) else 0.0,
                "contexto_n_candidatos": len(cands_ctx_slugs),
                "contexto_top1_predicho": top1_ctx,
                "contexto_acierto_top1": bool(es_conocido and top1_ctx == q.actor_real),
                "contexto_acierto_top3": bool(es_conocido and q.actor_real in top3_ctx),
                "contexto_s1": round(s1_ctx, 4),
                "contexto_margen_12": round(s1_ctx - s2_ctx, 4) if np.isfinite(s2_ctx) else 0.0,
            }
        )

    # Calibración cruzada (split mitad/mitad) para umbrales de rechazo y calibrador de probabilidades
    n_m = len(muestras_rechazo_abierto)
    if n_m >= 4:
        idx_perm = rng.permutation(n_m)
        mitad = n_m // 2
        val_set = [muestras_rechazo_abierto[int(i)] for i in idx_perm[:mitad]]
        test_set = [muestras_rechazo_abierto[int(i)] for i in idx_perm[mitad:]]
    else:
        val_set = test_set = muestras_rechazo_abierto

    umbrales_cal = calibrar_umbrales(
        val_set,
        max_fpr_desconocidos=cfg.max_fpr_desconocidos,
        min_precision_conocidos=cfg.min_precision_conocidos,
    )
    calibrador_prob = CalibradorConfianza().ajustar(val_set)

    # Anotar cada consulta con la decisión de rechazo (abierta y contextual)
    for fila, m_ab in zip(resultados_detalle, muestras_rechazo_abierto):
        dec_ab = clasificar_consulta(
            m_ab.puntuaciones,
            m_ab.ids_candidatos,
            umbrales_cal,
            n_galeria_total=m_ab.n_galeria_total,
            con_contexto=False,
            calibrador=calibrador_prob,
        )
        fila["estado_rechazo"] = dec_ab.estado.value
        fila["porcentaje_ui_actual"] = dec_ab.porcentaje_interfaz_actual
        fila["confianza_calibrada"] = round(dec_ab.confianza_calibrada or 0.0, 4)

    metricas_globales = _calcular_top_metricas(resultados_detalle)
    conocidas_ctx = [r for r in resultados_detalle if r["es_actor_conocido"]]
    n_ctx = len(conocidas_ctx)
    metricas_contexto = {
        "sin_contexto_top1_pct": metricas_globales["top1_pct"],
        "sin_contexto_top3_pct": metricas_globales["top3_pct"],
        "con_contexto_top1_pct": round(100.0 * sum(1 for r in conocidas_ctx if r["contexto_acierto_top1"]) / n_ctx, 2)
        if n_ctx
        else 0.0,
        "con_contexto_top3_pct": round(100.0 * sum(1 for r in conocidas_ctx if r["contexto_acierto_top3"]) / n_ctx, 2)
        if n_ctx
        else 0.0,
        "ganancia_contexto_top1_puntos": 0.0,
        "tamano_medio_reparto": round(float(np.mean([r["contexto_n_candidatos"] for r in conocidas_ctx])), 1)
        if n_ctx
        else 0.0,
    }
    metricas_contexto["ganancia_contexto_top1_puntos"] = round(
        metricas_contexto["con_contexto_top1_pct"] - metricas_contexto["sin_contexto_top1_pct"], 2
    )

    # Desglose en una sola pasada por fuente, duración, actor y grabación
    conocidas_por_fuente: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conocidas_por_duracion: dict[float, list[dict[str, Any]]] = defaultdict(list)
    conocidas_por_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conocidas_por_grabacion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in resultados_detalle:
        if r["es_actor_conocido"]:
            conocidas_por_fuente[r["fuente"]].append(r)
            conocidas_por_duracion[r["duracion_seg"]].append(r)
            if r["actor_real"]:
                conocidas_por_actor[r["actor_real"]].append(r)
            conocidas_por_grabacion[r["grupo_grabacion"]].append(r)

    por_fuente = {f: _calcular_top_metricas(conocidas_por_fuente[f]) for f in sorted(conocidas_por_fuente)}
    por_duracion = {f"{d:g}s": _calcular_top_metricas(conocidas_por_duracion[d]) for d in sorted(conocidas_por_duracion)}
    por_actor = {a: _calcular_top_metricas(conocidas_por_actor[a]) for a in sorted(conocidas_por_actor)}
    por_grabacion = {g: _calcular_top_metricas(conocidas_por_grabacion[g]) for g in sorted(conocidas_por_grabacion)}

    rechazo_abierto_eval = evaluar_rechazo(test_set, umbrales_cal, calibrador=calibrador_prob)
    rechazo_contexto_eval = (
        evaluar_rechazo(muestras_rechazo_contexto, umbrales_cal, calibrador=calibrador_prob)
        if muestras_rechazo_contexto
        else {}
    )
    calibracion_eval = evaluar_calibracion_porcentajes(test_set, calibrador=calibrador_prob)

    return {
        "fecha_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuracion": asdict(cfg),
        "resumen_galeria": {
            "num_actores_totales": len(actores_todos),
            "num_actores_galeria": len(actores_galeria),
            "num_actores_desconocidos_retenidos": len(actores_desconocidos),
            "num_trozos_galeria": int(len(X_ref)),
            "dim_embedding": int(X_ref.shape[1]),
        },
        "metricas_globales": metricas_globales,
        "comparativa_contexto": metricas_contexto,
        "por_fuente": por_fuente,
        "por_duracion": por_duracion,
        "por_actor": por_actor,
        "por_grabacion": por_grabacion,
        "umbrales_rechazo_calibrados": umbrales_cal.a_dict(),
        "evaluacion_voces_desconocidas": {
            "busqueda_abierta": rechazo_abierto_eval,
            "busqueda_contextual": rechazo_contexto_eval,
        },
        "calibracion_porcentajes": calibracion_eval,
        "consultas_detalle": resultados_detalle,
    }


def exportar_informe_json_csv(informe: dict[str, Any], carpeta_salida: str | Path) -> dict[str, Path]:
    """Escribe el informe completo en JSON y tablas desglosadas en CSV."""
    dest = Path(carpeta_salida)
    dest.mkdir(parents=True, exist_ok=True)

    ruta_json = dest / "informe_evaluacion.json"
    ruta_json.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. Resumen CSV
    ruta_resumen = dest / "resumen_metricas.csv"
    glob = informe["metricas_globales"]
    ctx = informe["comparativa_contexto"]
    rec = informe["evaluacion_voces_desconocidas"]["busqueda_abierta"].get("con_rechazo", {})
    with open(ruta_resumen, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerow(["version_modelo", informe["configuracion"]["version_modelo"]])
        w.writerow(["nombre_experimento", informe["configuracion"]["nombre_experimento"]])
        w.writerow(["num_actores_galeria", informe["resumen_galeria"]["num_actores_galeria"]])
        w.writerow(["num_actores_desconocidos", informe["resumen_galeria"]["num_actores_desconocidos_retenidos"]])
        w.writerow(["n_consultas_conocidas", glob["n_consultas"]])
        w.writerow(["top1_pct", glob["top1_pct"]])
        w.writerow(["top3_pct", glob["top3_pct"]])
        w.writerow(["con_contexto_top1_pct", ctx["con_contexto_top1_pct"]])
        w.writerow(["con_contexto_top3_pct", ctx["con_contexto_top3_pct"]])
        w.writerow(["desconocidas_fpr_identificado_pct", round(rec.get("desconocidas_falso_positivo_identificado_pct", 0.0), 2)])
        w.writerow(["desconocidas_rechazo_estricto_pct", round(rec.get("desconocidas_rechazadas_estrictas_pct", 0.0), 2)])
        w.writerow(["precision_en_identificadas_conocidas_pct", round(rec.get("precision_en_identificadas_conocidas_pct", 0.0), 2)])

    # 2. Desglose por actor CSV
    ruta_actores = dest / "evaluacion_por_actor.csv"
    with open(ruta_actores, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["actor", "n_consultas", "top1_pct", "top3_pct", "similitud_media_top1", "margen_medio_12"])
        for actor, m in informe["por_actor"].items():
            w.writerow([actor, m["n_consultas"], m["top1_pct"], m["top3_pct"], m["similitud_media_top1"], m["margen_medio_12"]])

    # 3. Desglose por fuente y duración CSV
    ruta_segmentos = dest / "evaluacion_por_fuente_y_duracion.csv"
    with open(ruta_segmentos, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo_segmento", "clave", "n_consultas", "top1_pct", "top3_pct", "similitud_media_top1"])
        for fuente, m in informe["por_fuente"].items():
            w.writerow(["fuente", fuente, m["n_consultas"], m["top1_pct"], m["top3_pct"], m["similitud_media_top1"]])
        for dur, m in informe["por_duracion"].items():
            w.writerow(["duracion", dur, m["n_consultas"], m["top1_pct"], m["top3_pct"], m["similitud_media_top1"]])

    # 4. Detalle de consultas CSV
    ruta_consultas = dest / "detalle_consultas.csv"
    campos = [
        "consulta_id",
        "actor_real",
        "es_actor_conocido",
        "fuente",
        "duracion_seg",
        "top1_predicho",
        "acierto_top1",
        "acierto_top3",
        "s1",
        "margen_12",
        "estado_rechazo",
        "porcentaje_ui_actual",
        "confianza_calibrada",
        "contexto_n_candidatos",
        "contexto_acierto_top1",
    ]
    with open(ruta_consultas, "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        dw.writeheader()
        for fila in informe["consultas_detalle"]:
            dw.writerow(fila)

    return {
        "json": ruta_json,
        "resumen_csv": ruta_resumen,
        "por_actor_csv": ruta_actores,
        "por_fuente_duracion_csv": ruta_segmentos,
        "detalle_csv": ruta_consultas,
    }


def comparar_versiones(informe_base: dict[str, Any], informe_nuevo: dict[str, Any], tolerancia_caida_pct: float = 0.5) -> dict[str, Any]:
    """Compara dos informes de evaluación (p. ej., dos versiones de modelo) y detecta regresiones."""
    g_base = informe_base["metricas_globales"]
    g_nuevo = informe_nuevo["metricas_globales"]
    ctx_base = informe_base["comparativa_contexto"]
    ctx_nuevo = informe_nuevo["comparativa_contexto"]

    rec_base = informe_base["evaluacion_voces_desconocidas"]["busqueda_abierta"]["con_rechazo"]
    rec_nuevo = informe_nuevo["evaluacion_voces_desconocidas"]["busqueda_abierta"]["con_rechazo"]

    delta_top1 = round(float(g_nuevo["top1_pct"]) - float(g_base["top1_pct"]), 2)
    delta_top3 = round(float(g_nuevo["top3_pct"]) - float(g_base["top3_pct"]), 2)
    delta_ctx_top1 = round(float(ctx_nuevo["con_contexto_top1_pct"]) - float(ctx_base["con_contexto_top1_pct"]), 2)
    delta_fpr = round(
        float(rec_nuevo["desconocidas_falso_positivo_identificado_pct"])
        - float(rec_base["desconocidas_falso_positivo_identificado_pct"]),
        2,
    )

    regresiones: list[str] = []
    if delta_top1 < -tolerancia_caida_pct:
        regresiones.append(f"Caída en Top-1 global: {delta_top1:+.2f} puntos ({g_base['top1_pct']}% -> {g_nuevo['top1_pct']}%)")
    if delta_top3 < -tolerancia_caida_pct:
        regresiones.append(f"Caída en Top-3 global: {delta_top3:+.2f} puntos ({g_base['top3_pct']}% -> {g_nuevo['top3_pct']}%)")
    if delta_ctx_top1 < -tolerancia_caida_pct:
        regresiones.append(
            f"Caída en Top-1 con contexto: {delta_ctx_top1:+.2f} puntos "
            f"({ctx_base['con_contexto_top1_pct']}% -> {ctx_nuevo['con_contexto_top1_pct']}%)"
        )
    if delta_fpr > tolerancia_caida_pct:
        regresiones.append(f"Aumento de falsos positivos en voces desconocidas: {delta_fpr:+.2f} puntos")

    return {
        "version_base": informe_base["configuracion"]["version_modelo"],
        "version_nueva": informe_nuevo["configuracion"]["version_modelo"],
        "experimento_base": informe_base["configuracion"]["nombre_experimento"],
        "experimento_nuevo": informe_nuevo["configuracion"]["nombre_experimento"],
        "diferencias": {
            "delta_top1_pct": delta_top1,
            "delta_top3_pct": delta_top3,
            "delta_con_contexto_top1_pct": delta_ctx_top1,
            "delta_fpr_desconocidos_pct": delta_fpr,
        },
        "hay_regresion": len(regresiones) > 0,
        "regresiones_detectadas": regresiones,
    }


def guardar_en_historial(
    informe: dict[str, Any],
    ruta_historial: str | Path,
    tolerancia_caida_pct: float = 0.5,
) -> dict[str, Any]:
    """Registra una entrada resumida en un archivo histórico JSONL y comprueba si hay regresión respecto a la última."""
    ruta = Path(ruta_historial)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    previo: dict[str, Any] | None = None
    if ruta.exists():
        lineas = [ln.strip() for ln in ruta.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lineas:
            previo = json.loads(lineas[-1])

    comparativa = comparar_versiones(previo, informe, tolerancia_caida_pct) if previo else None
    entrada_guardar = {
        "fecha_utc": informe["fecha_utc"],
        "configuracion": informe["configuracion"],
        "resumen_galeria": informe["resumen_galeria"],
        "metricas_globales": informe["metricas_globales"],
        "comparativa_contexto": informe["comparativa_contexto"],
        "evaluacion_voces_desconocidas": informe["evaluacion_voces_desconocidas"],
        "calibracion_porcentajes": informe["calibracion_porcentajes"],
    }
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada_guardar, ensure_ascii=False) + "\n")

    return {
        "guardado_en": str(ruta),
        "comparativa_con_anterior": comparativa,
    }


def generar_datos_sinteticos_controlados(
    n_actores: int = 30,
    clips_por_actor: int = 3,
    trozos_por_clip: int = 4,
    dim: int = 256,
    ruido_grabacion: float = 1.4,
    ruido_canal_movil: float = 2.2,
    semilla: int = 42,
) -> tuple[list[MuestraReferencia], list[ConsultaEvaluacion], dict[str, list[str]]]:
    """Genera un banco de pruebas sintético reproducible para verificar el motor cuando `data/` no está disponible.

    Simula:
      - Un vector vocal latente por actor y un subespacio de canal/móvil que la LDA debe aprender a suprimir.
      - Muestras de `eldoblaje` (limpias y `#movil`) y de `voces` (entrevistas con desplazamiento de dominio).
      - Consultas de distintas duraciones (3 s, 6 s, 10 s, 15 s; menor ruido efectivo a mayor duración).
      - Repartos contextuales ("¿Qué estás viendo?").
    """
    rng = np.random.default_rng(semilla)
    # Subespacio de perturbación de canal (primeras 24 dimensiones con alta varianza no vocal)
    direcciones_canal = normaliza(rng.standard_normal((16, dim)).astype(np.float32))
    prototipos_actor = {
        f"actor-{i:03d}": normaliza(rng.standard_normal(dim).astype(np.float32)) for i in range(n_actores)
    }
    actores = sorted(prototipos_actor)

    referencias: list[MuestraReferencia] = []
    consultas: list[ConsultaEvaluacion] = []

    for actor in actores:
        proto = prototipos_actor[actor]
        for c_idx in range(clips_por_actor):
            sesgo_clip = normaliza(rng.standard_normal(16).astype(np.float32) @ direcciones_canal)
            es_entrevista = c_idx == clips_por_actor - 1 and clips_por_actor >= 3
            nombre_clip = f"voz_ep{c_idx:04d}.flac" if es_entrevista else f"Muestras{c_idx or ''}.mp3"
            clip_id = f"{actor}/{nombre_clip}"

            # Trozos limpios de referencia (amplitud de ruido invariante a la dimensión)
            ruido_t = normaliza(rng.standard_normal((trozos_por_clip, dim)).astype(np.float32)) * ruido_grabacion
            vecs_limpios = normaliza(proto[None, :] + 0.35 * sesgo_clip[None, :] + ruido_t)
            referencias.append(MuestraReferencia(actor=actor, clip_id=clip_id, vectores=vecs_limpios))

            # Trozos aumentados "#movil" de la misma grabación original
            sesgo_mov = normaliza(rng.standard_normal(16).astype(np.float32) @ direcciones_canal)
            ruido_m = normaliza(rng.standard_normal((trozos_por_clip, dim)).astype(np.float32)) * ruido_canal_movil
            vecs_movil = normaliza(proto[None, :] + 0.55 * sesgo_mov[None, :] + ruido_m)
            referencias.append(MuestraReferencia(actor=actor, clip_id=f"{clip_id}#movil", vectores=vecs_movil))

            # Generar consultas de móvil simulado sobre los clips de eldoblaje
            if not es_entrevista:
                for dur in (3.0, 6.0, 10.0, 15.0):
                    escala_dur = float(np.sqrt(10.0 / dur))
                    sesgo_q = normaliza(rng.standard_normal(16).astype(np.float32) @ direcciones_canal)
                    ruido_q = normaliza(rng.standard_normal((1, dim)).astype(np.float32)) * (ruido_canal_movil * escala_dur)
                    vec_q = normaliza(proto[None, :] + 0.55 * sesgo_q[None, :] + ruido_q)
                    consultas.append(
                        ConsultaEvaluacion(
                            consulta_id=f"{actor}__clip{c_idx}__{int(dur)}s",
                            actor_real=actor,
                            clip_id=clip_id,
                            vectores=vec_q,
                            duracion_seg=dur,
                            fuente="eldoblaje",
                        )
                    )

    # Crear repartos contextuales simulados de 6 actores por obra
    repartos: dict[str, list[str]] = {}
    for actor in actores:
        otros = [a for a in actores if a != actor]
        comp = rng.choice(otros, size=min(5, len(otros)), replace=False).tolist()
        repartos[actor] = sorted([actor, *comp])

    return referencias, consultas, repartos


def _segmentar_clips_pares(vi: np.ndarray) -> list[tuple[int, int]]:
    """Reconstruye los límites de cada clip `(inicio, L)` en `trozos.bin` explotando el par `(limpio, #movil)`.

    En `indexar.py` y `voces.py`, cada clip de `L` trozos de 6 s se guarda como `L` vectores limpios
    seguidos inmediatamente de `L` vectores `#movil` de esos mismos segmentos, por lo que la diagonal
    desplazada `S[pos + j, pos + L + j]` es máxima dentro de cada clip.
    """
    m = len(vi)
    if m < 2:
        return [(0, m)]
    dp = np.full(m + 1, -1e9, dtype=np.float64)
    prev: list[tuple[int, int] | None] = [None] * (m + 1)
    dp[0] = 0.0
    S = (vi @ vi.T).astype(np.float64)
    for pos in range(0, m - 1, 2):
        if dp[pos] < -1e8:
            continue
        for L in range(1, min(16, (m - pos) // 2 + 1)):
            nxt = pos + 2 * L
            par_sim = sum(float(S[pos + j, pos + L + j]) for j in range(L))
            score = dp[pos] + par_sim - 0.12
            if score > dp[nxt]:
                dp[nxt] = score
                prev[nxt] = (pos, L)
    if prev[m] is None:
        return [(0, m // 2)]
    bloques: list[tuple[int, int]] = []
    cur = m
    while cur > 0:
        paso = prev[cur]
        assert paso is not None
        pos, L = paso
        bloques.append((pos, L))
        cur = pos
    return bloques[::-1]


def cargar_desde_datos_web(
    carpeta_datos: str | Path,
) -> tuple[list[MuestraReferencia], list[ConsultaEvaluacion], dict[str, list[str]]]:
    """Carga los 836 actores, sus 22.946 trozos reales y las obras de `frontend/datos/` para evaluación real."""
    carpeta_datos = Path(carpeta_datos)
    meta = json.loads((carpeta_datos / "actores.json").read_text(encoding="utf-8"))
    actores = meta["actores"]
    slugs = [a["id"] for a in actores]
    dim = int(meta["dim"])
    nA = len(slugs)

    C = np.fromfile(carpeta_datos / "centroides.bin", dtype="<f4").reshape(nA, dim).astype(np.float64)
    raw_troz = (carpeta_datos / "trozos.bin").read_bytes()
    inicio = np.frombuffer(raw_troz, dtype="<u4", count=nA + 1)
    n_total = int(inicio[-1])
    escala = np.frombuffer(raw_troz, dtype="<f4", count=n_total, offset=(nA + 1) * 4).astype(np.float64)
    q8 = np.frombuffer(raw_troz, dtype=np.int8, count=n_total * dim, offset=(nA + 1 + n_total) * 4).reshape(n_total, dim).astype(np.float64)
    V = normaliza((q8 * escala[:, None]).astype(np.float32))

    referencias: list[MuestraReferencia] = []
    consultas: list[ConsultaEvaluacion] = []

    for i, slug in enumerate(slugs):
        vi = V[inicio[i] : inicio[i + 1]]
        if len(vi) == 0:
            continue
        m_global = normaliza(vi.mean(axis=0))
        err0 = float(np.linalg.norm(C[i] - m_global))
        k_voz = len(vi)
        if err0 >= 0.01 and len(vi) >= 4:
            mejor_k, mejor_err = len(vi), err0
            for k in range(2, len(vi), 2):
                c_est = normaliza(normaliza(vi[:k].mean(axis=0)) + 0.5 * normaliza(vi[k:].mean(axis=0)))
                e = float(np.linalg.norm(C[i] - c_est))
                if e < mejor_err:
                    mejor_err, mejor_k = e, k
            k_voz = mejor_k

        bloques_eld = _segmentar_clips_pares(vi[:k_voz]) if k_voz > 0 else []
        bloques_voz = _segmentar_clips_pares(vi[k_voz:]) if k_voz < len(vi) else []
        n_clips_actor = len(bloques_eld) + len(bloques_voz)

        for c_idx, (pos, L) in enumerate(bloques_eld):
            limpios = vi[pos : pos + L]
            moviles = vi[pos + L : pos + 2 * L]
            clip_id = f"{slug}/Muestras{c_idx or ''}.mp3"
            if len(limpios):
                referencias.append(MuestraReferencia(actor=slug, clip_id=clip_id, vectores=limpios))
            if len(moviles):
                referencias.append(MuestraReferencia(actor=slug, clip_id=f"{clip_id}#movil", vectores=moviles))
                if n_clips_actor >= 2:
                    for n_tr, dur_s in ((1, 6.0), (2, 12.0), (3, 18.0)):
                        if len(moviles) >= n_tr:
                            consultas.append(
                                ConsultaEvaluacion(
                                    consulta_id=f"{slug}__clip{c_idx}__{int(dur_s)}s",
                                    actor_real=slug,
                                    clip_id=clip_id,
                                    vectores=moviles[:n_tr],
                                    duracion_seg=dur_s,
                                    fuente="eldoblaje",
                                )
                            )

        for v_idx, (pos, L) in enumerate(bloques_voz):
            base_pos = k_voz + pos
            limpios = vi[base_pos : base_pos + L]
            moviles = vi[base_pos + L : base_pos + 2 * L]
            clip_id = f"{slug}/voz_ep{v_idx:03d}.flac"
            if len(limpios):
                referencias.append(MuestraReferencia(actor=slug, clip_id=clip_id, vectores=limpios))
            if len(moviles):
                referencias.append(MuestraReferencia(actor=slug, clip_id=f"{clip_id}#movil", vectores=moviles))

    repartos: dict[str, list[str]] = {}
    ruta_busq = carpeta_datos / "busqueda.json"
    if ruta_busq.exists():
        entradas = json.loads(ruta_busq.read_text(encoding="utf-8"))
        for tipo, _texto, ids in entradas:
            if tipo == "obra" and 3 <= len(ids) <= 25:
                reparto_slugs = [slugs[idx] for idx in ids if 0 <= idx < len(slugs)]
                for s in reparto_slugs:
                    if s not in repartos:
                        repartos[s] = reparto_slugs

    return referencias, consultas, repartos


def cargar_desde_indice_y_cache(
    ruta_indice: Path,
    carpeta_cache_consultas: Path | None = None,
) -> tuple[list[MuestraReferencia], list[ConsultaEvaluacion]]:
    """Carga referencias reales desde `data/indice.npz` y consultas desde la caché de evaluación."""
    datos = np.load(ruta_indice)
    extraer_info_indice(datos)
    X, y, clips = datos["vectores"], datos["etiquetas"], datos["clips"]

    # Agrupar trozos del índice por (actor, clip)
    por_clip: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for vec, act, clp in zip(X, y, clips):
        por_clip[(str(act), str(clp))].append(vec)

    referencias = [
        MuestraReferencia(actor=act, clip_id=clp, vectores=np.stack(vs).astype(np.float32))
        for (act, clp), vs in sorted(por_clip.items())
    ]

    consultas: list[ConsultaEvaluacion] = []
    if carpeta_cache_consultas and carpeta_cache_consultas.exists():
        for f in sorted(carpeta_cache_consultas.glob("*.npz")):
            d = np.load(f)
            actor, _, nom = f.stem.partition("__")
            clip_id = f"{actor}/{nom}.mp3"
            for k in sorted(d.files):
                if k.startswith(("q10_", "vb_q", "q")):
                    arr = d[k]
                    if arr.ndim == 2 and len(arr) > 0:
                        dur = 10.0 if ("10" in k or "vb_q" in k) else 8.0
                        consultas.append(
                            ConsultaEvaluacion(
                                consulta_id=f"{f.stem}__{k}",
                                actor_real=actor,
                                clip_id=clip_id,
                                vectores=normaliza(arr.astype(np.float32)),
                                duracion_seg=dur,
                            )
                        )
    return referencias, consultas


def main() -> None:
    p = argparse.ArgumentParser(description="Sistema reproducible de evaluación de EstaVoz")
    p.add_argument("--indice", type=Path, default=RAIZ / "data" / "indice.npz")
    p.add_argument("--cache-consultas", type=Path, default=RAIZ / "data" / "cache_huellas" / "r100-largos2")
    p.add_argument("--datos-web", type=Path, default=RAIZ / "frontend" / "datos", help="Evaluar sobre artefactos web publicados")
    p.add_argument("--grabaciones-reales", type=Path, help="Manifiesto JSON de grabaciones reales con permiso verificado")
    p.add_argument("--sintetico", action="store_true", help="Ejecutar verificación con datos sintéticos controlados")
    p.add_argument("--solo-galeria-independiente", action="store_true")
    p.add_argument("--usar-cov-acustica", action="store_true", help="Inyecta covarianza de canal acústico en Sw de LDA")
    p.add_argument("--salida", type=Path, default=RAIZ / "data" / "evaluaciones" / "ultima")
    p.add_argument("--historial", type=Path, default=RAIZ / "data" / "evaluaciones" / "historial.jsonl")
    args = p.parse_args()

    entrenar_lda = True
    if args.sintetico:
        refs, cons, repartos = generar_datos_sinteticos_controlados()
        nombre_exp = "verificacion_sintetica_controlada"
    elif args.indice.exists():
        refs, cons = cargar_desde_indice_y_cache(args.indice, args.cache_consultas)
        repartos = None
        nombre_exp = "evaluacion_corpus_real" + ("_independiente" if args.solo_galeria_independiente else "")
    elif (args.datos_web / "trozos.bin").exists():
        refs, cons, repartos = cargar_desde_datos_web(args.datos_web)
        entrenar_lda = False  # `trozos.bin` ya está en el subespacio LDA publicado
        nombre_exp = "evaluacion_artefactos_web_publicados" + ("_independiente" if args.solo_galeria_independiente else "")
    else:
        sys.exit(
            f"No se encontró {args.indice} ni {args.datos_web / 'trozos.bin'}. "
            "Descargue los datos o ejecute con `--sintetico`."
        )

    if args.grabaciones_reales:
        cons.extend(cargar_grabaciones_reales(args.grabaciones_reales))

    cfg = ConfigEvaluacion(
        nombre_experimento=nombre_exp,
        entrenar_lda_en_pliegues=entrenar_lda,
        solo_galeria_independiente=args.solo_galeria_independiente,
        usar_cov_acustica=args.usar_cov_acustica,
    )
    informe = ejecutar_evaluacion_completa(refs, cons, config=cfg, repartos_contexto=repartos)
    rutas = exportar_informe_json_csv(informe, args.salida)
    hist = guardar_en_historial(informe, args.historial)

    glob = informe["metricas_globales"]
    ctx = informe["comparativa_contexto"]
    rec = informe["evaluacion_voces_desconocidas"]["busqueda_abierta"]["con_rechazo"]
    print(f"Experimento: {nombre_exp} ({glob['n_consultas']} consultas conocidas)")
    print(f"  Sin contexto -> Top-1: {glob['top1_pct']:.1f}% | Top-3: {glob['top3_pct']:.1f}%")
    print(f"  Con contexto -> Top-1: {ctx['con_contexto_top1_pct']:.1f}% | Top-3: {ctx['con_contexto_top3_pct']:.1f}%")
    print(
        f"  Con rechazo  -> Precisión en identificadas: {rec['precision_en_identificadas_conocidas_pct']:.1f}% | "
        f"FPR desconocidas: {rec['desconocidas_falso_positivo_identificado_pct']:.1f}% | "
        f"Rechazo desconocidas: {rec['desconocidas_no_identificadas_pct']:.1f}%"
    )
    print(f"  Informe JSON : {rutas['json']}")
    if hist["comparativa_con_anterior"] and hist["comparativa_con_anterior"]["hay_regresion"]:
        print("  ! REGRESIONES DETECTADAS:", hist["comparativa_con_anterior"]["regresiones_detectadas"])


if __name__ == "__main__":
    main()
