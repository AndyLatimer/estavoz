"""Reconocimiento de voces desconocidas (open-set) y calibración experimental de confianza.

Implementa la Fase 3 del plan técnico sin modificar el comportamiento en producción:
  1. Clasificación en tres estados:
     - `identificado`: identificación suficientemente fiable.
     - `dudoso`: identificación dudosa con varios candidatos próximos o similitud moderada.
     - `desconocido`: voz no identificada (ausente de la base o audio sin coincidencia suficiente).
  2. Señales combinadas:
     - Similitud absoluta del candidato principal (`s1`).
     - Margen entre el primer y segundo candidato (`s1 - s2`).
     - Z-score respecto a la cohorte de candidatos evaluados (`z_cohorte`).
     - Adaptación al tamaño de búsqueda contextual ("¿Qué estás viendo?", incluyendo N=1).
  3. Calibración automática de umbrales a partir de un conjunto de validación (voces conocidas
     y voces ausentes de la galería), sin depender de umbrales arbitrarios fijos.
  4. Evaluación de calibración probabilística (ECE y Brier score) del porcentaje actual de la
     interfaz (`aPorcentaje` en `frontend/index.html`) frente a un calibrador isotónico/logístico.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


class EstadoIdentificacion(str, Enum):
    IDENTIFICADO = "identificado"
    DUDOSO = "dudoso"
    DESCONOCIDO = "desconocido"


@dataclass(frozen=True)
class SenalesConsulta:
    """Señales extraídas de la distribución de similitudes de una consulta."""
    s1: float
    s2: float
    s3: float
    margen_12: float
    margen_13: float
    z_cohorte: float
    n_candidatos: int
    n_galeria_total: int
    con_contexto: bool

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UmbralesRechazo:
    """Umbrales calibrados sobre datos de validación para decidir entre los 3 estados."""
    umbral_fiable: float
    umbral_dudoso: float
    margen_fiable: float
    margen_dudoso: float
    z_fiable: float
    # Para búsqueda contextual (N_ctx < N_total): descuento máximo por menor colisión de extremo
    descuento_contexto_max: float = 0.035
    # Suelo de verificación 1-a-1 cuando el filtro contextual deja un único candidato (N_ctx == 1)
    umbral_1a1_fiable: float = 0.30
    umbral_1a1_dudoso: float = 0.23
    # Penalización logarítmica de exceso de trozos para corregir la ventaja estadística de 'actores imán' (hubness)
    beta_exceso_trozos: float = 0.012
    origen_calibracion: str = "validacion"
    metricas_validacion: dict[str, float] | None = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, d: dict[str, Any]) -> UmbralesRechazo:
        return cls(
            umbral_fiable=float(d["umbral_fiable"]),
            umbral_dudoso=float(d["umbral_dudoso"]),
            margen_fiable=float(d["margen_fiable"]),
            margen_dudoso=float(d["margen_dudoso"]),
            z_fiable=float(d["z_fiable"]),
            descuento_contexto_max=float(d.get("descuento_contexto_max", 0.035)),
            umbral_1a1_fiable=float(d.get("umbral_1a1_fiable", 0.30)),
            umbral_1a1_dudoso=float(d.get("umbral_1a1_dudoso", 0.23)),
            beta_exceso_trozos=float(d.get("beta_exceso_trozos", 0.012)),
            origen_calibracion=str(d.get("origen_calibracion", "dict")),
            metricas_validacion=d.get("metricas_validacion"),
        )


# Umbrales calibrados empíricamente sobre 844 actores independientes (AS-Norm k=50, top-4 trozos)
# con VoxBlink2 SimAM-ResNet100 (VERSION_MODELO = "3"), logrando ~99,4 % de precisión en
# identificaciones fiables y 96,3 % de rechazo/duda en voces desconocidas.
UMBRALES_VOXBLINK_R100_CALIBRADOS = UmbralesRechazo(
    umbral_fiable=0.5087,
    umbral_dudoso=0.4756,
    margen_fiable=0.1438,
    margen_dudoso=0.0647,
    z_fiable=5.3457,
    descuento_contexto_max=0.035,
    umbral_1a1_fiable=0.4756,
    umbral_1a1_dudoso=0.4376,
    beta_exceso_trozos=0.012,
    origen_calibracion="voxblink2_r100_asnorm_844_actores",
)


@dataclass(frozen=True)
class DecisionRechazo:
    estado: EstadoIdentificacion
    candidato_principal: str | None
    candidatos_dudosos: list[str]
    puntuacion_principal: float
    margen_12: float
    z_cohorte: float
    porcentaje_interfaz_actual: int
    confianza_calibrada: float | None
    motivo: str

    def a_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["estado"] = self.estado.value
        return d


@dataclass(frozen=True)
class MuestraValidacionRechazo:
    """Muestra etiquetada para calibrar y evaluar el detector de voces desconocidas."""
    puntuaciones: np.ndarray
    ids_candidatos: list[str]
    actor_real: str | None          # None si es una voz ausente de la base (desconocida)
    es_conocido: bool               # True si actor_real está entre ids_candidatos
    n_galeria_total: int
    con_contexto: bool = False


# Nodos empíricos (similitud AS-Norm, porcentaje) calibrados sobre 3.419 consultas independientes
# (conocidas + desconocidas) del modelo VoxBlink2 SimAM-ResNet100 (VERSION_MODELO = "3", ECE = 0.0124).
CURVA_CALIBRADA_R100: tuple[tuple[float, float], ...] = (
    (0.25, 0.0),
    (0.35, 14.0),
    (0.42, 21.0),
    (0.47, 30.0),
    (0.52, 46.0),
    (0.58, 65.0),
    (0.66, 86.0),
    (0.74, 96.0),
    (0.85, 100.0),
)


def porcentaje_interfaz_actual(similitud: float) -> int:
    """Reproduce exactamente la función lineal histórica `aPorcentaje(s)` de `frontend/index.html`."""
    pct = max(0.0, min(1.0, (float(similitud) - 0.15) / 0.35)) * 100.0
    return int(np.floor(pct + 0.5))


def porcentaje_calibrado_r100(similitud: float) -> int:
    """Traduce la similitud `s` a porcentaje calibrado por tramos sobre VoxBlink2 ResNet100."""
    xs = [p[0] for p in CURVA_CALIBRADA_R100]
    ys = [p[1] for p in CURVA_CALIBRADA_R100]
    val = float(np.interp(float(similitud), xs, ys))
    return int(np.floor(max(0.0, min(100.0, val)) + 0.5))


def extraer_senales(
    puntuaciones: np.ndarray,
    n_galeria_total: int | None = None,
    con_contexto: bool = False,
    max_cohorte: int = 50,
) -> SenalesConsulta:
    """Calcula las señales de decisión a partir del vector de similitudes de los candidatos."""
    p = np.asarray(puntuaciones, dtype=np.float64)
    p = p[np.isfinite(p)]
    n_cand = len(p)
    n_total = int(n_galeria_total if n_galeria_total is not None else max(n_cand, 1))
    if n_cand == 0:
        return SenalesConsulta(
            s1=-1.0,
            s2=-1.0,
            s3=-1.0,
            margen_12=0.0,
            margen_13=0.0,
            z_cohorte=0.0,
            n_candidatos=0,
            n_galeria_total=n_total,
            con_contexto=con_contexto,
        )

    orden = np.sort(p)[::-1]
    s1 = float(orden[0])
    s2 = float(orden[1]) if n_cand >= 2 else float("nan")
    s3 = float(orden[2]) if n_cand >= 3 else (s2 if n_cand == 2 else float("nan"))
    margen_12 = float(s1 - s2) if n_cand >= 2 else 0.0
    margen_13 = float(s1 - s3) if n_cand >= 3 else margen_12

    if n_cand >= 3:
        cohorte = orden[1 : min(n_cand, max_cohorte + 1)]
        mu_c = float(cohorte.mean())
        sd_c = float(cohorte.std())
        z_cohorte = (s1 - mu_c) / max(sd_c, 1e-4)
    elif n_cand == 2:
        z_cohorte = margen_12 / 0.05
    else:
        z_cohorte = 0.0

    return SenalesConsulta(
        s1=s1,
        s2=s2,
        s3=s3,
        margen_12=margen_12,
        margen_13=margen_13,
        z_cohorte=float(z_cohorte),
        n_candidatos=n_cand,
        n_galeria_total=n_total,
        con_contexto=con_contexto,
    )


def calcular_estadisticas_cohorte(centroides: np.ndarray, k: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Calcula `(mu_a, sd_a)` por actor sobre sus `k` vecinos más cercanos en la galería (sin fuga)."""
    C = np.asarray(centroides, dtype=np.float64)
    n = len(C)
    if n <= 1:
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
    normas = np.linalg.norm(C, axis=1, keepdims=True)
    Cn = np.where(normas > 0, C / np.maximum(normas, 1e-12), 0.0)
    CC = Cn @ Cn.T
    np.fill_diagonal(CC, -np.inf)
    k_eff = min(max(1, int(k)), n - 1)
    top = -np.sort(-CC, axis=1)[:, :k_eff]
    mu_a = top.mean(axis=1).astype(np.float32)
    sd_a = np.maximum(top.std(axis=1), 1e-4).astype(np.float32)
    return mu_a, sd_a


def aplicar_as_norm(
    puntuaciones: np.ndarray,
    mu_a: np.ndarray | None = None,
    sd_a: np.ndarray | None = None,
    k: int = 50,
    peso: float = 0.04,
) -> np.ndarray:
    """Combina similitud coseno con AS-Norm adaptativo para atenuar actores imán (hubness)."""
    s = np.asarray(puntuaciones, dtype=np.float32)
    if peso <= 0 or len(s) < 4:
        return s
    validos = np.isfinite(s)
    if validos.sum() < 4:
        return s
    orden = np.sort(s[validos])[::-1]
    k_eff = min(max(2, int(k)), len(orden) - 1)
    coh_q = orden[1 : k_eff + 1]
    mu_q = float(coh_q.mean())
    sd_q = max(float(coh_q.std()), 1e-4)
    z_q = (s - mu_q) / sd_q
    if mu_a is not None and sd_a is not None and len(mu_a) == len(s):
        z_a = (s - mu_a) / np.maximum(sd_a, 1e-4)
        as_val = 0.5 * (z_q + z_a)
    else:
        as_val = z_q
    return np.where(validos, s + float(peso) * as_val, s).astype(np.float32)


def _descuento_contextual(n_candidatos: int, n_total: int, max_descuento: float) -> float:
    """Ajuste de umbral por menor máximo espurio al reducir el conjunto de candidatos."""
    if n_candidatos <= 1 or n_total <= 2 or n_candidatos >= n_total:
        return 0.0
    ratio_log = 1.0 - np.sqrt(np.log(max(n_candidatos, 2)) / np.log(max(n_total, 2)))
    return float(np.clip(ratio_log * max_descuento, 0.0, max_descuento))


def clasificar_consulta(
    puntuaciones: np.ndarray,
    ids_candidatos: list[str],
    umbrales: UmbralesRechazo,
    n_galeria_total: int | None = None,
    con_contexto: bool = False,
    calibrador: CalibradorConfianza | None = None,
    max_dudosos: int = 3,
) -> DecisionRechazo:
    """Clasifica una consulta en `identificado`, `dudoso` o `desconocido`."""
    p = np.asarray(puntuaciones, dtype=np.float64)
    if len(p) != len(ids_candidatos):
        raise ValueError("`puntuaciones` e `ids_candidatos` deben tener la misma longitud.")

    mask_ok = np.isfinite(p)
    if not np.any(mask_ok):
        return DecisionRechazo(
            estado=EstadoIdentificacion.DESCONOCIDO,
            candidato_principal=None,
            candidatos_dudosos=[],
            puntuacion_principal=-1.0,
            margen_12=0.0,
            z_cohorte=0.0,
            porcentaje_interfaz_actual=0,
            confianza_calibrada=0.0,
            motivo="sin_candidatos_validos",
        )

    orden_idx = np.argsort(-np.where(mask_ok, p, -np.inf))
    p_ord = p[orden_idx]
    ids_ord = [ids_candidatos[int(i)] for i in orden_idx if mask_ok[int(i)]]
    p_ord = p_ord[: len(ids_ord)]

    senales = extraer_senales(p_ord, n_galeria_total=n_galeria_total, con_contexto=con_contexto)
    pct_ui = porcentaje_interfaz_actual(senales.s1)
    conf_cal = calibrador.predecir_probabilidad(senales) if calibrador is not None else None

    # Caso especial: filtro contextual con un único candidato (verificación 1-a-1, sin margen s1-s2)
    if senales.n_candidatos == 1:
        if senales.s1 >= umbrales.umbral_1a1_fiable:
            return DecisionRechazo(
                estado=EstadoIdentificacion.IDENTIFICADO,
                candidato_principal=ids_ord[0],
                candidatos_dudosos=[ids_ord[0]],
                puntuacion_principal=senales.s1,
                margen_12=0.0,
                z_cohorte=0.0,
                porcentaje_interfaz_actual=pct_ui,
                confianza_calibrada=conf_cal,
                motivo="verificacion_1a1_fiable",
            )
        if senales.s1 >= umbrales.umbral_1a1_dudoso:
            return DecisionRechazo(
                estado=EstadoIdentificacion.DUDOSO,
                candidato_principal=ids_ord[0],
                candidatos_dudosos=[ids_ord[0]],
                puntuacion_principal=senales.s1,
                margen_12=0.0,
                z_cohorte=0.0,
                porcentaje_interfaz_actual=pct_ui,
                confianza_calibrada=conf_cal,
                motivo="verificacion_1a1_dudosa",
            )
        return DecisionRechazo(
            estado=EstadoIdentificacion.DESCONOCIDO,
            candidato_principal=None,
            candidatos_dudosos=[],
            puntuacion_principal=senales.s1,
            margen_12=0.0,
            z_cohorte=0.0,
            porcentaje_interfaz_actual=pct_ui,
            confianza_calibrada=conf_cal,
            motivo="verificacion_1a1_rechazada",
        )

    descuento = (
        _descuento_contextual(senales.n_candidatos, senales.n_galeria_total, umbrales.descuento_contexto_max)
        if con_contexto
        else 0.0
    )
    u_fiable = max(umbrales.umbral_1a1_dudoso, umbrales.umbral_fiable - descuento)
    u_dudoso = max(0.15, umbrales.umbral_dudoso - descuento)

    # 1. ¿Es suficientemente fiable?
    # Requiere superar el umbral de similitud absoluta Y tener separación clara (margen o z-score)
    # o bien una similitud muy alta (u_fiable + margen_fiable) con margen al menos dudoso.
    sep_clara = (senales.margen_12 >= umbrales.margen_fiable) and (senales.z_cohorte >= umbrales.z_fiable)
    sim_muy_alta = (senales.s1 >= u_fiable + umbrales.margen_fiable) and (senales.margen_12 >= umbrales.margen_dudoso)

    if senales.s1 >= u_fiable and (sep_clara or sim_muy_alta):
        return DecisionRechazo(
            estado=EstadoIdentificacion.IDENTIFICADO,
            candidato_principal=ids_ord[0],
            candidatos_dudosos=[ids_ord[0]],
            puntuacion_principal=senales.s1,
            margen_12=senales.margen_12,
            z_cohorte=senales.z_cohorte,
            porcentaje_interfaz_actual=pct_ui,
            confianza_calibrada=conf_cal,
            motivo="similitud_y_margen_fiables",
        )

    # 2. ¿Es dudosa con varios candidatos?
    # O bien supera u_fiable pero el 2.º candidato está demasiado cerca, o está entre u_dudoso y u_fiable
    # con margen mínimo frente al ruido de fondo.
    en_zona_dudosa = (senales.s1 >= u_fiable) or (
        senales.s1 >= u_dudoso and (senales.margen_12 >= umbrales.margen_dudoso or senales.z_cohorte >= umbrales.z_fiable)
    )
    if en_zona_dudosa:
        # Incluir los candidatos cercanos al primero (dentro de max(margen_fiable, 0.04))
        ventana = max(umbrales.margen_fiable, 0.04)
        cercanos = [
            cid for cid, sc in zip(ids_ord[:max_dudosos], p_ord[:max_dudosos]) if (senales.s1 - float(sc)) <= ventana
        ]
        if not cercanos:
            cercanos = ids_ord[: min(2, len(ids_ord))]
        motivo = "empate_entre_candidatos" if senales.s1 >= u_fiable else "similitud_intermedia"
        return DecisionRechazo(
            estado=EstadoIdentificacion.DUDOSO,
            candidato_principal=ids_ord[0],
            candidatos_dudosos=cercanos,
            puntuacion_principal=senales.s1,
            margen_12=senales.margen_12,
            z_cohorte=senales.z_cohorte,
            porcentaje_interfaz_actual=pct_ui,
            confianza_calibrada=conf_cal,
            motivo=motivo,
        )

    # 3. Voz no identificada (rechazada)
    return DecisionRechazo(
        estado=EstadoIdentificacion.DESCONOCIDO,
        candidato_principal=None,
        candidatos_dudosos=[],
        puntuacion_principal=senales.s1,
        margen_12=senales.margen_12,
        z_cohorte=senales.z_cohorte,
        porcentaje_interfaz_actual=pct_ui,
        confianza_calibrada=conf_cal,
        motivo="baja_similitud_o_sin_destacar",
    )


def evaluar_rechazo(
    muestras: list[MuestraValidacionRechazo],
    umbrales: UmbralesRechazo,
    calibrador: CalibradorConfianza | None = None,
) -> dict[str, Any]:
    """Evalúa el impacto del mecanismo de rechazo sobre aciertos conocidos y falsos positivos."""
    conocidas = [m for m in muestras if m.es_conocido]
    desconocidas = [m for m in muestras if not m.es_conocido]

    # Contadores en voces conocidas
    c_ident_ok = c_ident_err = 0
    c_dudoso_top1_ok = c_dudoso_en_lista = c_dudoso_err = 0
    c_rechazado_conocido = 0
    baseline_top1_ok = 0

    for m in conocidas:
        orden = np.argsort(-m.puntuaciones)
        top1_id = m.ids_candidatos[int(orden[0])] if len(m.ids_candidatos) else None
        if top1_id == m.actor_real:
            baseline_top1_ok += 1

        dec = clasificar_consulta(
            m.puntuaciones,
            m.ids_candidatos,
            umbrales,
            n_galeria_total=m.n_galeria_total,
            con_contexto=m.con_contexto,
            calibrador=calibrador,
        )
        if dec.estado == EstadoIdentificacion.IDENTIFICADO:
            if dec.candidato_principal == m.actor_real:
                c_ident_ok += 1
            else:
                c_ident_err += 1
        elif dec.estado == EstadoIdentificacion.DUDOSO:
            if dec.candidato_principal == m.actor_real:
                c_dudoso_top1_ok += 1
            if m.actor_real in dec.candidatos_dudosos:
                c_dudoso_en_lista += 1
            else:
                c_dudoso_err += 1
        else:
            c_rechazado_conocido += 1

    # Contadores en voces desconocidas (fuera de la galería)
    u_fp_ident = u_dudoso = u_rechazo_ok = 0
    u_ui_mayor_50 = 0
    for m in desconocidas:
        s1_u = float(np.max(m.puntuaciones)) if len(m.puntuaciones) else -1.0
        if porcentaje_interfaz_actual(s1_u) >= 50:
            u_ui_mayor_50 += 1
        dec = clasificar_consulta(
            m.puntuaciones,
            m.ids_candidatos,
            umbrales,
            n_galeria_total=m.n_galeria_total,
            con_contexto=m.con_contexto,
            calibrador=calibrador,
        )
        if dec.estado == EstadoIdentificacion.IDENTIFICADO:
            u_fp_ident += 1
        elif dec.estado == EstadoIdentificacion.DUDOSO:
            u_dudoso += 1
        else:
            u_rechazo_ok += 1

    n_c = len(conocidas)
    n_u = len(desconocidas)
    total_ident = c_ident_ok + c_ident_err + u_fp_ident

    return {
        "n_conocidas": n_c,
        "n_desconocidas": n_u,
        # Sin rechazo (comportamiento actual de producción):
        "sin_rechazo": {
            "top1_conocidas_pct": 100.0 * baseline_top1_ok / n_c if n_c else 0.0,
            "fpr_desconocidas_pct": 100.0 if n_u else 0.0,
            "desconocidas_con_ui_ge_50_pct": 100.0 * u_ui_mayor_50 / n_u if n_u else 0.0,
        },
        # Con rechazo en 3 estados:
        "con_rechazo": {
            "conocidas_identificadas_correctas_pct": 100.0 * c_ident_ok / n_c if n_c else 0.0,
            "conocidas_identificadas_erroneas_pct": 100.0 * c_ident_err / n_c if n_c else 0.0,
            "conocidas_dudosas_pct": 100.0 * (c_dudoso_top1_ok + c_dudoso_err + (c_dudoso_en_lista - c_dudoso_top1_ok)) / n_c
            if n_c
            else 0.0,
            "conocidas_dudosas_con_actor_real_pct": 100.0 * c_dudoso_en_lista / n_c if n_c else 0.0,
            "conocidas_rechazadas_pct": 100.0 * c_rechazado_conocido / n_c if n_c else 0.0,
            "precision_en_identificadas_conocidas_pct": (
                100.0 * c_ident_ok / (c_ident_ok + c_ident_err) if (c_ident_ok + c_ident_err) else 100.0
            ),
            "precision_global_identificadas_pct": 100.0 * c_ident_ok / total_ident if total_ident else 100.0,
            "desconocidas_falso_positivo_identificado_pct": 100.0 * u_fp_ident / n_u if n_u else 0.0,
            "desconocidas_dudosas_pct": 100.0 * u_dudoso / n_u if n_u else 0.0,
            "desconocidas_rechazadas_estrictas_pct": 100.0 * u_rechazo_ok / n_u if n_u else 0.0,
            "desconocidas_no_identificadas_pct": 100.0 * (u_rechazo_ok + u_dudoso) / n_u if n_u else 0.0,
        },
    }


def calibrar_umbrales(
    muestras_validacion: list[MuestraValidacionRechazo],
    max_fpr_desconocidos: float = 0.05,
    min_precision_conocidos: float = 0.90,
) -> UmbralesRechazo:
    """Calibra los umbrales de rechazo a partir de muestras de validación (sin valores fijos arbitrarios).

    Busca en los cuantiles empíricos de similitud, margen y z-score la configuración que:
      1. Limita los falsos positivos de voces desconocidas clasificadas como `identificado`
         a `<= max_fpr_desconocidos` (o el mínimo alcanzable en validación).
      2. Mantiene la precisión sobre voces conocidas identificadas `>= min_precision_conocidos`.
      3. Maximiza la cobertura de identificaciones correctas (`conocidas_identificadas_correctas_pct`)
         y la tasa de rechazo estricto de desconocidas.
    """
    if not muestras_validacion:
        raise ValueError("Se requiere al menos una muestra de validación para calibrar umbrales.")

    senales_todas = []
    es_acierto_top1 = []
    es_conocido = []

    for m in muestras_validacion:
        if len(m.puntuaciones) == 0:
            continue
        orden = np.argsort(-m.puntuaciones)
        top1_id = m.ids_candidatos[int(orden[0])]
        sen = extraer_senales(m.puntuaciones[orden], n_galeria_total=m.n_galeria_total, con_contexto=m.con_contexto)
        senales_todas.append(sen)
        es_acierto_top1.append(bool(m.es_conocido and top1_id == m.actor_real))
        es_conocido.append(bool(m.es_conocido))

    s1_arr = np.array([s.s1 for s in senales_todas])
    m12_arr = np.array([s.margen_12 for s in senales_todas])
    z_arr = np.array([s.z_cohorte for s in senales_todas])
    acierto_arr = np.array(es_acierto_top1, dtype=bool)
    conocido_arr = np.array(es_conocido, dtype=bool)
    desconocido_arr = ~conocido_arr

    # Rejilla basada en cuantiles reales de los datos de validación
    cand_u_fiable = np.unique(np.quantile(s1_arr, np.linspace(0.20, 0.92, 13)))
    cand_u_dudoso = np.unique(np.quantile(s1_arr, np.linspace(0.05, 0.65, 9)))
    cand_m_fiable = np.unique(np.quantile(m12_arr, np.linspace(0.15, 0.85, 7)))
    cand_z_fiable = np.unique(np.quantile(z_arr, np.linspace(0.15, 0.75, 5)))

    mejor_tupla: tuple[float, ...] | None = None
    mejor_umbrales: UmbralesRechazo | None = None

    n_c = int(conocido_arr.sum())
    n_u = int(desconocido_arr.sum())

    for u_f in cand_u_fiable:
        for m_f in cand_m_fiable:
            m_d = max(0.005, float(m_f) * 0.45)
            for z_f in cand_z_fiable:
                sep_clara = (m12_arr >= m_f) & (z_arr >= z_f)
                sim_alta = (s1_arr >= (u_f + m_f)) & (m12_arr >= m_d)
                ident = (s1_arr >= u_f) & (sep_clara | sim_alta)

                ident_ok = int((ident & acierto_arr).sum())
                ident_err_c = int((ident & conocido_arr & ~acierto_arr).sum())
                fp_u = int((ident & desconocido_arr).sum())

                fpr_u = fp_u / n_u if n_u > 0 else 0.0
                prec_c = ident_ok / (ident_ok + ident_err_c) if (ident_ok + ident_err_c) > 0 else 1.0
                tpr_c = ident_ok / n_c if n_c > 0 else 0.0

                cumple_fpr = fpr_u <= max_fpr_desconocidos + 1e-9
                cumple_prec = prec_c >= min_precision_conocidos - 1e-9

                # Elegir u_dudoso < u_f que separe las consultas conocidas no identificadas de las desconocidas
                candidatos_d = cand_u_dudoso[cand_u_dudoso < (u_f - 0.015)]
                if len(candidatos_d) == 0:
                    candidatos_d = np.array([max(0.15, float(u_f) - 0.05)])

                restantes_c = conocido_arr & (~ident)
                n_rest_c = int(restantes_c.sum())
                mejor_u_d = float(candidatos_d[0])
                mejor_j_dudoso = -np.inf
                for u_d in candidatos_d:
                    en_dudoso = (~ident) & (s1_arr >= u_d) & ((m12_arr >= m_d) | (z_arr >= z_f * 0.7))
                    rechazado = (~ident) & (~en_dudoso)
                    trr_u = float((rechazado & desconocido_arr).sum()) / n_u if n_u > 0 else 0.0
                    rescata_c = float((en_dudoso & restantes_c).sum()) / n_rest_c if n_rest_c > 0 else 0.0
                    j_score = 0.55 * trr_u + 0.45 * rescata_c
                    if j_score > mejor_j_dudoso:
                        mejor_j_dudoso = j_score
                        mejor_u_d = float(u_d)

                # Orden lexicográfico: primero cumplir restricciones, luego menor FPR, mayor TPR y mayor J dudoso
                clave = (
                    1.0 if (cumple_fpr and cumple_prec) else (0.5 if cumple_fpr else 0.0),
                    -max(0.0, fpr_u - max_fpr_desconocidos),
                    -max(0.0, min_precision_conocidos - prec_c),
                    tpr_c + 0.25 * prec_c - 0.5 * fpr_u + 0.2 * mejor_j_dudoso,
                )
                if mejor_tupla is None or clave > mejor_tupla:
                    mejor_tupla = clave
                    u_1a1_f = float(np.clip(u_f * 0.92, mejor_u_d, u_f))
                    u_1a1_d = float(np.clip(mejor_u_d * 0.92, 0.15, u_1a1_f))
                    mejor_umbrales = UmbralesRechazo(
                        umbral_fiable=round(float(u_f), 4),
                        umbral_dudoso=round(float(mejor_u_d), 4),
                        margen_fiable=round(float(m_f), 4),
                        margen_dudoso=round(float(m_d), 4),
                        z_fiable=round(float(z_f), 4),
                        descuento_contexto_max=0.035,
                        umbral_1a1_fiable=round(u_1a1_f, 4),
                        umbral_1a1_dudoso=round(u_1a1_d, 4),
                        origen_calibracion="validacion_cuantiles",
                    )

    assert mejor_umbrales is not None
    resumen = evaluar_rechazo(muestras_validacion, mejor_umbrales)
    return UmbralesRechazo(
        **{
            **mejor_umbrales.a_dict(),
            "metricas_validacion": {
                **resumen["con_rechazo"],
                "baseline_top1_conocidas_pct": resumen["sin_rechazo"]["top1_conocidas_pct"],
            },
        }
    )


class CalibradorConfianza:
    """Calibrador probabilístico monótono (PAVA isotónico sobre índice lineal de `s1` y `margen_12`).

    Permite comparar la calibración empírica real frente a la fórmula lineal fija `aPorcentaje(s)`
    de `frontend/index.html`.
    """

    def __init__(self, peso_margen: float = 0.8):
        self.peso_margen = float(peso_margen)
        self.umbrales_x: np.ndarray = np.array([0.0, 1.0], dtype=np.float64)
        self.prob_y: np.ndarray = np.array([0.0, 1.0], dtype=np.float64)

    def _puntaje_compuesto(self, s1: np.ndarray, margen_12: np.ndarray) -> np.ndarray:
        return s1 + self.peso_margen * np.nan_to_num(margen_12, nan=0.0)

    @staticmethod
    def _pava_monotono(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm (isotonic regression no decreciente)."""
        n = len(y)
        valores = y.astype(np.float64).copy()
        pesos = np.ones(n, dtype=np.float64)
        i = 0
        while i < n - 1:
            if valores[i] > valores[i + 1]:
                total_w = pesos[i] + pesos[i + 1]
                media = (valores[i] * pesos[i] + valores[i + 1] * pesos[i + 1]) / total_w
                valores[i] = media
                pesos[i] = total_w
                valores = np.delete(valores, i + 1)
                pesos = np.delete(pesos, i + 1)
                n -= 1
                if i > 0:
                    i -= 1
            else:
                i += 1
        # Expandir de nuevo a la longitud original
        out = np.empty(int(pesos.sum()), dtype=np.float64)
        pos = 0
        for v, w in zip(valores, pesos):
            k = int(round(w))
            out[pos : pos + k] = v
            pos += k
        return out

    def ajustar(self, muestras: list[MuestraValidacionRechazo]) -> CalibradorConfianza:
        s1_list, m12_list, y_list = [], [], []
        for m in muestras:
            if len(m.puntuaciones) == 0:
                continue
            orden = np.argsort(-m.puntuaciones)
            top1_id = m.ids_candidatos[int(orden[0])]
            sen = extraer_senales(m.puntuaciones[orden], n_galeria_total=m.n_galeria_total, con_contexto=m.con_contexto)
            s1_list.append(sen.s1)
            m12_list.append(sen.margen_12)
            y_list.append(1.0 if (m.es_conocido and top1_id == m.actor_real) else 0.0)

        x = self._puntaje_compuesto(np.array(s1_list), np.array(m12_list))
        y = np.array(y_list, dtype=np.float64)
        orden = np.argsort(x, kind="stable")
        x_ord, y_ord = x[orden], y[orden]
        y_iso = self._pava_monotono(y_ord)
        self.umbrales_x = x_ord
        self.prob_y = np.clip(y_iso, 0.0, 1.0)
        return self

    def predecir_probabilidad(self, senales: SenalesConsulta) -> float:
        x = float(self._puntaje_compuesto(np.array([senales.s1]), np.array([senales.margen_12]))[0])
        return float(np.interp(x, self.umbrales_x, self.prob_y))


def evaluar_calibracion_porcentajes(
    muestras: list[MuestraValidacionRechazo],
    calibrador: CalibradorConfianza | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compara la calibración probabilística (ECE y Brier Score) de `aPorcentaje` frente al calibrador."""
    prob_ui, prob_cal, acierto_real = [], [], []
    for m in muestras:
        if len(m.puntuaciones) == 0:
            continue
        orden = np.argsort(-m.puntuaciones)
        top1_id = m.ids_candidatos[int(orden[0])]
        sen = extraer_senales(m.puntuaciones[orden], n_galeria_total=m.n_galeria_total, con_contexto=m.con_contexto)
        prob_ui.append(porcentaje_interfaz_actual(sen.s1) / 100.0)
        if calibrador is not None:
            prob_cal.append(calibrador.predecir_probabilidad(sen))
        acierto_real.append(1.0 if (m.es_conocido and top1_id == m.actor_real) else 0.0)

    p_ui = np.array(prob_ui, dtype=np.float64)
    y = np.array(acierto_real, dtype=np.float64)

    def metricas_cal(p: np.ndarray) -> dict[str, float]:
        if len(p) == 0:
            return {"ece": 0.0, "brier": 0.0, "confianza_media": 0.0, "acierto_medio": 0.0}
        brier = float(np.mean((p - y) ** 2))
        bordes = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            if b == n_bins - 1:
                mask = (p >= bordes[b]) & (p <= bordes[b + 1])
            else:
                mask = (p >= bordes[b]) & (p < bordes[b + 1])
            if np.any(mask):
                ece += (mask.sum() / len(p)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
        return {
            "ece": round(float(ece), 4),
            "brier": round(brier, 4),
            "confianza_media": round(float(p.mean()), 4),
            "acierto_medio": round(float(y.mean()), 4),
        }

    salida = {"interfaz_actual_aPorcentaje": metricas_cal(p_ui)}
    if calibrador is not None:
        p_c = np.array(prob_cal, dtype=np.float64)
        salida["calibrador_isotonico"] = metricas_cal(p_c)
    return salida
