"""Pruebas automatizadas del reconocimiento de voces desconocidas y calibración (Fase 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from rechazo import (  # noqa: E402
    CURVA_CALIBRADA_R100,
    UMBRALES_VOXBLINK_R100_CALIBRADOS,
    CalibradorConfianza,
    EstadoIdentificacion,
    MuestraValidacionRechazo,
    UmbralesRechazo,
    calibrar_umbrales,
    clasificar_consulta,
    evaluar_calibracion_porcentajes,
    evaluar_rechazo,
    extraer_senales,
    porcentaje_calibrado_r100,
    porcentaje_interfaz_actual,
)


def test_porcentaje_interfaz_actual_reproduce_js():
    """Verifica equivalencia exacta con `aPorcentaje(s)` y `aPorcentajeCalibrado(s)` de `frontend/index.html`."""
    assert porcentaje_interfaz_actual(0.10) == 0
    assert porcentaje_interfaz_actual(0.15) == 0
    assert porcentaje_interfaz_actual(0.325) == 50
    assert porcentaje_interfaz_actual(0.50) == 100
    assert porcentaje_interfaz_actual(0.65) == 100

    assert porcentaje_calibrado_r100(0.10) == 0
    assert porcentaje_calibrado_r100(0.25) == 0
    assert porcentaje_calibrado_r100(0.42) == 21
    assert porcentaje_calibrado_r100(0.52) == 46
    assert porcentaje_calibrado_r100(0.66) == 86
    assert porcentaje_calibrado_r100(0.85) == 100
    assert len(CURVA_CALIBRADA_R100) == 9


def test_clasificacion_tres_estados_abierto_y_contextual():
    """Comprueba los tres estados: identificado, dudoso y desconocido (incluyendo contexto N=1)."""
    ids = ["actor-a", "actor-b", "actor-c", "actor-d", "actor-e"]
    umbrales = UmbralesRechazo(
        umbral_fiable=0.40,
        umbral_dudoso=0.28,
        margen_fiable=0.06,
        margen_dudoso=0.025,
        z_fiable=2.0,
        umbral_1a1_fiable=0.35,
        umbral_1a1_dudoso=0.25,
    )

    # 1. Identificación claramente fiable (alta similitud y amplio margen frente al resto)
    d1 = clasificar_consulta(np.array([0.52, 0.24, 0.21, 0.19, 0.18]), ids, umbrales, n_galeria_total=500)
    assert d1.estado == EstadoIdentificacion.IDENTIFICADO
    assert d1.candidato_principal == "actor-a"

    # 2. Identificación dudosa por empate entre dos candidatos aunque s1 supere umbral_fiable
    d2 = clasificar_consulta(np.array([0.42, 0.405, 0.20, 0.19, 0.18]), ids, umbrales, n_galeria_total=500)
    assert d2.estado == EstadoIdentificacion.DUDOSO
    assert "actor-a" in d2.candidatos_dudosos and "actor-b" in d2.candidatos_dudosos
    assert d2.motivo == "empate_entre_candidatos"

    # 3. Voz no identificada (todos los candidatos con similitud baja y plana)
    d3 = clasificar_consulta(np.array([0.23, 0.22, 0.21, 0.20, 0.19]), ids, umbrales, n_galeria_total=500)
    assert d3.estado == EstadoIdentificacion.DESCONOCIDO
    assert d3.candidato_principal is None

    # 4. Búsqueda contextual con un único candidato (N=1): no debe aceptar una voz con similitud baja
    d4_rech = clasificar_consulta(np.array([0.20]), ["actor-unico"], umbrales, n_galeria_total=500, con_contexto=True)
    assert d4_rech.estado == EstadoIdentificacion.DESCONOCIDO

    d4_dud = clasificar_consulta(np.array([0.29]), ["actor-unico"], umbrales, n_galeria_total=500, con_contexto=True)
    assert d4_dud.estado == EstadoIdentificacion.DUDOSO

    d4_ok = clasificar_consulta(np.array([0.41]), ["actor-unico"], umbrales, n_galeria_total=500, con_contexto=True)
    assert d4_ok.estado == EstadoIdentificacion.IDENTIFICADO


def test_calibrar_umbrales_y_reduccion_falsos_positivos():
    """Verifica que `calibrar_umbrales` aprenda de validación y reduzca drásticamente el FPR en desconocidos."""
    rng = np.random.default_rng(123)
    ids = [f"actor-{i}" for i in range(40)]
    muestras: list[MuestraValidacionRechazo] = []

    # 60 consultas de actores conocidos (con pico claro en el actor real en la mayoría)
    for i in range(60):
        base = rng.normal(0.16, 0.05, size=len(ids)).clip(-0.1, 0.34)
        real_idx = i % len(ids)
        base[real_idx] = float(rng.uniform(0.38, 0.62)) if i % 6 != 0 else float(rng.uniform(0.27, 0.35))
        muestras.append(
            MuestraValidacionRechazo(
                puntuaciones=base,
                ids_candidatos=ids,
                actor_real=ids[real_idx],
                es_conocido=True,
                n_galeria_total=len(ids),
            )
        )

    # 40 consultas de voces desconocidas (ruido extremo entre 0.22 y 0.36, donde aPorcentaje da >= 50%)
    for _ in range(40):
        base = rng.normal(0.17, 0.055, size=len(ids)).clip(-0.1, 0.36)
        muestras.append(
            MuestraValidacionRechazo(
                puntuaciones=base,
                ids_candidatos=ids,
                actor_real=None,
                es_conocido=False,
                n_galeria_total=len(ids),
            )
        )

    umbrales = calibrar_umbrales(muestras, max_fpr_desconocidos=0.05, min_precision_conocidos=0.90)
    assert umbrales.umbral_fiable > umbrales.umbral_dudoso
    res = evaluar_rechazo(muestras, umbrales)

    assert res["sin_rechazo"]["fpr_desconocidas_pct"] == 100.0
    assert res["con_rechazo"]["desconocidas_falso_positivo_identificado_pct"] <= 5.0
    assert res["con_rechazo"]["precision_en_identificadas_conocidas_pct"] >= 90.0

    # Calibrador isotónico reduce el error de calibración (ECE / Brier) frente a aPorcentaje fijo
    cal = CalibradorConfianza().ajustar(muestras)
    metricas_cal = evaluar_calibracion_porcentajes(muestras, calibrador=cal)
    assert metricas_cal["calibrador_isotonico"]["brier"] <= metricas_cal["interfaz_actual_aPorcentaje"]["brier"]


def test_umbrales_calibrados_sincronizados_con_frontend_y_app():
    """Comprueba que `UMBRALES_VOXBLINK_R100_CALIBRADOS` está sincronizado con `frontend/index.html` y `backend/app.py`."""
    u = UMBRALES_VOXBLINK_R100_CALIBRADOS
    assert u.umbral_fiable > u.umbral_dudoso > 0.20
    assert u.margen_fiable > u.margen_dudoso > 0.0

    html = (RAIZ / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "function clasificarConsulta(" in html
    assert f"umbral_fiable: {u.umbral_fiable}" in html
    assert f"umbral_dudoso: {u.umbral_dudoso}" in html
    assert f"margen_fiable: {u.margen_fiable}" in html
    assert f"margen_dudoso: {u.margen_dudoso}" in html
    assert f"z_fiable: {u.z_fiable}" in html

    import app as backend_app
    if (RAIZ / "frontend" / "datos" / "actores.json").exists():
        backend_app.cargar_indice()
        estado_salud = backend_app.salud()
        assert estado_salud["ok"] is True
        assert estado_salud["modo_indice"] in ("web", "npz")
        q_prueba = np.ones((1, 256), dtype=np.float32) / np.sqrt(256.0)
        res = backend_app.puntuar(q_prueba, top=3)
        assert len(res) == 3
        assert "slug" in res[0] and "parecido" in res[0]
