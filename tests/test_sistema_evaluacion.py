"""Pruebas automatizadas del sistema reproducible de evaluación (Fase 2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from sistema_evaluacion import (  # noqa: E402
    ConfigEvaluacion,
    ConsultaEvaluacion,
    MuestraReferencia,
    cargar_grabaciones_reales,
    centroides_mezcla_web,
    comparar_versiones,
    cuantizar_descuantizar_int8,
    deducir_fuente_clip,
    ejecutar_evaluacion_completa,
    exportar_informe_json_csv,
    generar_datos_sinteticos_controlados,
    guardar_en_historial,
    id_grabacion_origen,
    puntuar_navegador,
)


def test_deduccion_fuente_y_agrupacion_antifuga():
    """Garantiza que clip limpio y #movil (o partes de un mismo episodio) comparten grupo_grabacion."""
    assert deducir_fuente_clip("108-jose-luis-gil/Muestras.mp3") == "eldoblaje"
    assert deducir_fuente_clip("108-jose-luis-gil/Muestras.mp3#movil") == "eldoblaje"
    assert deducir_fuente_clip("108-jose-luis-gil/voz_a1b2c3d4e5f6.flac") == "voces"
    assert deducir_fuente_clip("108-jose-luis-gil/podcast_0.wav") == "podcast"
    assert deducir_fuente_clip("108-jose-luis-gil/entrevista_1.wav") == "entrevista"

    assert id_grabacion_origen("actor/Muestras.mp3") == id_grabacion_origen("actor/Muestras.mp3#movil")
    assert id_grabacion_origen("actor/voz_a1b2c3d4e5f6_0.flac") == id_grabacion_origen("actor/voz_a1b2c3d4e5f6_1.flac")


def test_fidelidad_navegador_cuantizacion_centroides_y_puntuacion():
    """Comprueba que la cuantización int8, `centroides_mezcla_web` y `puntuar_navegador` replican la web."""
    rng = np.random.default_rng(7)
    V = rng.standard_normal((6, 32)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Vq = cuantizar_descuantizar_int8(V)
    # Similitud coseno > 0.999 tras cuantizar a int8 con escala por fila
    cos = np.sum(V * Vq, axis=1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(Vq, axis=1))
    assert np.all(cos > 0.999)

    y = np.array(["a1", "a1", "a1", "a2", "a2", "a2"])
    clips = np.array(["a1/Muestras.mp3", "a1/voz_1.flac", "a1/voz_2.flac", "a2/voz_1.flac", "a2/voz_2.flac", "a2/voz_3.flac"])
    C = centroides_mezcla_web(V, y, clips, ["a1", "a2"], peso_entrevistas=0.5)
    assert C.shape == (2, 32)
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)

    # Verificar cálculo exacto de (media + mejores_3) / 2
    q = V[:1]
    fa = np.array([0, 0, 0, 1, 1, 1])
    punt = puntuar_navegador(q, C, Vq, fa, peso_centroide=0.5, top_k_trozos=3)
    media_a0 = float(q[0] @ C[0])
    mejores_a0 = float(np.sort(Vq[:3] @ q[0])[::-1][:3].mean())
    assert np.isclose(punt[0], 0.5 * media_a0 + 0.5 * mejores_a0, atol=1e-5)


def test_permisos_grabaciones_reales(tmp_path: Path):
    """Rechaza grabaciones reales sin `permiso_verificado: true` y acepta las autorizadas."""
    manifiesto_mal = tmp_path / "sin_permiso.json"
    manifiesto_mal.write_text(
        json.dumps([{"id": "t1", "actor_real": "a1", "permiso_verificado": False, "huellas": [[1.0, 0.0]]}]),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="permiso_verificado"):
        cargar_grabaciones_reales(manifiesto_mal)

    manifiesto_ok = tmp_path / "con_permiso.json"
    manifiesto_ok.write_text(
        json.dumps([{"id": "t1", "actor_real": "a1", "permiso_verificado": True, "duracion_seg": 10.0, "huellas": [[3.0, 4.0]]}]),
        encoding="utf-8",
    )
    cargadas = cargar_grabaciones_reales(manifiesto_ok)
    assert len(cargadas) == 1
    assert cargadas[0].es_grabacion_real is True
    assert np.isclose(np.linalg.norm(cargadas[0].vectores[0]), 1.0)


def test_evaluacion_completa_informes_y_regresiones(tmp_path: Path):
    """Ejecuta el pipeline completo sobre datos sintéticos controlados y verifica JSON, CSV e historial."""
    refs, cons, repartos = generar_datos_sinteticos_controlados(
        n_actores=16,
        clips_por_actor=3,
        trozos_por_clip=3,
        dim=64,
        semilla=11,
    )
    cfg = ConfigEvaluacion(
        version_modelo="3",
        nombre_experimento="test_pipeline",
        pliegues_lda=4,
        semilla=11,
        proporcion_actores_desconocidos=0.2,
    )
    informe = ejecutar_evaluacion_completa(refs, cons, config=cfg, repartos_contexto=repartos)

    assert informe["resumen_galeria"]["num_actores_totales"] == 16
    assert informe["resumen_galeria"]["num_actores_desconocidos_retenidos"] == 3
    assert informe["metricas_globales"]["top3_pct"] >= informe["metricas_globales"]["top1_pct"] > 50.0
    assert informe["comparativa_contexto"]["con_contexto_top1_pct"] >= informe["metricas_globales"]["top1_pct"]
    assert set(informe["por_duracion"].keys()) == {"3s", "6s", "10s", "15s"}
    # A mayor duración (15s vs 3s), mayor o igual acierto
    assert informe["por_duracion"]["15s"]["top1_pct"] >= informe["por_duracion"]["3s"]["top1_pct"]

    rutas = exportar_informe_json_csv(informe, tmp_path / "salida")
    for p in rutas.values():
        assert p.exists() and p.stat().st_size > 0

    # Comprobar detector de regresiones e historial
    ruta_hist = tmp_path / "historial.jsonl"
    h1 = guardar_en_historial(informe, ruta_hist)
    assert h1["comparativa_con_anterior"] is None

    informe_peor = json.loads(json.dumps(informe))
    informe_peor["configuracion"]["nombre_experimento"] = "test_degradado"
    informe_peor["metricas_globales"]["top1_pct"] -= 5.0
    h2 = guardar_en_historial(informe_peor, ruta_hist, tolerancia_caida_pct=0.5)
    assert h2["comparativa_con_anterior"]["hay_regresion"] is True
    assert any("Top-1" in r for r in h2["comparativa_con_anterior"]["regresiones_detectadas"])


def test_filtrar_trozos_intrusos_y_blindaje_respuestas(tmp_path: Path):
    """Verifica la poda de trozos intrusos (`auditar_indice.py`) y el blindaje anti-envenenamiento en `exportar_web.py`."""
    from auditar_indice import filtrar_trozos_intrusos
    from lda import normaliza

    sys.path.insert(0, str(RAIZ / "despliegue"))
    from exportar_web import centroides_mezcla, respuestas_validas

    dim = 16
    v_a = normaliza(np.eye(1, dim, 0, dtype=np.float32))[0]
    v_b = normaliza(np.eye(1, dim, 1, dtype=np.float32))[0]
    # 5 trozos legítimos de actor-a y 1 trozo intruso (idéntico a actor-b, ortogonal a actor-a)
    Xp = np.vstack([v_a, v_a, v_a, v_a, v_a, v_b, v_b, v_b, v_b, v_b])
    y = np.array(["actor-a"] * 6 + ["actor-b"] * 4)
    C = np.vstack([v_a, v_b])
    conservar, intrusos = filtrar_trozos_intrusos(Xp, y, C, ["actor-a", "actor-b"], umbral_outlier=0.15, min_trozos_por_actor=4)
    assert len(intrusos) == 1
    assert intrusos[0]["actor_asignado"] == "actor-a"
    assert intrusos[0]["actor_mas_parecido"] == "actor-b"
    assert bool(conservar[5]) is False

    # Blindaje de respuestas_validas contra NaN, duplicados, vectores nulos y baja similitud
    mu = np.zeros(dim, dtype=np.float32)
    W = np.eye(dim, dtype=np.float32)
    ruta_resp = tmp_path / "respuestas.json"
    respuestas_mock = [
        {"version": "3", "propuesto": "actor-a", "acierto": True, "huellas": [v_a.tolist()]},
        {"version": "3", "propuesto": "actor-a", "acierto": True, "huellas": [v_a.tolist()]},  # duplicada exacta
        {"version": "3", "propuesto": "actor-a", "acierto": True, "huellas": [[0.0] * dim]},   # vector nulo
        {"version": "3", "propuesto": "actor-a", "acierto": True, "huellas": [[float("nan")] * dim]},  # NaN
        {"version": "3", "propuesto": "actor-a", "acierto": True, "huellas": [v_b.tolist()]},  # impostor (sim 0)
    ]
    ruta_resp.write_text(json.dumps(respuestas_mock), encoding="utf-8")
    Xr, yr = respuestas_validas(mu, W, C, ["actor-a", "actor-b"], ruta_respuestas=ruta_resp)
    assert len(Xr) == 1 and yr == ["actor-a"]

    # centroides_mezcla pondera respuestas a 0.35 sin desplazar el estudio
    C_mix = centroides_mezcla(
        np.vstack([v_a, v_b]),
        np.array(["actor-a", "actor-a"]),
        np.array(["actor-a/Muestras.mp3", "respuesta"]),
        ["actor-a"],
        peso_respuestas=0.35,
    )
    assert float(C_mix[0] @ v_a) > float(C_mix[0] @ v_b)
