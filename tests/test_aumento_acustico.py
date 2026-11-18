"""Tests para el módulo de aumento acústico, VAD y segmentación de intervenciones."""
import sys
from pathlib import Path
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from aumento_acustico import (
    DetectorVoz,
    ResultadoVAD,
    aislar_voz_continua,
    generar_rir_salon,
    segmentar_intervenciones,
    simular_sala_tv,
)


def _generar_senal_voz_sintetica(sr: int = 16000, duracion_seg: float = 3.0) -> np.ndarray:
    """Genera una señal pseudo-vocal con fundamental F0=150 Hz y formantes armónicos."""
    t = np.arange(int(sr * duracion_seg), dtype=np.float32) / sr
    f0 = 150.0
    voz = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 5 * f0 * t)
    ).astype(np.float32)
    # Modulación de amplitud para simular sílabas a ~4 Hz
    modulacion = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    return (voz * modulacion).astype(np.float32)


def test_vad_detecta_silencio():
    sr = 16000
    silencio = np.zeros(sr * 4, dtype=np.float32)
    detector = DetectorVoz(sr=sr)
    vad = detector.analizar(silencio)

    assert isinstance(vad, ResultadoVAD)
    assert not vad.es_toma_valida
    assert vad.duracion_voz_seg < 0.1
    assert len(vad.segmentos) == 0


def test_vad_detecta_voz_activa():
    sr = 16000
    voz = _generar_senal_voz_sintetica(sr=sr, duracion_seg=4.0)
    # Silencio inicial de 1 s + 4 s de voz + 1 s de silencio final
    audio = np.concatenate([np.zeros(sr, dtype=np.float32), voz, np.zeros(sr, dtype=np.float32)])

    detector = DetectorVoz(sr=sr, min_voz_util_seg=1.5)
    vad = detector.analizar(audio)

    assert vad.es_toma_valida
    assert vad.duracion_voz_seg >= 2.5
    assert len(vad.segmentos) >= 1
    # El primer segmento debe comenzar cerca de 1.0 s
    assert abs(vad.segmentos[0].inicio_seg - 1.0) < 0.4


def test_aislar_voz_continua():
    sr = 16000
    voz = _generar_senal_voz_sintetica(sr=sr, duracion_seg=3.0)
    # 2 s silencio + 3 s voz + 2 s silencio
    audio = np.concatenate([np.zeros(sr * 2, dtype=np.float32), voz, np.zeros(sr * 2, dtype=np.float32)])

    voz_limpia = aislar_voz_continua(audio, sr=sr)
    # Debe eliminar la mayor parte de los 4 s de silencio
    assert len(voz_limpia) < len(audio)
    assert len(voz_limpia) >= int(sr * 2.0)


def test_segmentar_intervenciones_multi_hablante():
    sr = 16000
    hablante_1 = _generar_senal_voz_sintetica(sr=sr, duracion_seg=3.0)
    pausa = np.zeros(sr * 1, dtype=np.float32)  # Pausa de 1 s entre turnos
    hablante_2 = _generar_senal_voz_sintetica(sr=sr, duracion_seg=3.0)

    audio = np.concatenate([hablante_1, pausa, hablante_2])
    detector = DetectorVoz(sr=sr, min_pausa_separacion_ms=400.0)
    vad = detector.analizar(audio)
    turnos = segmentar_intervenciones(audio, vad=vad, sr=sr, min_duracion_turno_seg=2.0)

    # Debe haber identificado los dos turnos separados
    assert len(turnos) == 2
    for t in turnos:
        assert len(t) >= int(sr * 2.0)


def test_generar_rir_salon():
    sr = 16000
    t60 = 0.35
    rir = generar_rir_salon(sr=sr, t60_seg=t60, distancia_m=2.0, semilla=42)

    assert isinstance(rir, np.ndarray)
    assert rir.dtype == np.float32
    assert len(rir) > int(sr * t60)
    assert not np.isnan(rir).any()
    assert not np.isinf(rir).any()
    # Norma unitaria
    assert np.isclose(np.linalg.norm(rir), 1.0, atol=1e-4)


def test_simular_sala_tv():
    sr = 16000
    audio = _generar_senal_voz_sintetica(sr=sr, duracion_seg=2.0)
    audio_sala = simular_sala_tv(audio, sr=sr, t60_seg=0.3, distancia_m=2.5, semilla=123)

    assert len(audio_sala) == len(audio)
    assert audio_sala.dtype == np.float32
    assert not np.isnan(audio_sala).any()
    # La reverberación y el filtrado deben modificar la forma de onda
    assert not np.array_equal(audio, audio_sala)
    # Energía relativa conservada aproximadamente (en rango comparable)
    rms_orig = np.sqrt(np.mean(audio * audio))
    rms_sala = np.sqrt(np.mean(audio_sala * audio_sala))
    assert np.isclose(rms_orig, rms_sala, rtol=0.25)


def test_lda_con_covarianza_acustica():
    from lda import entrenar_lda, estimar_covarianza_canal, proyectar, normaliza

    rng = np.random.default_rng(123)
    dim = 16
    n_actores = 4
    n_muestras_por_actor = 10

    # Generar vectores sintéticos para 4 actores
    centroides_verdaderos = rng.normal(0, 1, (n_actores, dim))
    X_limpio = []
    y = []
    for a in range(n_actores):
        ruido = rng.normal(0, 0.2, (n_muestras_por_actor, dim))
        X_limpio.append(centroides_verdaderos[a] + ruido)
        y.extend([f"actor_{a}"] * n_muestras_por_actor)
    X_limpio = normaliza(np.vstack(X_limpio))
    y = np.array(y)

    # Simular distorsión acústica de canal (añadir sesgo en ciertas dimensiones)
    distorsion_canal = rng.normal(0, 0.4, (len(X_limpio), dim))
    X_aumentado = normaliza(X_limpio + distorsion_canal)

    # Estimar covarianza de canal
    C_canal = estimar_covarianza_canal(X_limpio, X_aumentado)
    assert C_canal.shape == (dim, dim)

    # Entrenar LDA estándar y LDA con regularización acústica
    mu_std, W_std = entrenar_lda(X_limpio, y, dims=8)
    mu_ac, W_ac = entrenar_lda(X_limpio, y, dims=8, cov_acustica=C_canal, peso_cov_acustica=0.15)

    assert W_std.shape == (dim, 8)
    assert W_ac.shape == (dim, 8)

    # Proyección con la matriz acústica debe producir representaciones normalizadas válidas
    p_ac = proyectar(X_aumentado, mu_ac, W_ac)
    assert p_ac.shape == (len(X_limpio), 8)
    assert np.allclose(np.linalg.norm(p_ac, axis=1), 1.0, atol=1e-5)


def test_estimar_f0_y_registro_vocal():
    from aumento_acustico import (
        clasificar_registro_vocal,
        estimar_f0_mediana,
        normalizar_amplitud_pico,
    )

    sr = 16000
    dur = 1.0
    t = np.arange(int(sr * dur), dtype=np.float32) / sr

    # 1. Señal grave F0 = 110 Hz (voz masculina grave) con armónicos
    f0_grave = 110.0
    sig_grave = (
        0.6 * np.sin(2 * np.pi * f0_grave * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0_grave * t)
        + 0.1 * np.sin(2 * np.pi * 3 * f0_grave * t)
    ).astype(np.float32)

    # 2. Señal aguda F0 = 220 Hz (voz femenina/aguda) con armónicos
    f0_aguda = 220.0
    sig_aguda = (
        0.6 * np.sin(2 * np.pi * f0_aguda * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0_aguda * t)
        + 0.1 * np.sin(2 * np.pi * 3 * f0_aguda * t)
    ).astype(np.float32)

    # Estimación de tono
    est_grave = estimar_f0_mediana(sig_grave, sr=sr)
    est_aguda = estimar_f0_mediana(sig_aguda, sr=sr)

    assert est_grave is not None
    assert est_aguda is not None
    assert abs(est_grave - f0_grave) < 5.0
    assert abs(est_aguda - f0_aguda) < 5.0

    # Clasificación de registro
    assert clasificar_registro_vocal(est_grave) == "grave"
    assert clasificar_registro_vocal(est_aguda) == "agudo"
    assert clasificar_registro_vocal(150.0) == "medio"
    assert clasificar_registro_vocal(None) == "desconocido"

    # Normalización de pico estándar y rescate frente a pico impulsivo aislado
    sig_norm = normalizar_amplitud_pico(sig_grave, pico_objetivo=0.75)
    assert np.isclose(np.max(np.abs(sig_norm)), 0.75, atol=1e-3)

    sig_debil = (0.08 * sig_grave / max(float(np.max(np.abs(sig_grave))), 1e-6)).astype(np.float32)
    rms_lineal = float(np.sqrt(np.mean((sig_debil * 0.75) ** 2)))
    sig_con_clic = sig_debil.copy()
    sig_con_clic[100] = 1.0
    sig_rescatada = normalizar_amplitud_pico(sig_con_clic, pico_objetivo=0.75)
    assert np.isclose(np.max(np.abs(sig_rescatada)), 0.75, atol=1e-3)
    rms_rescatada = float(np.sqrt(np.mean(np.delete(sig_rescatada, 100) ** 2)))
    assert rms_rescatada >= 2.5 * rms_lineal


def test_vad_espectral_acepta_voz_continua_y_rechaza_ruido_o_zumbido():
    sr = 16000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    detector = DetectorVoz(sr=sr, min_voz_util_seg=1.5)

    # Voz continua sin pausas largas: p10 cercano a p50 pero banda vocal alta
    voz_continua = (
        0.35
        * (0.7 + 0.3 * np.sin(2 * np.pi * 4.0 * t))
        * (0.6 * np.sin(2 * np.pi * 150.0 * t) + 0.4 * np.sin(2 * np.pi * 300.0 * t))
    ).astype(np.float32)
    vad_cont = detector.analizar(voz_continua)
    assert vad_cont.es_toma_valida
    assert vad_cont.duracion_voz_seg >= 2.2

    # Ruido blanco de banda ancha: energía dispersa por encima de la banda vocal
    rng = np.random.default_rng(7)
    ruido_blanco = (0.2 * rng.standard_normal(sr * 3)).astype(np.float32)
    vad_ruido = detector.analizar(ruido_blanco)
    assert not vad_ruido.es_toma_valida

    # Zumbido subgrave de 45 Hz fuera de la banda vocal
    zumbido = (0.4 * np.sin(2 * np.pi * 45.0 * t)).astype(np.float32)
    vad_zumbido = detector.analizar(zumbido)
    assert not vad_zumbido.es_toma_valida


def test_lda_balancear_clases_limita_dominio_mayoritaria():
    from lda import entrenar_lda, proyectar

    rng = np.random.default_rng(123)
    D = 12

    # Clase 0 muy sobrerrepresentada (200 muestras) con dispersión moderada en e1;
    # sin balanceo infla Sw[1,1] y eclipsa la separación entre clases 1 y 2 en dims=2
    x0 = np.zeros((200, D), dtype=np.float32)
    x0[:, 0] = 1.0
    x0[:, 1] = 0.20 * rng.standard_normal(200).astype(np.float32)
    x0 += 0.03 * rng.standard_normal((200, D)).astype(np.float32)

    x1 = np.zeros((5, D), dtype=np.float32)
    x1[:, 1] = 1.0
    x1 += 0.02 * rng.standard_normal((5, D)).astype(np.float32)

    x2 = np.zeros((5, D), dtype=np.float32)
    x2[:, 1] = -1.0
    x2 += 0.02 * rng.standard_normal((5, D)).astype(np.float32)

    x3 = np.zeros((5, D), dtype=np.float32)
    x3[:, 0] = -0.7
    x3[:, 2] = 0.7 + 0.25 * rng.standard_normal(5).astype(np.float32)
    x3 += 0.02 * rng.standard_normal((5, D)).astype(np.float32)

    X = np.vstack([x0, x1, x2, x3])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array([0] * 200 + [1] * 5 + [2] * 5 + [3] * 5)

    mu_bal, W_bal = entrenar_lda(X, y, dims=2, reg=1e-4, balancear_clases=True, max_peso_clase=5)
    mu_raw, W_raw = entrenar_lda(X, y, dims=2, reg=1e-4, balancear_clases=False)

    assert mu_bal.shape == (D,) and W_bal.shape == (D, 2)
    assert np.all(np.isfinite(mu_bal)) and np.all(np.isfinite(W_bal))
    assert not np.allclose(W_bal, W_raw)

    def _dist_c1_c2(mu_v, W_v):
        p1 = proyectar(X[y == 1], mu_v, W_v).mean(0)
        p2 = proyectar(X[y == 2], mu_v, W_v).mean(0)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        return float(1.0 - p1 @ p2)

    # Con balanceo de clases la discriminación entre clases minoritarias no queda aplastada
    assert _dist_c1_c2(mu_bal, W_bal) > _dist_c1_c2(mu_raw, W_raw)


def test_realzar_voz_lejana_y_atenuar_fondo():
    from aumento_acustico import realzar_voz_lejana_y_atenuar_fondo

    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    # Señal muy apagada en agudos (baja frecuencia dominante -> r1 > 0.85) con tramo de siseo débil
    voz_apagada = (0.5 * np.sin(2 * np.pi * 140.0 * t) + 0.05 * np.sin(2 * np.pi * 1800.0 * t)).astype(np.float32)
    # Silencio con pequeño siseo de alta frecuencia en el último medio segundo
    voz_apagada[int(sr * 1.5) :] = (0.01 * np.sin(2 * np.pi * 6000.0 * t[: int(sr * 0.5)])).astype(np.float32)

    procesada = realzar_voz_lejana_y_atenuar_fondo(voz_apagada, sr=sr)
    assert procesada.shape == voz_apagada.shape
    assert np.all(np.isfinite(procesada))

    # El tramo de cola no vocal debe quedar atenuado respecto a la entrada
    rms_cola_in = float(np.sqrt(np.mean(voz_apagada[int(sr * 1.6) :] ** 2)))
    rms_cola_out = float(np.sqrt(np.mean(procesada[int(sr * 1.6) :] ** 2)))
    assert rms_cola_out < 0.75 * rms_cola_in


def test_lda_tikhonov_shrinkage_generaliza_en_pocas_muestras():
    import inspect
    from lda import entrenar_lda, normaliza, proyectar

    assert inspect.signature(entrenar_lda).parameters["reg"].default == 1.25

    rng = np.random.default_rng(77)
    D = 24
    # Pocas clases de entrenamiento con varianza intra-clase subestimada en ejes 0..3
    # (p. ej. canal limpio en entrenamiento) y alta en ejes 4..11: sin shrinkage suficiente
    # (reg=1e-4), LDA sobrepondera los ejes 0..3 y se vuelve frágil ante ruido en prueba.
    n_clases_tr = 12
    X_tr_l, y_tr_l = [], []
    for c in range(n_clases_tr):
        base = np.zeros(D, dtype=np.float32)
        base[c % 12] = 1.2
        base[(c + 4) % 12] = 1.0
        for _ in range(4):
            ruido = np.zeros(D, dtype=np.float32)
            ruido[:4] = 0.005 * rng.standard_normal(4).astype(np.float32)
            ruido[4:] = 0.25 * rng.standard_normal(D - 4).astype(np.float32)
            X_tr_l.append(base + ruido)
            y_tr_l.append(c)
    X_tr = normaliza(np.stack(X_tr_l))
    y_tr = np.array(y_tr_l)

    mu_def, W_def = entrenar_lda(X_tr, y_tr)
    mu_baja, W_baja = entrenar_lda(X_tr, y_tr, reg=1e-4)
    assert W_def.shape == (D, D)

    # Hablantes nuevos (fuera de entrenamiento) con ruido realista también en los ejes 0..3
    cA = np.zeros(D, dtype=np.float32)
    cA[4:10] = [1.0, -0.8, 0.7, 0.6, -0.5, 0.4]
    cA /= np.linalg.norm(cA)
    cB = np.zeros(D, dtype=np.float32)
    cB[4:10] = [-0.8, 0.9, -0.6, 0.7, 0.5, -0.4]
    cB /= np.linalg.norm(cB)

    def tomar_muestras(c, n=12):
        ruido = rng.normal(0.0, 0.18, size=(n, D)).astype(np.float32)
        return normaliza(c[None, :] + ruido)

    qA1 = tomar_muestras(cA, 8)
    qA2 = tomar_muestras(cA, 8)
    qB = tomar_muestras(cB, 8)

    P_def_A1 = proyectar(qA1, mu_def, W_def)
    P_def_A2 = proyectar(qA2, mu_def, W_def)
    P_def_B = proyectar(qB, mu_def, W_def)
    P_baja_A1 = proyectar(qA1, mu_baja, W_baja)
    P_baja_A2 = proyectar(qA2, mu_baja, W_baja)
    P_baja_B = proyectar(qB, mu_baja, W_baja)

    margen_def = float(np.mean(np.sum(P_def_A1 * P_def_A2, axis=1)) - np.mean(np.sum(P_def_A1 * P_def_B, axis=1)))
    margen_baja = float(np.mean(np.sum(P_baja_A1 * P_baja_A2, axis=1)) - np.mean(np.sum(P_baja_A1 * P_baja_B, axis=1)))
    assert margen_def > margen_baja + 0.10
