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

    # Normalización de pico
    sig_norm = normalizar_amplitud_pico(sig_grave, pico_objetivo=0.75)
    assert np.isclose(np.max(np.abs(sig_norm)), 0.75, atol=1e-3)


