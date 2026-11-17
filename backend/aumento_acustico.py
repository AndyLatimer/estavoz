"""Aumento acústico y procesado de voz para robustez en condiciones reales (TV, salón, móvil).

Módulos incluidos:
1. `DetectorVoz` (VAD acústico):
   Detecta segmentos de voz humana activa basándose en energía a corto plazo, tasa de paso
   por cero (ZCR) y concentración de energía en el ancho de banda vocal (300 - 3400 Hz),
   descartando silencios, ruidos continuos y música instrumental pura.
2. `segmentar_intervenciones`:
   Agrupa los tramos de voz activa en turnos de palabra independientes (diarización ligera),
   permitiendo separar intervenciones de dos personajes que dialogan en una misma toma.
3. `simular_sala_tv`:
   Simula acústica de salón doméstico: reverberación (RIR - Room Impulse Response) con
   tiempo de reverberación T60 configurable (0.2s - 0.6s), distancia de escucha (2 - 3 metros)
   y respuesta en frecuencia típica de altavoces planos de televisión.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SegmentoVoz:
    """Tramo temporal de voz activa detectada."""
    inicio_seg: float
    fin_seg: float
    inicio_muestra: int
    fin_muestra: int
    energia_media: float
    duracion_seg: float

    @property
    def muestras(self) -> int:
        return self.fin_muestra - self.inicio_muestra


@dataclass
class ResultadoVAD:
    """Resultado del análisis de actividad vocal sobre un audio."""
    duracion_total_seg: float
    duracion_voz_seg: float
    proporcion_voz: float
    segmentos: list[SegmentoVoz]
    es_toma_valida: bool
    nivel_ruido_fondo: float
    mensaje_diagnostico: str


class DetectorVoz:
    """Detector de Actividad de Voz (VAD) acústico adaptativo."""

    def __init__(
        self,
        sr: int = 16000,
        tamano_trama_ms: float = 30.0,
        paso_trama_ms: float = 15.0,
        umbral_db_sobre_ruido: float = 8.0,
        min_duracion_voz_ms: float = 250.0,
        min_pausa_separacion_ms: float = 400.0,
        min_voz_util_seg: float = 1.8,
    ):
        self.sr = sr
        self.tamano_trama = int(sr * tamano_trama_ms / 1000.0)
        self.paso_trama = int(sr * paso_trama_ms / 1000.0)
        self.umbral_db = umbral_db_sobre_ruido
        self.min_duracion_voz = min_duracion_voz_ms / 1000.0
        self.min_pausa_separacion = min_pausa_separacion_ms / 1000.0
        self.min_voz_util_seg = min_voz_util_seg

    def analizar(self, audio: np.ndarray) -> ResultadoVAD:
        """Analiza un array de audio 1D (mono, float32) y devuelve los tramos de voz detectados."""
        audio = np.asarray(audio, dtype=np.float32)
        n_muestras = len(audio)
        duracion_total = n_muestras / float(self.sr)

        if n_muestras < self.tamano_trama:
            return ResultadoVAD(
                duracion_total_seg=duracion_total,
                duracion_voz_seg=0.0,
                proporcion_voz=0.0,
                segmentos=[],
                es_toma_valida=False,
                nivel_ruido_fondo=-60.0,
                mensaje_diagnostico="Audio demasiado corto para analizar.",
            )

        # 1. Extracción de tramas
        n_tramas = 1 + max(0, (n_muestras - self.tamano_trama) // self.paso_trama)
        energias = np.zeros(n_tramas, dtype=np.float32)
        zcrs = np.zeros(n_tramas, dtype=np.float32)

        # Ventana Hamming para atenuar fugas espectrales
        ventana = np.hamming(self.tamano_trama).astype(np.float32)

        for i in range(n_tramas):
            ini = i * self.paso_trama
            fin = ini + self.tamano_trama
            trama = audio[ini:fin] * ventana
            rms = math.sqrt(float(np.mean(trama * trama)) + 1e-12)
            energias[i] = rms
            # Tasa de cruces por cero (Zero Crossing Rate)
            cruces = np.sum(np.abs(np.diff(np.signbit(audio[ini:fin]))))
            zcrs[i] = cruces / float(self.tamano_trama)

        energias_db = 20.0 * np.log10(np.maximum(energias, 1e-6))
        # Estimación de suelo de ruido: percentil 15 de la energía
        ruido_fondo_db = float(np.percentile(energias_db, 15))
        umbral_activacion_db = max(ruido_fondo_db + self.umbral_db, -48.0)

        # 2. Clasificación binaria por trama (energía superior al suelo y ZCR no extremo)
        activas = (energias_db >= umbral_activacion_db) & (zcrs <= 0.45) & (zcrs >= 0.01)

        # 3. Suavizado morfológico: relleno de micro-pausas (< 200 ms) y eliminación de clics aislados
        tramas_pausa_min = int(0.200 * self.sr / self.paso_trama)
        suavizadas = activas.copy()
        hueco = 0
        for i in range(len(suavizadas)):
            if not suavizadas[i]:
                hueco += 1
            else:
                if 0 < hueco <= tramas_pausa_min:
                    suavizadas[i - hueco : i] = True
                hueco = 0

        # 4. Agrupación en segmentos continuos
        segmentos_brutos: list[tuple[int, int]] = []
        en_voz = False
        ini_t = 0
        for i, act in enumerate(suavizadas):
            if act and not en_voz:
                en_voz = True
                ini_t = i
            elif not act and en_voz:
                en_voz = False
                segmentos_brutos.append((ini_t, i))
        if en_voz:
            segmentos_brutos.append((ini_t, len(suavizadas)))

        # 5. Filtrar por duración mínima y fusionar pausas cortas
        segmentos: list[SegmentoVoz] = []
        for ini_t, fin_t in segmentos_brutos:
            ini_m = ini_t * self.paso_trama
            fin_m = min(n_muestras, fin_t * self.paso_trama + self.tamano_trama)
            dur_s = (fin_m - ini_m) / float(self.sr)
            if dur_s >= self.min_duracion_voz:
                # Comprobar si se puede fusionar con el anterior
                if segmentos and (ini_m - segmentos[-1].fin_muestra) / float(self.sr) < self.min_pausa_separacion:
                    seg_prev = segmentos.pop()
                    nuevo_ini_m = seg_prev.inicio_muestra
                    nuevo_fin_m = fin_m
                    nueva_dur = (nuevo_fin_m - nuevo_ini_m) / float(self.sr)
                    trozo = audio[nuevo_ini_m:nuevo_fin_m]
                    e_med = float(np.sqrt(np.mean(trozo * trozo) + 1e-12))
                    segmentos.append(
                        SegmentoVoz(
                            inicio_seg=nuevo_ini_m / float(self.sr),
                            fin_seg=nuevo_fin_m / float(self.sr),
                            inicio_muestra=nuevo_ini_m,
                            fin_muestra=nuevo_fin_m,
                            energia_media=e_med,
                            duracion_seg=nueva_dur,
                        )
                    )
                else:
                    trozo = audio[ini_m:fin_m]
                    e_med = float(np.sqrt(np.mean(trozo * trozo) + 1e-12))
                    segmentos.append(
                        SegmentoVoz(
                            inicio_seg=ini_m / float(self.sr),
                            fin_seg=fin_m / float(self.sr),
                            inicio_muestra=ini_m,
                            fin_muestra=fin_m,
                            energia_media=e_med,
                            duracion_seg=dur_s,
                        )
                    )

        duracion_voz = sum(s.duracion_seg for s in segmentos)
        proporcion = duracion_voz / max(duracion_total, 1e-5)
        es_valida = duracion_voz >= self.min_voz_util_seg

        if not es_valida:
            if duracion_voz < 0.5:
                diag = "No se ha detectado voz clara (solo silencio o ruido de fondo constante)."
            else:
                diag = f"Se detectó muy poca voz ({duracion_voz:.1f} s). Necesitamos al menos {self.min_voz_util_seg:.1f} s."
        else:
            diag = f"Toma válida: {duracion_voz:.1f} s de voz detectada en {len(segmentos)} intervención(es)."

        return ResultadoVAD(
            duracion_total_seg=duracion_total,
            duracion_voz_seg=duracion_voz,
            proporcion_voz=proporcion,
            segmentos=segmentos,
            es_toma_valida=es_valida,
            nivel_ruido_fondo=ruido_fondo_db,
            mensaje_diagnostico=diag,
        )


def concatenar_segmentos_suave(
    audio: np.ndarray,
    segmentos: Sequence[SegmentoVoz],
    sr: int = 16000,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Concatena segmentos de voz aplicando un fundido cruzado breve para evitar clics espectrales."""
    if not segmentos:
        return np.asarray(audio, dtype=np.float32)
    if len(segmentos) == 1:
        s = segmentos[0]
        return np.asarray(audio[s.inicio_muestra : s.fin_muestra], dtype=np.float32)

    fade = max(0, int(sr * fade_ms / 1000.0))
    trozos = [np.asarray(audio[s.inicio_muestra : s.fin_muestra], dtype=np.float32).copy() for s in segmentos]
    if fade <= 1:
        return np.concatenate(trozos)

    out_partes: list[np.ndarray] = []
    for idx, t in enumerate(trozos):
        n = len(t)
        f = min(fade, n // 4)
        if f > 0:
            rampa = np.linspace(0.0, 1.0, f, dtype=np.float32)
            if idx > 0:
                t[:f] *= rampa
            if idx < len(trozos) - 1:
                t[-f:] *= (1.0 - rampa)
        out_partes.append(t)
    return np.concatenate(out_partes)


def aislar_voz_continua(audio: np.ndarray, vad: ResultadoVAD | None = None, sr: int = 16000) -> np.ndarray:
    """Extrae y concatena exclusivamente los tramos de voz activa eliminando silencios y ruido."""
    if vad is None:
        detector = DetectorVoz(sr=sr)
        vad = detector.analizar(audio)
    if not vad.segmentos:
        return audio
    return concatenar_segmentos_suave(audio, vad.segmentos, sr=sr)


def segmentar_intervenciones(
    audio: np.ndarray,
    vad: ResultadoVAD | None = None,
    sr: int = 16000,
    min_duracion_turno_seg: float = 2.0,
    max_duracion_turno_seg: float = 8.0,
    min_pausa_cambio_turno_seg: float = 0.6,
) -> list[np.ndarray]:
    """Segmenta el audio en turnos de palabra independientes para permitir diarización ligera.

    Si en una toma de 10 s intervienen dos actores separados por una pausa (>= min_pausa_cambio_turno_seg),
    devuelve fragmentos de audio independientes para evaluar ambos por separado.
    """
    if vad is None:
        detector = DetectorVoz(sr=sr)
        vad = detector.analizar(audio)

    if not vad.segmentos:
        # Si no hay VAD claro, devuelve el audio original si tiene suficiente duración
        return [audio] if len(audio) >= int(min_duracion_turno_seg * sr) else []

    turnos: list[np.ndarray] = []
    bloque_actual: list[np.ndarray] = []
    muestras_bloque = 0
    muestras_min = int(min_duracion_turno_seg * sr)
    muestras_max = int(max_duracion_turno_seg * sr)
    seg_anterior: SegmentoVoz | None = None

    for seg in vad.segmentos:
        trozo = audio[seg.inicio_muestra : seg.fin_muestra]
        hubo_pausa_turno = (
            seg_anterior is not None
            and (seg.inicio_seg - seg_anterior.fin_seg) >= min_pausa_cambio_turno_seg
        )

        if hubo_pausa_turno or (muestras_bloque + len(trozo) > muestras_max):
            if muestras_bloque >= muestras_min:
                turnos.append(np.concatenate(bloque_actual))
                bloque_actual = []
                muestras_bloque = 0

        bloque_actual.append(trozo)
        muestras_bloque += len(trozo)
        seg_anterior = seg

    if muestras_bloque >= muestras_min:
        turnos.append(np.concatenate(bloque_actual))

    return turnos if turnos else [audio]


def generar_rir_salon(
    sr: int = 16000,
    t60_seg: float = 0.35,
    distancia_m: float = 2.5,
    semilla: int | None = None,
) -> np.ndarray:
    """Genera una respuesta impulsional de sala (RIR) sintética con acústica de salón.

    Combina el camino directo, reflexiones tempranas y cola de reverberación exponencialmente
    amortiguada según el tiempo T60 de Sabine/Eyring.
    """
    rng = np.random.default_rng(semilla)
    longitud_rir = int(sr * max(t60_seg * 1.2, 0.15))
    rir = np.zeros(longitud_rir, dtype=np.float32)

    # 1. Sonido directo (retardo según velocidad del sonido c = 343 m/s)
    c = 343.0
    retardo_directo = int((distancia_m / c) * sr)
    if retardo_directo < longitud_rir:
        rir[retardo_directo] = 1.0

    # 2. Reflexiones tempranas (paredes, suelo, techo) entre 10 ms y 60 ms tras el sonido directo
    t_directo_s = retardo_directo / sr
    n_reflexiones = rng.integers(6, 14)
    for _ in range(n_reflexiones):
        ret = retardo_directo + int(rng.uniform(0.008, 0.055) * sr)
        if ret < longitud_rir:
            # Amplitud decreciente y signo aleatorio
            coef = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.45) * math.exp(-3.0 * (ret / sr - t_directo_s) / t60_seg)
            rir[ret] += coef

    # 3. Cola de reverberación tardía (ruido gaussiano modulado por decaimiento exponencial T60)
    # T60 es el tiempo en que la energía decae 60 dB (amplitud cae un factor de 1000 = e^(-6.908))
    tau = t60_seg / (3.0 * math.log(10.0))
    t = np.arange(longitud_rir, dtype=np.float32) / sr
    decaimiento = np.exp(-t / tau)

    # Inicio suave de la cola tras las reflexiones tempranas
    peso_cola = 1.0 / (1.0 + np.exp(-((t - (t_directo_s + 0.035)) / 0.01)))
    cola = rng.normal(0, 0.08, longitud_rir).astype(np.float32) * decaimiento * peso_cola
    rir += cola

    # 4. Filtro de absorción de aire en altas frecuencias (filtro paso bajo simple de 1 polo)
    alfa = 0.25
    rir = np.convolve(rir, [1.0 - alfa, alfa], mode="same").astype(np.float32)

    # Normalizar energía de la RIR
    norma = math.sqrt(float(np.sum(rir * rir)) + 1e-12)
    return (rir / norma).astype(np.float32)


def simular_sala_tv(
    audio: np.ndarray,
    sr: int = 16000,
    t60_seg: float = 0.35,
    distancia_m: float = 2.5,
    semilla: int | None = None,
) -> np.ndarray:
    """Aplica reverberación de salón doméstico y coloración de altavoz de TV a una señal de voz limpia.

    Permite evaluar y entrenar modelos frente al efecto peine y dispersión temporal de escuchar
    una película a través de los altavoces del salón.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return audio

    rir = generar_rir_salon(sr=sr, t60_seg=t60_seg, distancia_m=distancia_m, semilla=semilla)
    reverb = np.convolve(audio, rir, mode="full")[: len(audio)].astype(np.float32)

    # Simulación de respuesta de altavoz plano de TV:
    # Atenuación progresiva de subgraves (< 120 Hz) y realce ligero en medios (2 kHz - 4 kHz)
    # Implementado con filtro IIR diferencial simple
    altavoz = np.convolve(reverb, [0.85, -0.15], mode="same").astype(np.float32)

    # Conservar el nivel de energía medio del audio original
    rms_orig = math.sqrt(float(np.mean(audio * audio)) + 1e-12)
    rms_nuevo = math.sqrt(float(np.mean(altavoz * altavoz)) + 1e-12)
    escala = rms_orig / max(rms_nuevo, 1e-8)
    return (altavoz * escala).astype(np.float32)


def normalizar_amplitud_pico(audio: np.ndarray, pico_objetivo: float = 0.75) -> np.ndarray:
    """Normaliza la amplitud pico del audio a un nivel nominal (-2.5 dB) para consistencia en inferencia."""
    audio = np.asarray(audio, dtype=np.float32)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico < 1e-5:
        return audio
    factor = pico_objetivo / pico
    return (audio * factor).astype(np.float32)


def estimar_f0_mediana(
    audio: np.ndarray,
    sr: int = 16000,
    f0_min: float = 75.0,
    f0_max: float = 350.0,
) -> float | None:
    """Estima el tono fundamental medio (F0 en Hz) mediante autocorrelación normalizada en tramas sonoras.

    Permite clasificar el registro vocal del interlocutor (grave, barítono, agudo/femenino)
    para descartar candidatos anatómicamente incompatibles. Devuelve None si no hay suficiente voz.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < int(sr * 0.15):
        return None

    tam_trama = int(sr * 0.035)  # 35 ms
    paso = int(sr * 0.015)       # 15 ms
    lag_min = max(2, int(sr / f0_max))
    lag_max = min(tam_trama - 2, int(sr / f0_min))

    n_tramas = (len(audio) - tam_trama) // paso
    if n_tramas <= 0:
        return None

    f0s: list[float] = []
    for i in range(n_tramas):
        o = i * paso
        trama = audio[o : o + tam_trama]
        e = float(np.sum(trama * trama))
        if e < 1e-6:
            continue
        # Autocorrelación normalizada
        r = np.correlate(trama, trama, mode="full")[tam_trama - 1 :]
        r_norm = r / (math.sqrt(e * max(float(r[0]), 1e-9)) + 1e-9)

        r_sub = r_norm[lag_min : lag_max + 1]
        idx_pico = int(np.argmax(r_sub))
        val_pico = float(r_sub[idx_pico])
        lag_opt = lag_min + idx_pico

        # Umbral de periodicidad armónica
        if val_pico >= 0.35 and 1 < lag_opt < len(r_norm) - 1:
            # Interpolación parabólica sub-muestra
            alfa = float(r_norm[lag_opt - 1])
            beta = float(r_norm[lag_opt])
            gamma = float(r_norm[lag_opt + 1])
            denom = 2.0 * (2.0 * beta - alfa - gamma)
            delta = (alfa - gamma) / denom if abs(denom) > 1e-8 else 0.0
            lag_fino = lag_opt + delta
            f0 = sr / float(lag_fino)
            if f0_min <= f0 <= f0_max:
                f0s.append(f0)

    if len(f0s) < 3:
        return None
    return float(np.median(f0s))


def clasificar_registro_vocal(f0_hz: float | None) -> str:
    """Clasifica el registro vocal según F0: 'grave' (<135 Hz), 'medio' (135-175 Hz), 'agudo' (>175 Hz)."""
    if f0_hz is None:
        return "desconocido"
    if f0_hz < 135.0:
        return "grave"
    if f0_hz > 175.0:
        return "agudo"
    return "medio"

