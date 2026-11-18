"""Huellas de voz: convierte audio en un vector que resume cómo suena la voz."""
import io
from math import gcd
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

try:
    import torch
    _no_grad = torch.no_grad
except ImportError:  # permite usar ONNX y ejecutar tests sin PyTorch instalado
    torch = None

    def _no_grad():
        return lambda f: f

from metadatos_modelo import InfoModelo, info_por_version, inspeccionar_onnx

SAMPLE_RATE = 16000
TROZO_SEG = 6        # cada clip se parte en trozos de 6 s
TROZO_MIN_SEG = 2    # trozos más cortos se descartan


@lru_cache(maxsize=1)
def separador():
    from demucs.pretrained import get_model
    m = get_model("htdemucs")
    m.eval()
    return m


@_no_grad()
def solo_voz(estereo_44k: np.ndarray) -> np.ndarray:
    """Quita música y efectos con Demucs. Entra estéreo 44.1 kHz, sale mono 16 kHz."""
    from demucs.apply import apply_model
    import torchaudio.functional as F
    m = separador()
    x = torch.from_numpy(estereo_44k.T.copy())[None]
    fuentes = apply_model(m, x, device="cpu", shifts=0, split=True, overlap=0.1, progress=False)[0]
    voz = fuentes[m.sources.index("vocals")].mean(0)
    return F.resample(voz, m.samplerate, SAMPLE_RATE).numpy()


@lru_cache(maxsize=1)
def modelo():
    from speechbrain.inference.speaker import EncoderClassifier
    raiz = Path(__file__).resolve().parent.parent
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(raiz / "modelos" / "ecapa"),
        run_opts={"device": "cpu"},
    )


def _a_mono_16k(audio: np.ndarray, sr_origen: int, max_seg: int | None = None) -> np.ndarray:
    """Convierte a mono float32 a 16 kHz con remuestreo polifásico antialiasing."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if max_seg:
        audio = audio[: int(max_seg * sr_origen)]
    if sr_origen != SAMPLE_RATE and len(audio) > 0:
        g = gcd(int(sr_origen), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr_origen) // g).astype(np.float32)
    if max_seg:
        audio = audio[: int(max_seg * SAMPLE_RATE)]
    return audio


def cargar_audio_bytes(datos: bytes, max_seg: int | None = None) -> np.ndarray:
    """Decodifica bytes de audio (MP3, WAV, FLAC, OGG...) directamente a 16 kHz mono float32."""
    audio, sr_origen = sf.read(io.BytesIO(datos), dtype="float32")
    return _a_mono_16k(audio, sr_origen, max_seg=max_seg)


def cargar_audio(ruta: str | Path, separar: bool = False, max_seg: int | None = None) -> np.ndarray:
    """Cualquier formato (mp3, webm del móvil, wav...) -> 16 kHz mono float32.

    separar=True quita antes la música y los efectos (más lento).
    """
    ruta = Path(ruta)
    canales, frecuencia = ("2", "44100") if separar else ("1", str(SAMPLE_RATE))
    limite = ["-t", str(max_seg)] if max_seg else []

    # 1. Intentar decodificación directa con soundfile (WAV, FLAC, OGG, MP3...) sin requerir ffmpeg externo
    if not separar:
        try:
            audio, sr_origen = sf.read(str(ruta), dtype="float32")
            return _a_mono_16k(audio, sr_origen, max_seg=max_seg)
        except Exception:
            pass

    # 2. Si requiere demuxing/conversión (mp3, webm, mp4...) o separación demucs, usar ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta), *limite,
             "-ac", canales, "-ar", frecuencia, tmp_name],
            check=True,
        )
        audio, _ = sf.read(tmp_name, dtype="float32")
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

    return solo_voz(audio) if separar else audio


def trozos(audio: np.ndarray, filtrar_silencio: bool = False) -> list[np.ndarray]:
    paso = TROZO_SEG * SAMPLE_RATE
    min_muestras = TROZO_MIN_SEG * SAMPLE_RATE
    if len(audio) < min_muestras:
        return []
    if len(audio) < int(6.5 * SAMPLE_RATE):
        candidatos = [audio[:paso]] if len(audio) > paso else [audio]
    else:
        candidatos = []
        min_cola = int(0.5 * SAMPLE_RATE)
        for i in range(0, len(audio), paso):
            trozo = audio[i : i + paso]
            if len(trozo) == paso:
                candidatos.append(trozo)
            elif len(trozo) >= min_cola:
                # Ventana solapada completa de 6 s al final para aprovechar tomas >= 6.5 s sin colas cortas
                candidatos.append(audio[-paso:] if len(audio) >= paso else trozo)
    if filtrar_silencio or os.environ.get("ESTAVOZ_VAD") == "1":
        try:
            from aumento_acustico import DetectorVoz
            det = DetectorVoz(sr=SAMPLE_RATE, min_voz_util_seg=float(TROZO_MIN_SEG))
            candidatos = [p for p in candidatos if det.analizar(p).es_toma_valida]
        except Exception:
            pass
    return candidatos


# Con ESTAVOZ_ONNX=ruta.onnx se usa el mismo modelo que corre en el navegador,
# para que las huellas del índice y las del móvil salgan idénticas.
# Si no se define expresamente, pero existe el ONNX en frontend/modelo/voxblink-int8.onnx,
# se utiliza directamente para asegurar coherencia con los artefactos publicados y evitar
# caídas accidentales a ECAPA (Fase 1 / Fase 4 de la auditoría).
_RUTA_ONNX_DEFECTO = Path(__file__).resolve().parent.parent / "frontend" / "modelo" / "voxblink-int8.onnx"
ONNX = os.environ.get("ESTAVOZ_ONNX") or (str(_RUTA_ONNX_DEFECTO) if _RUTA_ONNX_DEFECTO.exists() else None)


@lru_cache(maxsize=1)
def sesion_onnx():
    import onnxruntime as ort
    opciones = ort.SessionOptions()
    if os.environ.get("ESTAVOZ_HILOS"):   # para repartir los núcleos entre varios procesos
        opciones.intra_op_num_threads = int(os.environ["ESTAVOZ_HILOS"])
    return ort.InferenceSession(ONNX, opciones)


@lru_cache(maxsize=1)
def info_modelo_actual() -> InfoModelo:
    """Devuelve los metadatos inequívocos del modelo activo (ONNX o ECAPA por defecto)."""
    ruta_onnx = os.environ.get("ESTAVOZ_ONNX") or ONNX
    if ruta_onnx:
        return inspeccionar_onnx(ruta_onnx)
    return info_por_version("1")


@_no_grad()
def huella_trozo(trozo: np.ndarray) -> np.ndarray:
    if ONNX:
        return sesion_onnx().run(None, {"wav": trozo[None].astype(np.float32)})[0][0]
    return modelo().encode_batch(torch.from_numpy(trozo).unsqueeze(0)).squeeze().numpy()


def huellas(audio: np.ndarray) -> np.ndarray:
    """Una huella normalizada (256 números en VoxBlink2, 192 en ECAPA) por cada trozo del audio."""
    partes = trozos(audio)
    if not partes:
        dim = sesion_onnx().get_outputs()[0].shape[-1] if ONNX else 192
        return np.zeros((0, dim if isinstance(dim, int) else 192), dtype=np.float32)
    vecs = [huella_trozo(p) for p in partes]
    vecs = np.stack(vecs).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
