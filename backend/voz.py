"""Huellas de voz: convierte audio en un vector que resume cómo suena la voz."""
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
TROZO_SEG = 6        # cada clip se parte en trozos de 6 s
TROZO_MIN_SEG = 2    # trozos más cortos se descartan


@lru_cache(maxsize=1)
def modelo():
    from speechbrain.inference.speaker import EncoderClassifier
    raiz = Path(__file__).resolve().parent.parent
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(raiz / "modelos" / "ecapa"),
        run_opts={"device": "cpu"},
    )


def cargar_audio(ruta: str | Path) -> np.ndarray:
    """Cualquier formato (mp3, webm del móvil, wav...) -> 16 kHz mono float32."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta),
             "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
            check=True,
        )
        audio, _ = sf.read(tmp.name, dtype="float32")
    return audio


def trozos(audio: np.ndarray) -> list[np.ndarray]:
    paso = TROZO_SEG * SAMPLE_RATE
    partes = [audio[i:i + paso] for i in range(0, len(audio), paso)]
    return [p for p in partes if len(p) >= TROZO_MIN_SEG * SAMPLE_RATE]


@torch.no_grad()
def huellas(audio: np.ndarray) -> np.ndarray:
    """Una huella normalizada (192 números) por cada trozo del audio."""
    partes = trozos(audio)
    if not partes:
        return np.zeros((0, 192), dtype=np.float32)
    vecs = []
    for p in partes:
        emb = modelo().encode_batch(torch.from_numpy(p).unsqueeze(0)).squeeze()
        vecs.append(emb.numpy())
    vecs = np.stack(vecs).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
