"""Simula cómo suena una escena grabada con el móvil delante de la tele.

- altavoz de tele: recorta graves y agudos (paso banda) y algo de saturación
- habitación: eco (respuesta al impulso sintética con cola exponencial)
- ruido de fondo de la casa
- micrófono del móvil: recorta graves

Se usa para medir el acierto en condiciones reales (backend/experimentar.py movil) y para
enseñar a la LDA que la misma voz grabada así sigue siendo la misma voz.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 16000


def eco(rng, rt60):
    n = int(SR * rt60)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)    # -60 dB al llegar a rt60
    ir[: int(SR * rng.uniform(0.003, 0.015))] = 0            # llegada del primer rebote
    ir[0] = 1.0                                              # sonido directo
    return ir / np.sqrt((ir ** 2).sum())


def movil(audio: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    x = audio.astype(np.float64)
    # altavoz de la tele
    graves, agudos = rng.uniform(120, 300), rng.uniform(5000, 7500)
    x = sosfilt(butter(2, [graves, agudos], btype="band", fs=SR, output="sos"), x)
    x = np.tanh(x * rng.uniform(1, 3)) / rng.uniform(1, 3)
    # habitación
    x = fftconvolve(x, eco(rng, rng.uniform(0.25, 0.7)))[: len(x)]
    # ruido de la casa (rosa aproximado), relación señal/ruido entre 8 y 25 dB
    ruido = np.cumsum(rng.standard_normal(len(x)))
    ruido = sosfilt(butter(1, 30, btype="high", fs=SR, output="sos"), ruido)
    snr = rng.uniform(8, 25)
    ruido *= np.sqrt((x ** 2).mean() / ((ruido ** 2).mean() + 1e-12) / 10 ** (snr / 10))
    x = x + ruido
    # micrófono del móvil
    x = sosfilt(butter(2, 100, btype="high", fs=SR, output="sos"), x)
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)
