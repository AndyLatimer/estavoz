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


# --- Versión realista: ecos de habitaciones y ruidos grabados de verdad -------------------------
# Base pública OpenSLR 28 (RIRS_NOISES, licencia Apache 2.0) en data/ruidos/RIRS_NOISES.
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

RUIDOS = Path(__file__).resolve().parent.parent / "data" / "ruidos" / "RIRS_NOISES"


@lru_cache(maxsize=1)
def _banco(n_ecos: int = 400, n_ruidos: int = 400, semilla: int = 0):
    import soundfile as sf
    rng = np.random.default_rng(semilla)
    reales = sorted(RUIDOS.glob("real_rirs_isotropic_noises/*rir*.wav"))
    simulados = sorted(RUIDOS.glob("simulated_rirs/smallroom/*/*.wav")) + sorted(RUIDOS.glob("simulated_rirs/mediumroom/*/*.wav"))
    rutas_eco = list(rng.choice(reales, min(len(reales), n_ecos // 2), replace=False)) + \
                list(rng.choice(simulados, n_ecos // 2, replace=False))
    ruidos = sorted(RUIDOS.glob("pointsource_noises/*.wav")) + sorted(RUIDOS.glob("real_rirs_isotropic_noises/*noise*.wav"))
    rutas_ruido = rng.choice(ruidos, min(len(ruidos), n_ruidos), replace=False)

    def leer(r):
        x, _ = sf.read(r, dtype="float32", always_2d=True)
        return x[:, 0]
    ecos = []
    for r in rutas_eco:
        x = leer(r)
        x = x[max(0, int(np.argmax(np.abs(x))) - 16):]   # empezar justo antes del sonido directo
        ecos.append(x / (np.sqrt((x ** 2).sum()) + 1e-9))
    return ecos, [x for x in map(leer, rutas_ruido) if len(x) > SR // 4]


def movil_real(audio: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Como movil(), pero con eco de una habitación real o simulada y ruido grabado de verdad."""
    rng = rng or np.random.default_rng()
    ecos, ruidos = _banco()
    x = audio.astype(np.float64)
    # altavoz de la tele
    graves, agudos = rng.uniform(120, 300), rng.uniform(5000, 7500)
    x = sosfilt(butter(2, [graves, agudos], btype="band", fs=SR, output="sos"), x)
    g = rng.uniform(1, 3)
    x = np.tanh(x * g) / g
    # habitación (eco real)
    x = fftconvolve(x, ecos[rng.integers(len(ecos))])[: len(x)]
    # ruido real de fondo, repetido si es corto, entre 5 y 20 dB por debajo
    r = ruidos[rng.integers(len(ruidos))]
    r = np.tile(r, len(x) // len(r) + 1)[: len(x)].astype(np.float64)
    snr = rng.uniform(5, 20)
    r *= np.sqrt((x ** 2).mean() / ((r ** 2).mean() + 1e-12) / 10 ** (snr / 10))
    x = x + r
    # micrófono del móvil
    x = sosfilt(butter(2, 100, btype="high", fs=SR, output="sos"), x)
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)
