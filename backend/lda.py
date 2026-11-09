"""LDA: proyección de las huellas que separa voces de personas distintas y junta las de
la misma persona aunque cambie la grabación (música, micro, anuncio, escena...).

Se entrena con nuestros propios clips de doblaje. En la prueba con dobladores que la LDA
no había visto subió el acierto a la primera del 65 % al 72,6 % (backend/experimentar.py).
"""
import numpy as np
from scipy.linalg import eigh


def normaliza(x):
    x64 = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x64, axis=-1, keepdims=True)
    return (x64 / np.maximum(n, 1e-12)).astype(np.asarray(x).dtype if np.asarray(x).dtype == np.float32 else np.float64)


def estimar_covarianza_canal(X_limpio: np.ndarray, X_aumentado: np.ndarray) -> np.ndarray:
    """Estima la matriz de dispersión acústica/canal (micro, sala, reverberación) a partir de pares.

    Cada fila de Delta representa la distorsión inducida por el canal sobre la misma locución.
    Inyectar esta covarianza en Sw minimiza activamente las direcciones afectadas por el entorno.
    """
    delta = np.asarray(X_aumentado, dtype=np.float64) - np.asarray(X_limpio, dtype=np.float64)
    if len(delta) == 0:
        d = X_limpio.shape[1] if len(X_limpio) else 256
        return np.zeros((d, d), dtype=np.float64)
    delta_c = delta - delta.mean(0)
    return (delta_c.T @ delta_c) / len(delta)


def entrenar_lda(
    X: np.ndarray,
    y: np.ndarray,
    dims: int | None = None,
    reg: float = 1e-2,
    cov_acustica: np.ndarray | None = None,
    peso_cov_acustica: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """X: huellas normalizadas (n, dim); y: actor de cada una. Devuelve (media, matriz W).

    Permite regularización acústica mediante `cov_acustica` (matriz de dispersión inter-canal/sala).
    """
    mu = X.mean(0)
    Xc = X - mu
    Sw = np.zeros((X.shape[1],) * 2, dtype=np.float64)
    Sb = np.zeros_like(Sw)
    for c in np.unique(y):
        Xk = Xc[y == c]
        mk = Xk.mean(0)
        Sw += (Xk - mk).T @ (Xk - mk)
        Sb += len(Xk) * np.outer(mk, mk)
    Sw /= len(X)
    Sb /= len(X)

    if cov_acustica is not None and cov_acustica.shape == Sw.shape:
        peso = max(0.0, min(1.0, float(peso_cov_acustica)))
        # Normalizar traza relativa para combinación balanceada
        tr_sw = float(np.trace(Sw))
        tr_cov = float(np.trace(cov_acustica))
        escala_cov = (tr_sw / max(tr_cov, 1e-9)) if tr_cov > 0 else 1.0
        Sw = (1.0 - peso) * Sw + peso * (cov_acustica * escala_cov)

    Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    val, vec = eigh(Sb, Sw)
    return mu, vec[:, np.argsort(-val)[:dims or X.shape[1]]]


def proyectar(X, mu, W):
    return normaliza((X - mu) @ W)


def centroides(Xp, y, orden):
    """Huella media (ya proyectada) de cada actor, en el orden dado."""
    return normaliza(np.stack([Xp[y == a].mean(0) for a in orden]))
