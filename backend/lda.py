"""LDA: proyección de las huellas que separa voces de personas distintas y junta las de
la misma persona aunque cambie la grabación (música, micro, anuncio, escena...).

Se entrena con nuestros propios clips de doblaje. En la prueba con dobladores que la LDA
no había visto subió el acierto a la primera del 65 % al 72,6 % (backend/experimentar.py).
"""
import numpy as np
from scipy.linalg import eigh


def normaliza(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def entrenar_lda(X, y, dims=192, reg=1e-2):
    """X: huellas normalizadas (n, 192); y: actor de cada una. Devuelve (media, matriz W)."""
    mu = X.mean(0)
    Xc = X - mu
    Sw = np.zeros((X.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in np.unique(y):
        Xk = Xc[y == c]
        mk = Xk.mean(0)
        Sw += (Xk - mk).T @ (Xk - mk)
        Sb += len(Xk) * np.outer(mk, mk)
    Sw /= len(X)
    Sb /= len(X)
    Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    val, vec = eigh(Sb, Sw)
    return mu, vec[:, np.argsort(-val)[:dims]]


def proyectar(X, mu, W):
    return normaliza((X - mu) @ W)


def centroides(Xp, y, orden):
    """Huella media (ya proyectada) de cada actor, en el orden dado."""
    return normaliza(np.stack([Xp[y == a].mean(0) for a in orden]))
