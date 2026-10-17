"""Compara formas de puntuar usando las huellas ya calculadas por evaluar.py.

Misma prueba que evaluar.py (se aparta cada clip y se buscan sus fragmentos de 8 s
entre todos los demás), pero sin recalcular huellas: lee data/cache_huellas/<modo>/.

Uso: .venv/bin/python backend/experimentar.py [normal-onnx]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lda import entrenar_lda  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def cargar(modo):
    clips = []
    for f in sorted((RAIZ / "data" / "cache_huellas" / modo).glob("*.npz")):
        d = np.load(f)
        if len(d["todo"]):
            clips.append((f.name.split("__")[0], d["todo"], [d[k] for k in d.files if k.startswith("q")]))
    return clips


def normaliza(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def preparar(clips, centrar):
    actores = sorted({a for a, _, _ in clips})
    idx = {a: i for i, a in enumerate(actores)}
    V = np.concatenate([t for _, t, _ in clips])
    fila_actor = np.concatenate([[idx[a]] * len(t) for a, t, _ in clips])
    fila_clip = np.concatenate([[i] * len(t) for i, (_, t, _) in enumerate(clips)])
    mu = V.mean(0) if centrar else np.zeros(V.shape[1], dtype=V.dtype)
    V = normaliza(V - mu)
    consultas = []  # (clip, actor, huellas de la consulta)
    por_actor = defaultdict(int)
    for a, _, qs in clips:
        por_actor[a] += bool(qs)  # solo cuentan los clips de los que salen consultas (eldoblaje)
    for i, (a, _, qs) in enumerate(clips):
        if por_actor[a] < 2:
            continue
        for q in qs:
            consultas.append((i, idx[a], normaliza(q - mu)))
    return actores, V, fila_actor, fila_clip, consultas


def evaluar(clips, centrar=False, metodo="top3", asnorm=0):
    actores, V, fila_actor, fila_clip, consultas = preparar(clips, centrar)
    n_act = len(actores)
    # centroides por actor (se recalculan sin el clip apartado)
    suma = np.zeros((n_act, V.shape[1]))
    np.add.at(suma, fila_actor, V)
    cuenta = np.bincount(fila_actor, minlength=n_act)
    cent_todos = normaliza(suma / cuenta[:, None])
    # estadísticas de cohorte de cada actor (AS-norm): parecido de su centroide con los demás
    if asnorm:
        cc = cent_todos @ cent_todos.T
        np.fill_diagonal(cc, -np.inf)
        top = -np.sort(-cc, axis=1)[:, :asnorm]
        mu_a, sd_a = top.mean(1), top.std(1) + 1e-6

    top1 = top3 = 0
    for clip, real, q in consultas:
        fuera = fila_clip == clip
        if metodo == "centroide":
            s_q = suma.copy(); c_q = cuenta.copy()
            s_q[real] -= V[fuera].sum(0); c_q[real] -= fuera.sum()
            punt = (q @ normaliza(s_q / c_q[:, None]).T).mean(0)
        else:
            sims = (q @ V.T).mean(0)
            sims[fuera] = -np.inf
            punt = np.full(n_act, -np.inf)
            orden = np.lexsort((-sims, fila_actor))  # por actor, de más a menos parecido
            fa, sv = fila_actor[orden], sims[orden]
            inicio = np.searchsorted(fa, np.arange(n_act))
            fin = np.searchsorted(fa, np.arange(n_act), side="right")
            k = {"top1": 1, "top3": 3, "top5": 5, "todos": None}[metodo]
            for a in range(n_act):
                v = sv[inicio[a]:fin[a]]
                v = v[np.isfinite(v)][:k] if k else v[np.isfinite(v)]
                if len(v):
                    punt[a] = v.mean()
        if asnorm:
            ok = np.isfinite(punt)
            coh = -np.sort(-punt[ok])[1:asnorm + 1]   # cohorte de la consulta (sin el mejor)
            mu_q, sd_q = coh.mean(), coh.std() + 1e-6
            punt = np.where(ok, 0.5 * ((punt - mu_q) / sd_q + (punt - mu_a) / sd_a), -np.inf)
        orden = np.argsort(-punt)
        top1 += orden[0] == real
        top3 += real in orden[:3]
    n = len(consultas)
    return 100 * top1 / n, 100 * top3 / n, n

def evaluar_lda(clips, dims=150, metodo="top3", pliegues=2, semilla=0, reg=1e-3):
    """Validación cruzada por actores: la LDA nunca ha visto a los actores que se evalúan."""
    actores = sorted({a for a, _, _ in clips})
    rng = np.random.default_rng(semilla)
    pliegue = dict(zip(actores, rng.integers(0, pliegues, len(actores))))
    top1 = top3 = n = 0
    for p in range(pliegues):
        entreno = [(a, t) for a, t, _ in clips if pliegue[a] != p]
        Xe = np.concatenate([t for _, t in entreno])
        ye = np.concatenate([[a] * len(t) for a, t in entreno])
        # solo actores con 2+ clips aportan variación "misma voz, distinta grabación"
        multi = {a for a in set(ye) if sum(1 for x, _ in entreno if x == a) >= 2}
        sel = np.isin(ye, list(multi))
        mu, W = entrenar_lda(Xe[sel], ye[sel], dims, reg)
        proy = lambda x: x if not len(x) else normaliza((x - mu) @ W)
        clips_p = [(a, proy(t), [proy(q) for q in qs] if pliegue[a] == p else []) for a, t, qs in clips]
        a1, a3, k = evaluar(clips_p, metodo=metodo)
        top1 += a1 * k / 100; top3 += a3 * k / 100; n += k
    return 100 * top1 / n, 100 * top3 / n, n


def principal_lda():
    clips = cargar(sys.argv[2] if len(sys.argv) > 2 else "normal-onnx")
    for dims in (100, 150, 192):
        for metodo in ("top3", "centroide"):
            a1, a3, n = evaluar_lda(clips, dims=dims, metodo=metodo)
            print(f"  LDA {dims:>3} dims + {metodo:<10} {a1:5.1f}%  top3 {a3:5.1f}%   ({n} consultas)", flush=True)


def consultas_movil(modo="normal-onnx", semilla=0):
    """Rehace las consultas de 8 s de cada clip de eldoblaje como si se grabaran con el móvil."""
    from simular_movil import movil
    from voz import SAMPLE_RATE, cargar_audio, huellas
    cache = RAIZ / "data" / "cache_huellas" / (modo + "-movil")
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    for f in sorted((RAIZ / "data" / "cache_huellas" / modo).glob("*.npz")):
        destino = cache / f.name
        if destino.exists() or "podcast_" in f.name:
            continue
        d = np.load(f)
        actor, nombre = f.stem.split("__")
        clip = RAIZ / "data" / "actores" / actor / "clips" / f"{nombre}.mp3"
        audio = cargar_audio(clip, max_seg=90)
        paso = 8 * SAMPLE_RATE
        qs = [huellas(movil(audio[i:i + paso], rng)) for i in range(0, max(len(audio) - paso, 1), paso)][:3]
        np.savez(destino, todo=d["todo"], **{f"q{i}": q for i, q in enumerate(qs) if len(q)})


def principal_movil():
    consultas_movil()
    for modo, nombre in (("normal-onnx", "audio limpio"), ("normal-onnx-movil", "simulando móvil")):
        clips = cargar(modo)
        a1, a3, n = evaluar_lda(clips, dims=192, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  {nombre:<16} {a1:5.1f}%  top3 {a3:5.1f}%   ({n} consultas)", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["movil"]:
    principal_movil()
elif __name__ == "__main__" and sys.argv[1:2] == ["lda"]:
    principal_lda()
elif __name__ == "__main__":
    clips = cargar(sys.argv[1] if len(sys.argv) > 1 else "normal-onnx")
    print(f"{len({a for a, _, _ in clips})} actores, {len(clips)} clips")
    pruebas = [
        ("actual (media de los 3 mejores trozos)", dict()),
        ("solo el mejor trozo", dict(metodo="top1")),
        ("media de los 5 mejores", dict(metodo="top5")),
        ("media de todos los trozos", dict(metodo="todos")),
        ("centroide por actor", dict(metodo="centroide")),
        ("centrar huellas + top3", dict(centrar=True)),
        ("centrar huellas + centroide", dict(centrar=True, metodo="centroide")),
        ("top3 + AS-norm 100", dict(asnorm=100)),
        ("centrar + top3 + AS-norm 100", dict(centrar=True, asnorm=100)),
        ("centrar + centroide + AS-norm 100", dict(centrar=True, metodo="centroide", asnorm=100)),
    ]
    for nombre, kw in pruebas:
        a1, a3, n = evaluar(clips, **kw)
        print(f"  {nombre:<40} {a1:5.1f}%  top3 {a3:5.1f}%   ({n} consultas)", flush=True)
