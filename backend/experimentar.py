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


def aumentar(modo="normal-onnx", semilla=1):
    """Huellas de cada clip como si se hubiera grabado con el móvil (otra semilla que la de las consultas)."""
    from simular_movil import movil
    from voz import cargar_audio, huellas
    salida = RAIZ / "data" / "cache_huellas" / (modo + "-aumentado")
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    for f in sorted((RAIZ / "data" / "cache_huellas" / modo).glob("*.npz")):
        destino = salida / f.name
        if destino.exists():
            continue
        actor, nombre = f.stem.split("__")
        clips = list((RAIZ / "data" / "actores" / actor / "clips").glob(nombre + ".*"))
        np.savez(destino, todo=huellas(movil(cargar_audio(clips[0], max_seg=90), rng)))


def combinar(limpio, aumentado):
    """Cada clip con sus huellas limpias + las simuladas de móvil; las consultas se quedan igual."""
    aum = {(a, i): t for i, (a, t, _) in enumerate(aumentado)}
    return [(a, np.concatenate([t, aum[(a, i)]]) if (a, i) in aum and len(aum[(a, i)]) else t, qs)
            for i, (a, t, qs) in enumerate(limpio)]


def principal_aumentado():
    aumentar()
    aum = cargar("normal-onnx-aumentado")
    for modo, nombre in (("normal-onnx", "audio limpio"), ("normal-onnx-movil", "simulando móvil")):
        base = cargar(modo)
        assert [a for a, _, _ in base] == [a for a, _, _ in aum]
        a1, a3, n = evaluar_lda(combinar(base, aum), dims=192, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  con aumento, {nombre:<16} {a1:5.1f}%  top3 {a3:5.1f}%   ({n} consultas)", flush=True)


def principal_duracion():
    """¿Cuánto ayuda grabar más? Consultas de móvil simulado de 10 y 15 s, con el índice aumentado."""
    from simular_movil import movil
    from voz import SAMPLE_RATE, cargar_audio, huellas
    base = cargar("normal-onnx")
    aum = cargar("normal-onnx-aumentado")
    nombres = sorted(f for f in (RAIZ / "data" / "cache_huellas" / "normal-onnx").glob("*.npz") if np.load(f)["todo"].size)
    for seg in (10, 15):
        cache = RAIZ / "data" / "cache_huellas" / f"movil-{seg}s"
        cache.mkdir(exist_ok=True)
        rng = np.random.default_rng(seg)
        clips = []
        for (a, t, _), f in zip(base, nombres):
            destino = cache / f.name
            if not destino.exists():
                actor, nombre = f.stem.split("__")
                audio = cargar_audio(RAIZ / "data" / "actores" / actor / "clips" / f"{nombre}.mp3", max_seg=90)
                paso = seg * SAMPLE_RATE
                qs = [huellas(movil(audio[i:i + paso], rng)) for i in range(0, max(len(audio) - paso, 1), paso)][:2]
                np.savez(destino, **{f"q{i}": q for i, q in enumerate(qs) if len(q)})
            d = np.load(destino)
            clips.append((a, t, [d[k] for k in d.files]))
        a1, a3, n = evaluar_lda(combinar(clips, aum), dims=192, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  móvil simulado, grabando {seg} s: {a1:5.1f}%  top3 {a3:5.1f}%   ({n} consultas)", flush=True)


def comparar_modelos():
    """ECAPA (actual) frente a SimAM-ResNet34 VoxBlink2, con las mismas grabaciones de móvil simuladas.

    Por cada clip: huellas del clip limpio y "de móvil" (para el índice) y hasta 2 consultas de 10 s
    "de móvil" con exactamente el mismo audio para los dos modelos.
    """
    import voxblink
    from simular_movil import movil
    from voz import SAMPLE_RATE, cargar_audio, huellas
    salida = RAIZ / "data" / "cache_huellas" / "comparar"
    salida.mkdir(parents=True, exist_ok=True)
    rng_aum, rng_q = np.random.default_rng(1), np.random.default_rng(10)
    nombres = sorted(f for f in (RAIZ / "data" / "cache_huellas" / "normal-onnx").glob("*.npz") if np.load(f)["todo"].size)
    for n, f in enumerate(nombres, 1):
        destino = salida / f.name
        actor, nombre = f.stem.split("__")
        audio = cargar_audio(RAIZ / "data" / "actores" / actor / "clips" / f"{nombre}.mp3", max_seg=90)
        aum = movil(audio, rng_aum)
        paso = 10 * SAMPLE_RATE
        qs = [movil(audio[i:i + paso], rng_q) for i in range(0, max(len(audio) - paso, 1), paso)][:2]
        if destino.exists():
            continue  # (el audio se genera igual para que los generadores aleatorios no cambien)
        d = {"vb_todo": voxblink.huellas(audio), "vb_aum": voxblink.huellas(aum)}
        for i, q in enumerate(qs):
            d[f"vb_q{i}"], d[f"ec_q{i}"] = voxblink.huellas(q), huellas(q)
        np.savez(destino, **d)
        if n % 50 == 0:
            print(f"  {n}/{len(nombres)} clips", flush=True)

    ecapa_base, ecapa_aum = cargar("normal-onnx"), cargar("normal-onnx-aumentado")
    vb_base, vb_aum, ec_q = [], [], []
    for (a, t, _), f in zip(ecapa_base, nombres):
        d = np.load(salida / f.name)
        qv = [d[k] for k in d.files if k.startswith("vb_q") and len(d[k])]
        qe = [d[k] for k in d.files if k.startswith("ec_q") and len(d[k])]
        vb_base.append((a, d["vb_todo"], qv)); vb_aum.append((a, d["vb_aum"], []))
        ec_q.append((a, t, qe))
    for nombre, clips in (("ECAPA (actual)", combinar(ec_q, ecapa_aum)), ("VoxBlink2 SimAM-ResNet34", combinar(vb_base, vb_aum))):
        dims = clips[0][1].shape[1]
        a1, a3, k = evaluar_lda(clips, dims=dims, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  {nombre:<26} móvil 10 s: {a1:5.1f}%  top3 {a3:5.1f}%   ({k} consultas)", flush=True)


def cargar_vb():
    """Clips con las huellas VoxBlink2 de la comparación de modelos: (actor, limpio+móvil, consultas de móvil)."""
    salida = RAIZ / "data" / "cache_huellas" / "comparar"
    clips = []
    for f in sorted(salida.glob("*.npz")):
        d = np.load(f)
        actor = f.stem.split("__")[0]
        todo = np.concatenate([d["vb_todo"], d["vb_aum"]]) if len(d["vb_aum"]) else d["vb_todo"]
        qs = [d[k] for k in d.files if k.startswith("vb_q") and len(d[k])]
        if len(todo):
            clips.append((actor, todo, qs))
    return clips


def entrenar_adaptador(X, y, dim=256, epocas=60, margen=0.2, escala=30.0, semilla=0):
    """Pequeña red (256 -> 512 -> dim) entrenada con AAM-softmax para separar a los dobladores."""
    import torch
    import torch.nn as nn
    torch.manual_seed(semilla)
    clases = {c: i for i, c in enumerate(sorted(set(y)))}
    t = torch.tensor(np.array([clases[c] for c in y]))
    Xt = torch.tensor(X, dtype=torch.float32)
    red = nn.Sequential(nn.Linear(X.shape[1], 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, dim))
    W = nn.Parameter(torch.randn(len(clases), dim) * 0.01)
    opt = torch.optim.AdamW(list(red.parameters()) + [W], lr=2e-3, weight_decay=1e-3)
    for _ in range(epocas):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            e = nn.functional.normalize(red(Xt[b]), dim=1)
            cos = e @ nn.functional.normalize(W, dim=1).T
            ang = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            logits = escala * torch.where(nn.functional.one_hot(t[b], len(clases)).bool(), torch.cos(ang + margen), cos)
            loss = nn.functional.cross_entropy(logits, t[b])
            opt.zero_grad(); loss.backward(); opt.step()
    red.eval()
    return lambda x: (lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True))(red(torch.tensor(x, dtype=torch.float32)).detach().numpy()) if len(x) else x


def evaluar_proyeccion(clips, entrenar, pliegues=5, semilla=0):
    """Como evaluar_lda, pero con cualquier proyección entrenada solo con los dobladores de los otros pliegues."""
    actores = sorted({a for a, _, _ in clips})
    rng = np.random.default_rng(semilla)
    pliegue = dict(zip(actores, rng.integers(0, pliegues, len(actores))))
    top1 = top3 = n = 0
    for p in range(pliegues):
        entreno = [(a, t) for a, t, _ in clips if pliegue[a] != p]
        Xe = np.concatenate([t for _, t in entreno]); ye = np.concatenate([[a] * len(t) for a, t in entreno])
        proy = entrenar(Xe, ye)
        clips_p = [(a, proy(t), [proy(q) for q in qs] if pliegue[a] == p else []) for a, t, qs in clips]
        a1, a3, k = evaluar(clips_p, metodo="centroide")
        top1 += a1 * k / 100; top3 += a3 * k / 100; n += k
    return 100 * top1 / n, 100 * top3 / n, n


def principal_mejoras():
    clips = cargar_vb()
    print(f"{len({a for a, _, _ in clips})} actores, {len(clips)} clips, {sum(len(q) for _, _, q in clips)} consultas")

    def lda(X, y, dims=None):
        from lda import entrenar_lda, proyectar
        mu, W = entrenar_lda(X, y, dims)
        return lambda x: proyectar(x, mu, W) if len(x) else x
    pruebas = [
        ("LDA (actual)", lambda X, y: lda(X, y)),
        ("LDA 200 dimensiones", lambda X, y: lda(X, y, 200)),
        ("LDA 150 dimensiones", lambda X, y: lda(X, y, 150)),
        ("adaptador AAM 256", lambda X, y: entrenar_adaptador(X, y)),
        ("adaptador AAM 256 + LDA", lambda X, y: (lambda f: (lambda g: lambda x: g(f(x)) if len(x) else x)(lda(f(X), y)))(entrenar_adaptador(X, y))),
    ]
    for nombre, f in pruebas:
        a1, a3, k = evaluar_proyeccion(clips, f)
        print(f"  {nombre:<28} {a1:5.1f}%  top3 {a3:5.1f}%   ({k} consultas)", flush=True)


def comparar_r100(n_actores=250):
    """SimAM-ResNet34 frente a ResNet100 (VoxBlink2), con exactamente el mismo audio de móvil simulado
    que la comparación de modelos, sobre un subconjunto de dobladores (sin aumento, igual para los dos)."""
    import voxblink
    from simular_movil import movil
    from voz import SAMPLE_RATE, cargar_audio
    origen = RAIZ / "data" / "cache_huellas" / "comparar"
    salida = RAIZ / "data" / "cache_huellas" / "r100"
    salida.mkdir(parents=True, exist_ok=True)
    nombres = sorted(f for f in (RAIZ / "data" / "cache_huellas" / "normal-onnx").glob("*.npz") if np.load(f)["todo"].size)
    actores = sorted({f.stem.split("__")[0] for f in nombres})
    elegidos = set(np.random.default_rng(0).choice(actores, n_actores, replace=False))
    rng_aum, rng_q = np.random.default_rng(1), np.random.default_rng(10)
    for f in nombres:   # mismo orden y mismas llamadas aleatorias que comparar_modelos()
        actor, nombre = f.stem.split("__")
        audio = cargar_audio(RAIZ / "data" / "actores" / actor / "clips" / f"{nombre}.mp3", max_seg=90)
        movil(audio, rng_aum)
        paso = 10 * SAMPLE_RATE
        qs = [movil(audio[i:i + paso], rng_q) for i in range(0, max(len(audio) - paso, 1), paso)][:2]
        destino = salida / f.name
        if actor not in elegidos or destino.exists():
            continue
        d = {"todo": voxblink.huellas(audio, "r100")}
        for i, q in enumerate(qs):
            d[f"q{i}"] = voxblink.huellas(q, "r100")
        np.savez(destino, **d)
        print(f"  {f.stem}", flush=True)

    r34, r100 = [], []
    for f in nombres:
        actor = f.stem.split("__")[0]
        if actor not in elegidos:
            continue
        a, b = np.load(origen / f.name), np.load(salida / f.name)
        r34.append((actor, a["vb_todo"], [a[k] for k in a.files if k.startswith("vb_q") and len(a[k])]))
        r100.append((actor, b["todo"], [b[k] for k in b.files if k.startswith("q") and len(b[k])]))
    for nombre, clips in (("ResNet34 (actual)", r34), ("ResNet100", r100)):
        a1, a3, k = evaluar_lda(clips, dims=256, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  {nombre:<18} {n_actores} dobladores, móvil 10 s: {a1:5.1f}%  top3 {a3:5.1f}%   ({k} consultas)", flush=True)


def largos(n_actores=250, parte=(0, 1)):
    """¿Huellas de fragmentos más largos? ResNet100, 250 dobladores, audio de móvil simulado con una
    semilla propia por clip (para poder repartir el trabajo entre procesos). Por cada clip:
    consultas de 10 s troceadas en 6 + 4 s (como ahora) y enteras, y muestras en trozos de 6 s,
    de 20 s y enteras (hasta 40 s, para que no tarde horas)."""
    import zlib
    import onnxruntime as ort
    from simular_movil import movil
    from voz import SAMPLE_RATE, cargar_audio
    salida = RAIZ / "data" / "cache_huellas" / "r100-largos2"
    salida.mkdir(parents=True, exist_ok=True)
    nombres = sorted(f for f in (RAIZ / "data" / "cache_huellas" / "normal-onnx").glob("*.npz") if np.load(f)["todo"].size)
    actores = sorted({f.stem.split("__")[0] for f in nombres})
    elegidos = set(np.random.default_rng(0).choice(actores, n_actores, replace=False))
    norma = lambda v: v / np.linalg.norm(v)
    # el modelo comprimido de la web (ResNet100), un núcleo por proceso
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 1 if parte[1] > 1 else 4
    sesion = ort.InferenceSession(str(RAIZ / "frontend" / "modelo" / "voxblink-int8.onnx"), opciones)
    huella1 = lambda a: norma(sesion.run(None, {"wav": a[None].astype(np.float32)})[0][0])
    from voz import trozos
    class voxblink:  # misma interfaz que backend/voxblink.py, con el modelo ONNX
        huella = staticmethod(lambda a, _v=None: huella1(a))
        huellas = staticmethod(lambda a, _v=None: np.stack([huella1(t) for t in trozos(a)]) if trozos(a) else np.zeros((0, 256), np.float32))
    for num, f in enumerate(nombres):
        actor, nombre = f.stem.split("__")
        destino = salida / f.name
        if actor not in elegidos or destino.exists() or num % parte[1] != parte[0]:
            continue
        rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
        audio = cargar_audio(RAIZ / "data" / "actores" / actor / "clips" / f"{nombre}.mp3", max_seg=40)
        paso = 10 * SAMPLE_RATE
        qs = [movil(audio[i:i + paso], rng) for i in range(0, max(len(audio) - paso, 1), paso)][:2]
        d = {"t6": voxblink.huellas(audio, "r100"), "entero": norma(voxblink.huella(audio, "r100"))[None]}
        t20 = [t for t in (audio[i:i + 20 * SAMPLE_RATE] for i in range(0, len(audio), 20 * SAMPLE_RATE))
               if len(t) >= 4 * SAMPLE_RATE] or [audio]
        d["t20"] = np.stack([norma(voxblink.huella(t, "r100")) for t in t20])
        for k, q in enumerate(qs):
            if len(q) >= 2 * SAMPLE_RATE:
                d[f"q6_{k}"] = voxblink.huellas(q, "r100")
                d[f"q10_{k}"] = norma(voxblink.huella(q, "r100"))[None]
        np.savez(destino, **d)
        print(f"  {f.stem}", flush=True)
    if parte[1] > 1:
        return
    combos = {}
    for f in sorted(salida.glob("*.npz")):
        b = np.load(f)
        act = f.stem.split("__")[0]
        q6 = [b[k] for k in b.files if k.startswith("q6_") and len(b[k])]
        q10 = [b[k] for k in b.files if k.startswith("q10_")]
        for nombre, ref, qs in (("muestras 6 s + consulta 6+4 s (actual)", b["t6"], q6),
                                ("muestras 6 s + consulta 10 s entera", b["t6"], q10),
                                ("muestras 20 s + consulta 10 s entera", b["t20"], q10),
                                ("muestras enteras + consulta 10 s entera", b["entero"], q10),
                                ("muestras 6 s y 20 s + consulta 10 s", np.concatenate([b["t6"], b["t20"]]), q10)):
            combos.setdefault(nombre, []).append((act, ref, qs))
    for nombre, clips in combos.items():
        a1, a3, k = evaluar_lda(clips, dims=256, metodo="centroide", pliegues=5, reg=1e-2)
        print(f"  {nombre:<44} {a1:5.1f}%  top3 {a3:5.1f}%   ({k} consultas)", flush=True)


def puntuacion():
    """¿Cómo puntuar a cada actor? Con el índice real (todos los dobladores, limpio + #movil) y las
    consultas de móvil de 10 s de "largos": parecido con la huella media, con los 3 trozos más
    parecidos, o una mezcla. Resultado: la mezcla 50/50 gana ~1,5 puntos a la primera y ~3 en el top 3,
    y la escala de parecido no cambia (sirve la misma traducción a porcentaje)."""
    from lda import normaliza, proyectar
    d = np.load(RAIZ / "data" / "indice.npz")
    X, y, c = d["vectores"], d["etiquetas"], d["clips"]
    base = np.array([x.split("#")[0] for x in c])
    actores = sorted(set(y))
    ia = {a: i for i, a in enumerate(actores)}
    fa = np.array([ia[a] for a in y])
    consultas = []
    for f in sorted((RAIZ / "data" / "cache_huellas" / "r100-largos2").glob("*.npz")):
        b = np.load(f)
        act, nom = f.stem.split("__")
        clip = f"{act}/{nom}.mp3"
        if (base == clip).any() and ((y == act) & (base != clip)).any():
            consultas += [(act, clip, b[k]) for k in b.files if k.startswith("q10_")]

    def top3(s, filas):
        out = np.full(len(actores), -1.0)
        orden = np.lexsort((-s, fa[filas]))
        fs, ss = fa[filas][orden], s[orden]
        ini, fin = np.searchsorted(fs, np.arange(len(actores))), np.searchsorted(fs, np.arange(len(actores)), side="right")
        for a in range(len(actores)):
            if fin[a] > ini[a]:
                out[a] = ss[ini[a]:min(fin[a], ini[a] + 3)].mean()
        return out

    pesos = (0, .25, .5, .75, 1)
    res = {w: [0, 0, 0] for w in pesos}
    n_clips = {a: len(set(base[y == a])) for a in actores}
    qact = sorted({q[0] for q in consultas})
    for semilla in range(2):
        pl = dict(zip(qact, np.random.default_rng(semilla).integers(0, 5, len(qact))))
        for p in range(5):
            sel = np.isin(y, [a for a in actores if n_clips[a] >= 2 and pl.get(a, -1) != p])
            mu, W = entrenar_lda(X[sel], y[sel])
            V = proyectar(X, mu, W)
            for act, clip, q in consultas:
                if pl[act] != p:
                    continue
                real, filas = ia[act], np.flatnonzero(base != clip)
                Q = proyectar(q, mu, W)
                suma = np.zeros((len(actores), V.shape[1]))
                np.add.at(suma, fa[filas], V[filas])
                pc, pk = (Q @ normaliza(suma + 1e-9).T).mean(0), top3((Q @ V[filas].T).mean(0), filas)
                for w in pesos:
                    o = np.argsort(-(w * pc + (1 - w) * pk))
                    res[w][0] += o[0] == real; res[w][1] += real in o[:3]; res[w][2] += 1
    for w, (a1, a3, n) in res.items():
        print(f"  {w:.0%} huella media + {1 - w:.0%} 3 mejores trozos: {100 * a1 / n:5.1f}%  top3 {100 * a3 / n:5.1f}%   ({n} consultas)")


if __name__ == "__main__" and sys.argv[1:2] == ["largos"]:
    import torch
    if len(sys.argv) > 2:
        torch.set_num_threads(1)
    largos(parte=tuple(map(int, sys.argv[2].split("/"))) if len(sys.argv) > 2 else (0, 1))
elif __name__ == "__main__" and sys.argv[1:2] == ["puntuacion"]:
    puntuacion()
elif __name__ == "__main__" and sys.argv[1:2] == ["r100"]:
    comparar_r100()
elif __name__ == "__main__" and sys.argv[1:2] == ["mejoras"]:
    principal_mejoras()
elif __name__ == "__main__" and sys.argv[1:2] == ["modelos"]:
    comparar_modelos()
elif __name__ == "__main__" and sys.argv[1:2] == ["duracion"]:
    principal_duracion()
elif __name__ == "__main__" and sys.argv[1:2] == ["aumentado"]:
    principal_aumentado()
elif __name__ == "__main__" and sys.argv[1:2] == ["movil"]:
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
