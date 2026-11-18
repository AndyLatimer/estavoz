"""Pruebas automatizadas del cliente web (`frontend/index.html`, `privacidad.html`, `manifest.json`)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
FRONTEND = RAIZ / "frontend"
sys.path.insert(0, str(RAIZ / "backend"))

from filmografia import (  # noqa: E402
    bucket_busqueda,
    construir_entradas_busqueda,
    exportar_indices_busqueda,
)
from rechazo import (  # noqa: E402
    UMBRALES_VOXBLINK_R100_CALIBRADOS,
    aplicar_as_norm,
    calcular_estadisticas_cohorte,
    clasificar_consulta,
    porcentaje_calibrado_r100,
)


def _extraer_script_principal() -> str:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    bloques = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert len(bloques) >= 1, "No se encontró el script principal en frontend/index.html"
    return bloques[-1]


def test_index_html_sintaxis_js_y_sin_duplicados(tmp_path: Path):
    """Comprueba sintaxis JavaScript válida (vía Node si está disponible) y ausencia de duplicados rotos."""
    script = _extraer_script_principal()

    # Comprobación estructural independiente de Node: declaraciones top-level únicas
    for patron in (
        r"^let sesion,",
        r"^const CACHE_ARTEFACTOS\s*=",
        r"^const CURVA_PORCENTAJE\s*=",
        r"^function juntarSegmentos\(",
        r"^async function papelDePista\(",
        r"^function reproducirMuestra\(",
    ):
        ocurrencias = re.findall(patron, script, flags=re.MULTILINE)
        assert len(ocurrencias) == 1, f"Declaración duplicada o ausente para {patron}: {len(ocurrencias)}"

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js no está disponible en el entorno")

    js_tmp = tmp_path / "index_principal.js"
    js_tmp.write_text(script, encoding="utf-8")
    proc = subprocess.run([node, "--check", str(js_tmp)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Error de sintaxis JS en frontend/index.html:\n{proc.stderr}"


def test_index_html_coincide_en_scoring_y_calibracion_con_backend(tmp_path: Path):
    """Ejecuta `puntuar`, `clasificarConsulta` y `aPorcentajeCalibrado` en Node y valida equivalencia con Python."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js no está disponible en el entorno")

    script = _extraer_script_principal()
    # Cortamos antes del arranque final `actualizarBotonHistorial(); cargar();` y damos stubs mínimos de DOM.
    script_sin_arranque = re.sub(r"\bactualizarBotonHistorial\(\);\s*cargar\(\);\s*$", "", script.strip())

    rng = np.random.default_rng(42)
    n_actores = 12
    dim = 16
    C = rng.standard_normal((n_actores, dim)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # 3 trozos por actor ligeramente perturbados alrededor de su centroide
    trozos_list = []
    for i in range(n_actores):
        ti = C[i] + 0.05 * rng.standard_normal((3, dim)).astype(np.float32)
        ti /= np.linalg.norm(ti, axis=1, keepdims=True)
        trozos_list.append(ti)
    V = np.vstack(trozos_list)
    inicio = np.arange(0, (n_actores + 1) * 3, 3, dtype=np.int32)
    q = (C[3] + 0.02 * rng.standard_normal(dim)).astype(np.float32)
    q /= np.linalg.norm(q)

    # Cálculo equivalente en Python (top-4 trozos ponderados por rango, 0.5*media + 0.5*mejores + AS-Norm)
    mu_a, sd_a = calcular_estadisticas_cohorte(C, k=50)
    sims_cent = C @ q
    pesos_rango = np.array([0.34, 0.27, 0.21, 0.18], dtype=np.float32)
    crudos = np.empty(n_actores, dtype=np.float32)
    for i in range(n_actores):
        t_sims = np.sort(V[inicio[i] : inicio[i + 1]] @ q)[::-1][:4]
        w_r = pesos_rango[: len(t_sims)]
        mejores = float(np.dot(t_sims, w_r) / w_r.sum())
        crudos[i] = 0.5 * (float(sims_cent[i]) + mejores)
    ajustados = aplicar_as_norm(crudos, mu_a=mu_a, sd_a=sd_a, k=50, peso=0.04)
    orden = np.argsort(-ajustados)
    py_slugs = [f"{i}-actor-{i}" for i in range(n_actores)]
    py_dec = clasificar_consulta(
        ajustados, py_slugs, UMBRALES_VOXBLINK_R100_CALIBRADOS, n_galeria_total=n_actores, con_contexto=False
    )
    py_pct = porcentaje_calibrado_r100(float(ajustados[orden[0]]))

    runner = tmp_path / "run_check.mjs"
    payload = {
        "dim": dim,
        "actores": [{"id": s, "nombre": f"Actor {i}", "trabajos": []} for i, s in enumerate(py_slugs)],
        "C": C.ravel().tolist(),
        "V": V.ravel().tolist(),
        "inicio": inicio.tolist(),
        "q": q.tolist(),
        "mu_a": mu_a.tolist(),
        "sd_a": sd_a.tolist(),
    }
    (tmp_path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")

    runner.write_text(
        """
import fs from 'node:fs';
import vm from 'node:vm';

const makeEl = () => ({
  style: {},
  dataset: {},
  classList: { toggle() {}, add() {}, remove() {} },
  addEventListener() {},
  querySelectorAll() { return []; },
  appendChild(x) { return x; },
  firstElementChild: { style: {} },
  scrollIntoView() {},
  remove() {},
  innerHTML: '',
  textContent: '',
  value: '',
  disabled: false,
});
const els = new Map();
const doc = {
  getElementById: id => (els.has(id) ? els.get(id) : (els.set(id, makeEl()), els.get(id))),
  createElement: () => makeEl(),
  documentElement: { style: { setProperty() {} } },
};
const sandbox = {
  window: { ESTAVOZ_CONFIG: {} },
  self: { crossOriginIsolated: false },
  location: { href: 'http://localhost/' },
  navigator: { hardwareConcurrency: 2 },
  document: doc,
  ort: {
    env: { wasm: {} },
    Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } },
  },
  URL,
  Float32Array,
  Uint32Array,
  Int8Array,
  Uint8Array,
  Set,
  Map,
  Math,
  JSON,
  console,
  setTimeout,
  clearTimeout,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(__SCRIPT_JSON__, sandbox);

const p = JSON.parse(fs.readFileSync(__PAYLOAD_PATH_JSON__, 'utf8'));
vm.runInContext(`
  dim = ${p.dim};
  actores = ${JSON.stringify(p.actores)};
  ldaMu = new Float32Array(dim);
  ldaW = new Float32Array(dim * dim);
  for (let i = 0; i < dim; i++) ldaW[i * dim + i] = 1;
  centroides = new Float32Array(${JSON.stringify(p.C)});
  trozos = { inicio: new Uint32Array(${JSON.stringify(p.inicio)}), v: new Float32Array(${JSON.stringify(p.V)}) };
  cohorteMu = new Float32Array(${JSON.stringify(p.mu_a)});
  cohorteSd = new Float32Array(${JSON.stringify(p.sd_a)});
  umbrales = UMBRALES_DEFECTO;
  const q = new Float32Array(${JSON.stringify(p.q)});
  q.durPeso = 1;
  const top = puntuar([q], 3, null);
  const seg = juntarSegmentos(new Float32Array(3200), [
    { iniM: 0, finM: 800, dur: 0.05 },
    { iniM: 1200, finM: 2400, dur: 0.075 },
  ]);
  const vadSilencio = analizarVAD(new Float32Array(16000 * 3), 16000);
  const sinVoz = new Float32Array(16000 * 5);
  for (let i = 16000; i < 16000 * 4; i++) {
    const t = i / 16000;
    const mod = 0.5 * (1 + Math.sin(2 * Math.PI * 4 * t));
    sinVoz[i] = mod * (0.5 * Math.sin(2 * Math.PI * 150 * t) + 0.3 * Math.sin(2 * Math.PI * 300 * t));
  }
  const vadVoz = analizarVAD(sinVoz, 16000);
  const sinContinua = new Float32Array(16000 * 3);
  for (let i = 0; i < sinContinua.length; i++) {
    const t = i / 16000;
    const mod = 0.7 + 0.3 * Math.sin(2 * Math.PI * 4 * t);
    sinContinua[i] = 0.35 * mod * (0.6 * Math.sin(2 * Math.PI * 150 * t) + 0.4 * Math.sin(2 * Math.PI * 300 * t));
  }
  const vadContinuo = analizarVAD(sinContinua, 16000);

  const ruidoBlanco = new Float32Array(16000 * 3);
  let seed = 1337;
  for (let i = 0; i < ruidoBlanco.length; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    ruidoBlanco[i] = ((seed / 4294967296) * 2 - 1) * 0.2;
  }
  const vadRuido = analizarVAD(ruidoBlanco, 16000);

  const genOnda = (f0) => {
    const out = new Float32Array(16000);
    for (let i = 0; i < out.length; i++) {
      const t = i / 16000;
      out[i] = 0.6 * Math.sin(2 * Math.PI * f0 * t) + 0.4 * Math.sin(4 * Math.PI * f0 * t);
    }
    return out;
  };
  const f0110 = estimarF0(genOnda(110), 16000);
  const f0220 = estimarF0(genOnda(220), 16000);

  actores[top[0]._idx].f0 = 260;
  const topF0 = puntuar([q], 3, null, 100);
  delete actores[top[0]._idx].f0;
  const penalizadoF0 = topF0.find(x => x._idx === top[0]._idx);

  const actorDemo = {
    ...top[0],
    num_trabajos: 12,
    trabajos: ['Buzz Lightyear en TOY STORY [Animación] (1995)'],
    resumen_tipos: { animacion: 1 },
    vozh: [{ nombre: 'Tim Allen', n: 5 }],
  };
  const listaDemo = [actorDemo, ...top.slice(1)];
  listaDemo.estado = 'identificado';
  ultimaLista = listaDemo;
  ultimoAudioActivo = new Float32Array(1600);
  ultimoDiagnostico = { duracionVoz: 4.5, medVoc: 0.40, snrDb: 12, f0: 112 };
  pintar(listaDemo);
  const htmlIdent = res.innerHTML;
  ultimoDiagnostico = null;

  const listaDesc = [...top];
  listaDesc.estado = 'desconocido';
  pintar(listaDesc);
  const htmlDesc = res.innerHTML;

  verCandidato(top[1]._idx);
  const htmlCandidato = res.innerHTML;

  inspeccionarDoblador(1);
  const htmlCatalogo = res.innerHTML;

  location.hash = '#actor=' + actores[2].id;
  abrirEnlaceInicial();
  const htmlEnlace = res.innerHTML;
  location.hash = '';

  const qAlt = new Float32Array(dim);
  qAlt[0] = 1;
  const dialDemo = { actor1: top[0], actor2: top[1], dur1: 3.0, dur2: 2.5, consulta1: [q], consulta2: [qAlt], f01: 140, f02: 210, activo: 1 };
  rescorarDialogoConPista(dialDemo, [2, 5]);

  const sigImp = new Float32Array(3000);
  let sqBase = 0;
  for (let i = 0; i < sigImp.length; i++) {
    sigImp[i] = 0.08 * Math.sin(2 * Math.PI * 160 * (i / 16000));
    if (i !== 500) sqBase += sigImp[i] * sigImp[i];
  }
  const rmsLinealImp = Math.sqrt(sqBase / (sigImp.length - 1)) * (0.75 / 0.98);
  sigImp[500] = 0.98;
  const normImp = normalizarPico(sigImp, 0.75);
  let sqNorm = 0;
  for (let i = 0; i < normImp.length; i++) if (i !== 500) sqNorm += normImp[i] * normImp[i];
  const rmsNormImp = Math.sqrt(sqNorm / (normImp.length - 1));
  const normImpGananciaRatio = rmsNormImp / Math.max(rmsLinealImp, 1e-6);

  const combInter = combinarPistas([
    { texto: 'Toy Story', ids: [1, 2, 5] },
    { texto: 'Los Simpson', ids: [2, 5, 8] },
  ]);
  const combUnion = combinarPistas([
    { texto: 'Obra A', ids: [1, 3] },
    { texto: 'Obra B', ids: [4, 7] },
  ]);
  const vozRealzada = realzarVozLejana(sinVoz, 16000);
  const listaConGemelo = [...listaDemo];
  listaConGemelo.estado = 'identificado';
  listaConGemelo.gemeloVocal = { id: top[1].id, nombre: top[1].nombre, _idx: top[1]._idx, simPct: 71 };
  ultimaLista = listaConGemelo;
  ultimoAudioOriginal = new Float32Array(16000 * 8);
  recorteIniSeg = 1.0;
  recorteFinSeg = 6.0;
  onCambioRecorte('ini', 1.5);
  onCambioRecorte('fin', 5.5);
  const tramoRec = obtenerAudioRecortado();
  alternarCaraACara(top[1]._idx);
  const htmlGemelo = res.innerHTML;
  const cvTarjeta = generarTarjetaCanvas(actorDemo, 88);

  const bucketJs = bucketBusqueda('to');
  globalThis.__out = {
    id: top[0].id,
    parecido: top[0].parecido,
    estado: top.estado,
    pct: aPorcentajeCalibrado(top[0].parecido),
    segLen: seg.length,
    vadSilencioOk: vadSilencio.esValida,
    vadSilencioMotivo: vadSilencio.motivo,
    vadVozOk: vadVoz.esValida,
    vadContinuoOk: vadContinuo.esValida,
    vadRuidoOk: vadRuido.esValida,
    vadRuidoMotivo: vadRuido.motivo,
    f0110,
    f0220,
    f0PenalizadoOk: Boolean(penalizadoF0 && penalizadoF0.parecido < top[0].parecido - 0.02),
    tieneFicha: htmlIdent.includes('estado-identificado') && htmlIdent.includes('medidor') && htmlIdent.includes('Oír mi grabación') && htmlIdent.includes('btn-muestra mini'),
    tieneDiagnostico: htmlIdent.includes('diagnostico-acustico') && htmlIdent.includes('Timbre Grave (~112 Hz)') && htmlIdent.includes('Música o ruido de fondo'),
    tieneRechazoLimpio: htmlDesc.includes('estado-desconocido') && htmlDesc.includes('No reconozco esta voz con seguridad'),
    tieneCandidatoAlternativo: htmlCandidato.includes('CANDIDATO') && htmlCandidato.includes('VOLVER AL PRINCIPAL'),
    tieneFichaCatalogo: htmlCatalogo.includes('CATÁLOGO') && htmlCatalogo.includes('Actor 1') && !htmlCatalogo.includes('¿ACERTAMOS?'),
    abreEnlaceActorOk: htmlEnlace.includes('CATÁLOGO') && htmlEnlace.includes(actores[2].nombre),
    dialRescoradoOk: [2, 5].includes(dialDemo.actor1._idx) && [2, 5].includes(dialDemo.actor2._idx),
    normImpGananciaRatio,
    combInterOk: combInter.tipo === 'interseccion' && JSON.stringify(combInter.ids) === '[2,5]',
    combUnionOk: combUnion.tipo === 'union' && combUnion.vacia === true && combUnion.ids.length === 4,
    realzarOk: vozRealzada.length === sinVoz.length,
    tieneAvisoGemelo: htmlGemelo.includes('aviso-gemelo') && htmlGemelo.includes('71% de similitud vocal'),
    tieneCaraACara: htmlGemelo.includes('CARA A CARA') && htmlGemelo.includes('CANDIDATO #1') && htmlGemelo.includes('CANDIDATO #2'),
    tieneRecorteUI: htmlGemelo.includes('panelRecorte') && tramoRec.length === 16000 * 4,
    tieneObraFiltrable: htmlIdent.includes('class="filtrable"') && htmlIdent.includes('filtrarPorObraDesdeFicha') && htmlIdent.includes('Filtrar ↗'),
    tieneAccionesUtil: htmlIdent.includes('acciones-util') && htmlIdent.includes('Tarjeta PNG') && htmlIdent.includes('aria-expanded="false"'),
    tarjetaOk: Boolean(cvTarjeta && cvTarjeta.resumenDatos && cvTarjeta.resumenDatos.pct === 88 && cvTarjeta.resumenDatos.numRoles === 1),
    tipoAnim: clasificarObraJs('TOY STORY [Animación]'),
    origFmt: nombreOriginalJs('Bale, Christian'),
    bucketTo: bucketJs,
  };
`, sandbox);
await vm.runInContext(`(async () => {
  const largos = [];
  sesion = { run: async ({ wav }) => { largos.push(wav.data.length); return { huella: { data: new Float32Array(dim).fill(1) } }; } };
  await huellasDe(new Float32Array(16000 * 9));
  globalThis.__out.ventanas9s = largos;
  busquedaTop = Promise.resolve(mapearEntradas([['obra', 'Toy Story', [0, 1]]]));
  usaMonolitoBusqueda = true;
  const sugsDoblador = await sugerencias('Actor 1');
  globalThis.__out.buscaDobladorOk = sugsDoblador.some(s => s.tipo === 'doblador' && s.texto === 'Actor 1');
  await filtrarPorObraDesdeFicha('TOY STORY [Animación]');
  globalThis.__out.filtraPorObraOk = Boolean(pistaElegida && pistaElegida.texto === 'Toy Story');
})()`, sandbox);
console.log(JSON.stringify(sandbox.__out));
"""
        .replace("__SCRIPT_JSON__", json.dumps(script_sin_arranque))
        .replace("__PAYLOAD_PATH_JSON__", json.dumps(str(tmp_path / "payload.json"))),
        encoding="utf-8",
    )

    proc = subprocess.run([node, str(runner)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Fallo ejecutando JS de index.html:\n{proc.stderr}"
    out = json.loads(proc.stdout.strip())
    assert out["id"] == py_slugs[int(orden[0])]
    assert abs(out["parecido"] - float(ajustados[orden[0]])) < 1e-4
    assert out["estado"] == py_dec.estado.value
    assert out["pct"] == py_pct
    assert out["segLen"] == 2000
    assert out["vadSilencioOk"] is False
    assert out["vadSilencioMotivo"] == "silencio"
    assert out["vadVozOk"] is True
    assert out["vadContinuoOk"] is True
    assert out["vadRuidoOk"] is False
    assert out["vadRuidoMotivo"] in ("musica_ruido", "corto")
    assert out["f0110"] is not None and abs(out["f0110"] - 110.0) < 8.0
    assert out["f0220"] is not None and abs(out["f0220"] - 220.0) < 12.0
    assert out["f0PenalizadoOk"] is True
    assert out["ventanas9s"] == [16000 * 6, 16000 * 6]
    assert out["tieneFicha"] is True
    assert out["tieneDiagnostico"] is True
    assert out["tieneRechazoLimpio"] is True
    assert out["tieneCandidatoAlternativo"] is True
    assert out["tieneFichaCatalogo"] is True
    assert out["abreEnlaceActorOk"] is True
    assert out["dialRescoradoOk"] is True
    assert out["normImpGananciaRatio"] >= 2.5
    assert out["combInterOk"] is True
    assert out["combUnionOk"] is True
    assert out["realzarOk"] is True
    assert out["tieneAvisoGemelo"] is True
    assert out["tieneCaraACara"] is True
    assert out["tieneRecorteUI"] is True
    assert out["tieneObraFiltrable"] is True
    assert out["tieneAccionesUtil"] is True
    assert out["tarjetaOk"] is True
    assert out["buscaDobladorOk"] is True
    assert out["filtraPorObraOk"] is True
    assert out["tipoAnim"] == "animacion"
    assert out["origFmt"] == "Christian Bale"
    assert out["bucketTo"] == bucket_busqueda("to")


def test_busqueda_fragmentada_prioriza_obras_y_encuentra_en_shards(tmp_path: Path):
    """Comprueba ordenación (obras/actores antes que nombres genéricos) y búsqueda por shards."""
    filas_por_actor = [
        [["TOY STORY [Animación]", "Buzz Lightyear", 1995, "Allen, Tim"], ["EL CLUB", "Sam", 2001, ""]],
        [["TOY STORY [Animación]", "Woody", 1995, "Hanks, Tom"], ["OTRA", "Sam", 2002, ""]],
    ] + [[["SERIE X", "Sam", 2000 + i, ""]] for i in range(10)]

    entradas = construir_entradas_busqueda(filas_por_actor)
    # Toy Story (obra) debe ir por delante del personaje genérico Sam aunque Sam tenga 12 actores
    tipos_ordenados = [e[0] for e in entradas]
    assert tipos_ordenados.index("obra") < [e[1] for e in entradas].index("Sam")

    res = exportar_indices_busqueda(entradas, tmp_path, top_n=1, n_shards=64)
    assert res["top"] == 1
    assert (tmp_path / "busqueda_top.json").exists()
    assert (tmp_path / "busqueda" / f"{bucket_busqueda('bu'):02d}.json").exists()

    # Verificar que "Buzz Lightyear" (fuera del top_n=1) cayó en su shard exacto
    shard_bu = json.loads((tmp_path / "busqueda" / f"{bucket_busqueda('bu'):02d}.json").read_text(encoding="utf-8"))
    assert any(e[1] == "Buzz Lightyear" for e in shard_bu)

    # Verificar también en Node que sugerencias() de index.html recupera resultados de top + shard
    node = shutil.which("node")
    if node is None:
        return
    script = _extraer_script_principal()
    script_sin_arranque = re.sub(r"\bactualizarBotonHistorial\(\);\s*cargar\(\);\s*$", "", script.strip())
    runner = tmp_path / "check_search.mjs"
    runner.write_text(
        """
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const baseDir = __BASE_DIR__;
const makeEl = () => ({
  style: {}, dataset: {}, classList: { toggle() {}, add() {}, remove() {} },
  addEventListener() {}, querySelectorAll() { return []; }, appendChild(x) { return x; },
  firstElementChild: { style: {} }, scrollIntoView() {}, remove() {},
  innerHTML: '', textContent: '', value: '', disabled: false,
});
const els = new Map();
const sandbox = {
  window: { ESTAVOZ_CONFIG: {} },
  self: { crossOriginIsolated: false },
  location: { href: 'http://localhost/' },
  navigator: { hardwareConcurrency: 2 },
  document: {
    getElementById: id => (els.has(id) ? els.get(id) : (els.set(id, makeEl()), els.get(id))),
    createElement: () => makeEl(),
    documentElement: { style: { setProperty() {} } },
  },
  ort: { env: { wasm: {} } },
  fetch: async (url) => {
    const rel = url.replace(/^datos\\//, '');
    const p = path.join(baseDir, rel);
    const txt = fs.readFileSync(p, 'utf8');
    return { ok: true, status: 200, text: async () => txt, json: async () => JSON.parse(txt) };
  },
  URL, Float32Array, Uint32Array, Int8Array, Uint8Array, Set, Map, Math, JSON, console, setTimeout, clearTimeout,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(__SCRIPT__, sandbox);
const res = await vm.runInContext("sugerencias('buzz')", sandbox);
console.log(JSON.stringify(res.map(x => [x.tipo, x.texto])));
"""
        .replace("__BASE_DIR__", json.dumps(str(tmp_path)))
        .replace("__SCRIPT__", json.dumps(script_sin_arranque)),
        encoding="utf-8",
    )
    proc = subprocess.run([node, str(runner)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    encontrados = json.loads(proc.stdout.strip())
    assert ["personaje", "Buzz Lightyear"] in encontrados


def test_privacidad_y_manifest_actualizados():
    """Verifica coherencia de privacidad.html (256 dimensiones) y manifest.json."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in html
    priv = (FRONTEND / "privacidad.html").read_text(encoding="utf-8")
    assert "192 números" not in priv
    assert "256 números" in priv

    manifest = json.loads((FRONTEND / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("name") == "Estavoz"
    for icono in manifest.get("icons", []):
        assert (FRONTEND / icono["src"]).exists(), f"Falta el icono {icono['src']}"


def test_supabase_sql_y_responder_con_respaldo_esquema(tmp_path: Path):
    """Valida el SQL idempotente de Supabase y el envío/respaldo HTTP 400 de `responder()`."""
    sql = (RAIZ / "despliegue" / "supabase.sql").read_text(encoding="utf-8").lower()
    assert "create table if not exists respuestas" in sql
    for col in ("version", "estado", "parecido", "f0", "tomas", "con_pista"):
        assert f"add column if not exists {col}" in sql
    assert "create index if not exists idx_respuestas_version_id" in sql
    assert "create index if not exists idx_respuestas_creado" in sql
    assert "security_invoker = true" in sql
    assert "grant insert on table respuestas to anon" in sql

    node = shutil.which("node")
    if node is None:
        return
    script = _extraer_script_principal()
    script_sin_arranque = re.sub(r"\bactualizarBotonHistorial\(\);\s*cargar\(\);\s*$", "", script.strip())
    runner = tmp_path / "check_supabase.mjs"
    runner.write_text(
        """
import vm from 'node:vm';

const makeEl = () => ({
  style: {}, dataset: {}, classList: { toggle() {}, add() {}, remove() {} },
  addEventListener() {}, querySelectorAll() { return []; }, appendChild(x) { return x; },
  firstElementChild: { style: {} }, scrollIntoView() {}, remove() {},
  innerHTML: '', textContent: '', value: '', disabled: false,
});
const els = new Map();
const llamadas = [];
let simular400Primero = false;

const sandbox = {
  window: { ESTAVOZ_CONFIG: { supabaseUrl: 'https://ejemplo.supabase.co/', supabaseKey: 'anon-key' } },
  self: { crossOriginIsolated: false },
  location: { href: 'http://localhost/' },
  navigator: { hardwareConcurrency: 2 },
  document: {
    getElementById: id => (els.has(id) ? els.get(id) : (els.set(id, makeEl()), els.get(id))),
    createElement: () => makeEl(),
    documentElement: { style: { setProperty() {} } },
  },
  ort: { env: { wasm: {} } },
  fetch: async (_url, opts) => {
    const body = JSON.parse(opts.body);
    llamadas.push(body);
    if (simular400Primero && 'estado' in body) {
      return { ok: false, status: 400 };
    }
    return { ok: true, status: 201 };
  },
  URL, Float32Array, Uint32Array, Int8Array, Uint8Array, Set, Map, Math, JSON, console, setTimeout, clearTimeout,
};
sandbox.window = sandbox;
sandbox.ESTAVOZ_CONFIG = { supabaseUrl: 'https://ejemplo.supabase.co', supabaseKey: 'anon-key' };
vm.createContext(sandbox);
vm.runInContext(__SCRIPT__, sandbox);

vm.runInContext(`
  ultimo = { id: '1005-constantino-romero', parecido: 0.61234 };
  ultimoEstadoResultado = 'identificado';
  ultimaF0 = 112.46;
  tomasAcumuladas = 2;
  ultimaPista = { texto: 'Star Wars', n: 12 };
  ultimaConsulta = [new Float32Array([0.12345, -0.5])];
`, sandbox);
await vm.runInContext("responder(true, 'ignorado-en-acierto')", sandbox);

simular400Primero = true;
await vm.runInContext("responder(false, '200-otro-actor')", sandbox);

console.log(JSON.stringify(llamadas));
""".replace("__SCRIPT__", json.dumps(script_sin_arranque)),
        encoding="utf-8",
    )
    proc = subprocess.run([node, str(runner)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    llamadas = json.loads(proc.stdout.strip())
    assert len(llamadas) == 3  # 1ª llamada OK (201) + 2ª llamada (400 ampliado -> retry 201 base)
    assert llamadas[0]["propuesto"] == "1005-constantino-romero"
    assert llamadas[0]["acierto"] is True
    assert llamadas[0]["real"] is None
    assert llamadas[0]["version"] == "3"
    assert llamadas[0]["estado"] == "identificado"
    assert abs(llamadas[0]["parecido"] - 0.6123) < 1e-4
    assert abs(llamadas[0]["f0"] - 112.5) < 1e-2
    assert llamadas[0]["tomas"] == 2
    assert llamadas[0]["con_pista"] is True

    assert "estado" in llamadas[1]
    assert set(llamadas[2].keys()) == {"propuesto", "acierto", "real", "huellas", "version"}
    assert llamadas[2]["real"] == "200-otro-actor"
