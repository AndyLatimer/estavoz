"""Identificación inequívoca de modelos de voz, proyecciones LDA y validación de compatibilidad.

Resuelve la ambigüedad entre arquitecturas que comparten dimensión de salida (p. ej. VoxBlink2
SimAM-ResNet34 y SimAM-ResNet100 producen ambas embeddings de 256 números y se han exportado con
el mismo nombre de archivo `voxblink-int8.onnx`) o caídas silenciosas a ECAPA-TDNN (192 números).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VERSION_LDA_ACTUAL = "1.0"

# Catálogo canónico de modelos soportados por EstaVoz
CATALOGO_MODELOS: dict[str, dict[str, Any]] = {
    "1": {
        "version": "1",
        "arquitectura": "speechbrain-ecapa-tdnn",
        "variante": "ecapa",
        "dim_embedding": 192,
        "bloques": (),
        "conv_esperadas_min": 10,
        "conv_esperadas_max": 32,
        "trozo_indexado_seg": 6,
    },
    "2": {
        "version": "2",
        "arquitectura": "voxblink2-simam-resnet34",
        "variante": "r34",
        "dim_embedding": 256,
        "bloques": (3, 4, 6, 3),
        # 1 (fbank) + 1 (conv1) + 2*(3+4+6+3)=32 + 3 (downsample) + 2 (ASP) = 39 Conv
        "conv_esperadas_min": 33,
        "conv_esperadas_max": 60,
        "trozo_indexado_seg": 6,
    },
    "3": {
        "version": "3",
        "arquitectura": "voxblink2-simam-resnet100",
        "variante": "r100",
        "dim_embedding": 256,
        "bloques": (6, 16, 24, 3),
        # 1 (fbank) + 1 (conv1) + 2*(6+16+24+3)=98 + 3 (downsample) + 2 (ASP) = 105 Conv
        "conv_esperadas_min": 80,
        "conv_esperadas_max": 140,
        "trozo_indexado_seg": 6,
    },
}

VARIANTE_A_VERSION = {info["variante"]: ver for ver, info in CATALOGO_MODELOS.items()}
ARQUITECTURA_A_VERSION = {info["arquitectura"]: ver for ver, info in CATALOGO_MODELOS.items()}


class IncompatibilidadModeloError(ValueError):
    """Error lanzado cuando dos artefactos (ONNX, índice, LDA o datos web) son incompatibles."""


@dataclass(frozen=True)
class InfoModelo:
    version: str
    arquitectura: str
    variante: str
    dim_embedding: int
    bloques: tuple[int, ...] = ()
    sha256_onnx: str | None = None
    tamano_bytes: int | None = None
    num_nodos_conv: int | None = None
    origen_deteccion: str = "catalogo"

    def a_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bloques"] = list(self.bloques)
        return d

    @classmethod
    def desde_dict(cls, d: dict[str, Any]) -> InfoModelo:
        return cls(
            version=str(d["version"]),
            arquitectura=str(d["arquitectura"]),
            variante=str(d["variante"]),
            dim_embedding=int(d["dim_embedding"]),
            bloques=tuple(int(x) for x in d.get("bloques", ())),
            sha256_onnx=d.get("sha256_onnx"),
            tamano_bytes=int(d["tamano_bytes"]) if d.get("tamano_bytes") is not None else None,
            num_nodos_conv=int(d["num_nodos_conv"]) if d.get("num_nodos_conv") is not None else None,
            origen_deteccion=str(d.get("origen_deteccion", "dict")),
        )


@dataclass(frozen=True)
class InfoLDA:
    version_lda: str
    dim_entrada: int
    dim_salida: int
    reg: float
    peso_entrevistas: float
    num_actores_entreno: int
    num_trozos_entreno: int
    sha256_lda: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_archivo(ruta: str | Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    leidos = 0
    with open(ruta, "rb") as f:
        while True:
            bloque = f.read(1 << 20)
            if not bloque:
                break
            if max_bytes is not None and leidos + len(bloque) > max_bytes:
                h.update(bloque[: max_bytes - leidos])
                break
            h.update(bloque)
            leidos += len(bloque)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def info_por_version(version: str) -> InfoModelo:
    v = str(version)
    if v not in CATALOGO_MODELOS:
        raise IncompatibilidadModeloError(
            f"Versión de modelo desconocida: {version!r}. Versiones válidas: {sorted(CATALOGO_MODELOS)}"
        )
    c = CATALOGO_MODELOS[v]
    return InfoModelo(
        version=c["version"],
        arquitectura=c["arquitectura"],
        variante=c["variante"],
        dim_embedding=c["dim_embedding"],
        bloques=tuple(c["bloques"]),
        origen_deteccion="catalogo",
    )


def info_por_variante(variante: str) -> InfoModelo:
    if variante not in VARIANTE_A_VERSION:
        raise IncompatibilidadModeloError(
            f"Variante de modelo desconocida: {variante!r}. Variantes válidas: {sorted(VARIANTE_A_VERSION)}"
        )
    return info_por_version(VARIANTE_A_VERSION[variante])


def inscribir_metadatos_onnx(ruta_onnx: str | Path, info: InfoModelo) -> None:
    """Escribe metadatos identificativos dentro del propio archivo ONNX (`metadata_props`)."""
    import onnx

    modelo = onnx.load(str(ruta_onnx))
    props = {
        "estavoz.version": str(info.version),
        "estavoz.arquitectura": str(info.arquitectura),
        "estavoz.variante": str(info.variante),
        "estavoz.dim_embedding": str(info.dim_embedding),
        "estavoz.bloques": json.dumps(list(info.bloques)),
    }
    existentes = {p.key: p for p in modelo.metadata_props}
    for k, v in props.items():
        if k in existentes:
            existentes[k].value = v
        else:
            entrada = modelo.metadata_props.add()
            entrada.key = k
            entrada.value = v
    onnx.save(modelo, str(ruta_onnx))


def _inspeccionar_onnx_sin_libreria(ruta: Path, digest: str | None, tamano: int) -> InfoModelo:
    """Respaldo ligero para entornos sin el paquete `onnx` (p. ej. GitHub Actions nocturno)."""
    import re

    cabecera = ruta.read_bytes()
    # Buscar propiedades `estavoz.*` codificadas en el protobuf si existen
    m_ver = re.search(rb"estavoz\.version[\x00-\x20]+([123])", cabecera)
    if m_ver:
        base = info_por_version(m_ver.group(1).decode("ascii"))
        return InfoModelo(
            version=base.version,
            arquitectura=base.arquitectura,
            variante=base.variante,
            dim_embedding=base.dim_embedding,
            bloques=base.bloques,
            sha256_onnx=digest,
            tamano_bytes=tamano,
            num_nodos_conv=None,
            origen_deteccion="protobuf_bytes",
        )
    if tamano >= 40_000_000:
        base = info_por_version("3")
    elif tamano >= 25_000_000:
        base = info_por_version("2")
    elif tamano >= 15_000_000:
        base = info_por_version("1")
    else:
        raise IncompatibilidadModeloError(
            f"No se puede determinar la arquitectura de {ruta.name} ({tamano} bytes) sin la librería `onnx`."
        )
    return InfoModelo(
        version=base.version,
        arquitectura=base.arquitectura,
        variante=base.variante,
        dim_embedding=base.dim_embedding,
        bloques=base.bloques,
        sha256_onnx=digest,
        tamano_bytes=tamano,
        num_nodos_conv=None,
        origen_deteccion="tamano_archivo",
    )


def inspeccionar_onnx(ruta_onnx: str | Path, calcular_hash: bool = True) -> InfoModelo:
    """Inspecciona un archivo ONNX real para determinar su arquitectura, versión y dimensión.

    Distingue inequívocamente entre:
      - ECAPA-TDNN (dim=192)
      - VoxBlink2 SimAM-ResNet34 (dim=256, ~39 nodos Conv/ConvInteger)
      - VoxBlink2 SimAM-ResNet100 (dim=256, ~105 nodos Conv/ConvInteger)
    incluso si el archivo fue renombrado a `voxblink-int8.onnx` sin metadatos previos.
    """
    ruta = Path(ruta_onnx)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo ONNX: {ruta}")

    tamano = ruta.stat().st_size
    digest = sha256_archivo(ruta) if calcular_hash else None

    try:
        import onnx
    except ImportError:
        return _inspeccionar_onnx_sin_libreria(ruta, digest, tamano)

    modelo = onnx.load(str(ruta), load_external_data=False)
    props = {p.key: p.value for p in modelo.metadata_props}
    nodos_conv = sum(1 for n in modelo.graph.node if n.op_type in ("Conv", "ConvInteger", "QLinearConv"))

    # Dimensión de salida según el grafo ONNX
    dim_grafo: int | None = None
    if modelo.graph.output:
        tipo_tensor = modelo.graph.output[0].type.tensor_type
        if tipo_tensor.HasField("shape") and len(tipo_tensor.shape.dim) >= 1:
            ultima = tipo_tensor.shape.dim[-1]
            if ultima.HasField("dim_value") and ultima.dim_value > 0:
                dim_grafo = int(ultima.dim_value)

    # Si no figura en la forma estática de salida, buscar la última matriz Gemm/MatMul (bottleneck)
    if dim_grafo is None:
        init_shapes = {init.name: tuple(init.dims) for init in modelo.graph.initializer}
        for n in reversed(modelo.graph.node):
            if n.op_type in ("Gemm", "MatMul", "MatMulInteger"):
                for inp in n.input:
                    if inp in init_shapes and len(init_shapes[inp]) == 2:
                        # En Gemm (transB=1) el peso es (dim, in_features); en MatMul (in_features, dim)
                        s0, s1 = init_shapes[inp]
                        if s0 in (192, 256):
                            dim_grafo = int(s0)
                            break
                        if s1 in (192, 256):
                            dim_grafo = int(s1)
                            break
            if dim_grafo is not None:
                break

    # 1. Si el ONNX tiene metadatos explícitos de EstaVoz, verificar que coincidan con la estructura real
    if "estavoz.version" in props:
        ver = props["estavoz.version"]
        base = info_por_version(ver)
        dim_meta = int(props.get("estavoz.dim_embedding", base.dim_embedding))
        if dim_grafo is not None and dim_grafo != dim_meta:
            raise IncompatibilidadModeloError(
                f"El archivo ONNX {ruta.name} declara dim={dim_meta} en metadatos pero su grafo produce dim={dim_grafo}"
            )
        cat = CATALOGO_MODELOS[ver]
        if nodos_conv > 0 and not (cat["conv_esperadas_min"] <= nodos_conv <= cat["conv_esperadas_max"]):
            raise IncompatibilidadModeloError(
                f"El archivo ONNX {ruta.name} declara versión {ver} ({base.arquitectura}), pero contiene "
                f"{nodos_conv} nodos Conv (esperados entre {cat['conv_esperadas_min']} y {cat['conv_esperadas_max']})"
            )
        return InfoModelo(
            version=ver,
            arquitectura=props.get("estavoz.arquitectura", base.arquitectura),
            variante=props.get("estavoz.variante", base.variante),
            dim_embedding=dim_grafo or dim_meta,
            bloques=tuple(json.loads(props["estavoz.bloques"])) if "estavoz.bloques" in props else base.bloques,
            sha256_onnx=digest,
            tamano_bytes=tamano,
            num_nodos_conv=nodos_conv,
            origen_deteccion="metadata_props",
        )

    # 2. Inferencia estructural por dimensión de salida y conteo de capas convolucionales
    if dim_grafo == 192:
        base = info_por_version("1")
        return InfoModelo(
            version=base.version,
            arquitectura=base.arquitectura,
            variante=base.variante,
            dim_embedding=192,
            bloques=base.bloques,
            sha256_onnx=digest,
            tamano_bytes=tamano,
            num_nodos_conv=nodos_conv,
            origen_deteccion="estructura_grafo",
        )

    if dim_grafo == 256:
        if nodos_conv >= CATALOGO_MODELOS["3"]["conv_esperadas_min"]:
            base = info_por_version("3")
        elif nodos_conv >= CATALOGO_MODELOS["2"]["conv_esperadas_min"]:
            base = info_por_version("2")
        elif tamano >= 40_000_000:
            base = info_por_version("3")
        elif tamano >= 20_000_000:
            base = info_por_version("2")
        else:
            raise IncompatibilidadModeloError(
                f"No se puede determinar si {ruta.name} (dim=256, conv={nodos_conv}, bytes={tamano}) "
                "es ResNet34 o ResNet100. Inscriba metadatos con inscribir_metadatos_onnx()."
            )
        return InfoModelo(
            version=base.version,
            arquitectura=base.arquitectura,
            variante=base.variante,
            dim_embedding=256,
            bloques=base.bloques,
            sha256_onnx=digest,
            tamano_bytes=tamano,
            num_nodos_conv=nodos_conv,
            origen_deteccion="estructura_grafo",
        )

    raise IncompatibilidadModeloError(
        f"El archivo ONNX {ruta.name} tiene dimensión de salida no reconocida ({dim_grafo}) y carece de metadatos."
    )


def extraer_info_indice(datos_npz: Any) -> dict[str, Any]:
    """Extrae metadatos del archivo `data/indice.npz` (compatible con índices antiguos y nuevos)."""
    archivos = set(datos_npz.files)
    X = datos_npz["vectores"]
    y = datos_npz["etiquetas"]
    clips = datos_npz["clips"]
    if X.ndim != 2:
        raise IncompatibilidadModeloError(f"`vectores` en el índice debe tener 2 dimensiones, tiene {X.ndim}")
    if len(X) != len(y) or len(X) != len(clips):
        raise IncompatibilidadModeloError(
            f"Longitudes incoherentes en el índice: vectores={len(X)}, etiquetas={len(y)}, clips={len(clips)}"
        )

    dim = int(X.shape[1])
    version = str(datos_npz["modelo_version"].item()) if "modelo_version" in archivos else None
    arquitectura = str(datos_npz["modelo_arquitectura"].item()) if "modelo_arquitectura" in archivos else None
    variante = str(datos_npz["modelo_variante"].item()) if "modelo_variante" in archivos else None
    sha256_onnx = str(datos_npz["modelo_sha256"].item()) if "modelo_sha256" in archivos else None

    if version is not None and version in CATALOGO_MODELOS:
        esperada = CATALOGO_MODELOS[version]["dim_embedding"]
        if dim != esperada:
            raise IncompatibilidadModeloError(
                f"El índice declara modelo_version={version} (dim esperada {esperada}), "
                f"pero sus vectores tienen dimensión {dim}."
            )

    return {
        "dim_embedding": dim,
        "num_trozos": int(len(X)),
        "num_actores": int(len(set(y))),
        "separado": bool(datos_npz["separado"]) if "separado" in archivos else False,
        "modelo_version": version,
        "modelo_arquitectura": arquitectura,
        "modelo_variante": variante,
        "modelo_sha256": sha256_onnx or None,
        "tiene_metadatos_modelo": version is not None,
    }


def validar_compatibilidad_indice(
    datos_npz: Any,
    info_modelo: InfoModelo | None = None,
    version_esperada: str | None = None,
    estricto: bool = False,
) -> dict[str, Any]:
    """Verifica que `indice.npz` sea compatible con el modelo ONNX activo y/o la versión esperada.

    Si `estricto=True`, exige que el índice contenga metadatos explícitos de modelo y que coincida
    el hash SHA-256 del ONNX si ambos están presentes.
    """
    meta = extraer_info_indice(datos_npz)
    dim_indice = meta["dim_embedding"]

    if version_esperada is not None:
        base_esp = info_por_version(version_esperada)
        if dim_indice != base_esp.dim_embedding:
            raise IncompatibilidadModeloError(
                f"Incompatibilidad de dimensiones: el índice tiene dim={dim_indice}, pero la versión "
                f"esperada {version_esperada} ({base_esp.arquitectura}) requiere dim={base_esp.dim_embedding}."
            )
        if meta["modelo_version"] is not None and meta["modelo_version"] != str(version_esperada):
            raise IncompatibilidadModeloError(
                f"Incompatibilidad de versión de modelo: el índice fue generado con versión "
                f"{meta['modelo_version']} ({meta['modelo_arquitectura']}), pero se esperaba "
                f"versión {version_esperada} ({base_esp.arquitectura})."
            )

    if info_modelo is not None:
        if dim_indice != info_modelo.dim_embedding:
            raise IncompatibilidadModeloError(
                f"Incompatibilidad entre índice (dim={dim_indice}) y modelo ONNX {info_modelo.arquitectura} "
                f"(dim={info_modelo.dim_embedding})."
            )
        if meta["modelo_version"] is not None and meta["modelo_version"] != info_modelo.version:
            raise IncompatibilidadModeloError(
                f"Incompatibilidad crítica: el índice fue creado con modelo versión {meta['modelo_version']} "
                f"({meta['modelo_arquitectura']}), pero el modelo ONNX es versión {info_modelo.version} "
                f"({info_modelo.arquitectura}). Ambos tienen dim={dim_indice} pero sus espacios son incompatibles."
            )
        if meta["modelo_variante"] is not None and meta["modelo_variante"] != info_modelo.variante:
            raise IncompatibilidadModeloError(
                f"Incompatibilidad de variante: el índice es {meta['modelo_variante']} y el ONNX es {info_modelo.variante}."
            )
        if (
            estricto
            and meta["modelo_sha256"]
            and info_modelo.sha256_onnx
            and meta["modelo_sha256"] != info_modelo.sha256_onnx
        ):
            raise IncompatibilidadModeloError(
                f"El hash SHA-256 del modelo ONNX ({info_modelo.sha256_onnx[:12]}...) no coincide con el "
                f"registrado en el índice ({meta['modelo_sha256'][:12]}...)."
            )

    if estricto and not meta["tiene_metadatos_modelo"]:
        raise IncompatibilidadModeloError(
            "El índice no contiene metadatos de versión de modelo (`modelo_version`). "
            "Regenérelo o inyecte los metadatos verificados."
        )

    return meta


def validar_artefactos_web(
    carpeta_datos: str | Path,
    ruta_onnx: str | Path | None = None,
    version_esperada: str | None = None,
) -> dict[str, Any]:
    """Verifica la coherencia mutua de todos los archivos exportados para el navegador (`frontend/datos/`).

    Comprueba:
      - `actores.json`: existencia de `dim`, `actores` y metadatos opcionales (`version_modelo`, `arquitectura`).
      - `lda.bin`: tamaño exacto `(dim + dim * dim) * 4` bytes y ausencia de NaN/Inf.
      - `centroides.bin`: tamaño exacto `num_actores * dim * 4` bytes y vectores unitarios.
      - `trozos.bin`: cabecera `uint32` de longitud `num_actores + 1` monótona creciente, escalas `float32`
        y matriz cuantizada `int8` de tamaño `num_trozos * dim`.
      - Si se proporciona `ruta_onnx`, verifica que la arquitectura y dimensión del ONNX coincidan con los datos.
    """
    datos = Path(carpeta_datos)
    requeridos = ("actores.json", "lda.bin", "centroides.bin", "trozos.bin")
    for nombre in requeridos:
        if not (datos / nombre).exists():
            raise FileNotFoundError(f"Falta el artefacto web requerido: {datos / nombre}")

    meta = json.loads((datos / "actores.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    actores = meta["actores"]
    n_actores = len(actores)

    # 1. Validar lda.bin
    lda_floats = np.fromfile(datos / "lda.bin", dtype="<f4")
    esperado_lda = dim + dim * dim
    if len(lda_floats) != esperado_lda:
        raise IncompatibilidadModeloError(
            f"`lda.bin` tiene {len(lda_floats)} floats, pero para dim={dim} se esperaban {esperado_lda} "
            f"({dim} de media + {dim}x{dim} de matriz W)."
        )
    if not np.all(np.isfinite(lda_floats)):
        raise IncompatibilidadModeloError("`lda.bin` contiene valores NaN o Inf.")

    # 2. Validar centroides.bin
    cent_floats = np.fromfile(datos / "centroides.bin", dtype="<f4")
    esperado_cent = n_actores * dim
    if len(cent_floats) != esperado_cent:
        raise IncompatibilidadModeloError(
            f"`centroides.bin` tiene {len(cent_floats)} floats, pero para {n_actores} actores y dim={dim} "
            f"se esperaban {esperado_cent}."
        )
    C = cent_floats.reshape(n_actores, dim)
    normas_c = np.linalg.norm(C, axis=1)
    if not np.allclose(normas_c, 1.0, atol=1e-2):
        raise IncompatibilidadModeloError(
            f"`centroides.bin` contiene vectores no normalizados (norma mín={normas_c.min():.4f}, máx={normas_c.max():.4f})."
        )

    # 3. Validar trozos.bin
    raw_trozos = (datos / "trozos.bin").read_bytes()
    n_cabecera = n_actores + 1
    if len(raw_trozos) < n_cabecera * 4:
        raise IncompatibilidadModeloError("`trozos.bin` está truncado en la cabecera de índices.")
    inicio = np.frombuffer(raw_trozos, dtype="<u4", count=n_cabecera)
    if int(inicio[0]) != 0 or np.any(np.diff(inicio.astype(np.int64)) < 0):
        raise IncompatibilidadModeloError("`trozos.bin` tiene una tabla `inicio` no monótona o que no empieza en 0.")
    n_trozos = int(inicio[-1])
    bytes_esperados = n_cabecera * 4 + n_trozos * 4 + n_trozos * dim
    if len(raw_trozos) != bytes_esperados:
        raise IncompatibilidadModeloError(
            f"`trozos.bin` mide {len(raw_trozos)} bytes, pero para {n_actores} actores, {n_trozos} trozos "
            f"y dim={dim} debe medir exactamente {bytes_esperados} bytes."
        )

    # 4. Validar coherencia con versión esperada y modelo ONNX
    ver_web = str(meta.get("version_modelo")) if meta.get("version_modelo") is not None else None
    if version_esperada is not None:
        info_esp = info_por_version(version_esperada)
        if dim != info_esp.dim_embedding:
            raise IncompatibilidadModeloError(
                f"Los datos web tienen dim={dim}, pero la versión esperada {version_esperada} "
                f"({info_esp.arquitectura}) requiere dim={info_esp.dim_embedding}."
            )
        if ver_web is not None and ver_web != str(version_esperada):
            raise IncompatibilidadModeloError(
                f"`actores.json` declara version_modelo={ver_web!r}, pero se esperaba {version_esperada!r}."
            )

    info_onnx = None
    if ruta_onnx is not None and Path(ruta_onnx).exists():
        info_onnx = inspeccionar_onnx(ruta_onnx)
        if info_onnx.dim_embedding != dim:
            raise IncompatibilidadModeloError(
                f"El modelo ONNX ({info_onnx.arquitectura}, dim={info_onnx.dim_embedding}) es incompatible "
                f"con los datos web (dim={dim})."
            )
        if ver_web is not None and info_onnx.version != ver_web:
            raise IncompatibilidadModeloError(
                f"El modelo ONNX es versión {info_onnx.version} ({info_onnx.arquitectura}), pero "
                f"`actores.json` fue exportado para la versión {ver_web} ({meta.get('arquitectura_modelo')})."
            )

    return {
        "dim": dim,
        "num_actores": n_actores,
        "num_trozos": n_trozos,
        "version_modelo": ver_web,
        "arquitectura_modelo": meta.get("arquitectura_modelo"),
        "variante_modelo": meta.get("variante_modelo"),
        "version_lda": meta.get("version_lda"),
        "sha256_lda": sha256_array(lda_floats),
        "onnx": info_onnx.a_dict() if info_onnx else None,
    }
