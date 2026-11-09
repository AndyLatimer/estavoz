"""Pruebas automatizadas de coherencia de modelos y validación de artefactos (Fase 4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from metadatos_modelo import (  # noqa: E402
    IncompatibilidadModeloError,
    _inspeccionar_onnx_sin_libreria,
    extraer_info_indice,
    info_por_variante,
    info_por_version,
    inscribir_metadatos_onnx,
    inspeccionar_onnx,
    validar_artefactos_web,
    validar_compatibilidad_indice,
)


def _crear_onnx_sintetico(ruta: Path, dim_salida: int, num_nodos_conv: int) -> Path:
    """Construye un modelo ONNX válido con el número indicado de nodos Conv y dimensión de salida."""
    inp = helper.make_tensor_value_info("wav", TensorProto.FLOAT, [1, None])
    out = helper.make_tensor_value_info("huella", TensorProto.FLOAT, [1, dim_salida])

    nodos = []
    prev = "wav"
    for i in range(num_nodos_conv):
        salida_i = f"conv_out_{i}"
        nodos.append(helper.make_node("Conv", inputs=[prev, f"w_{i}"], outputs=[salida_i], name=f"conv_{i}"))
        prev = salida_i

    peso_final = helper.make_tensor("w_gemm", TensorProto.FLOAT, [dim_salida, 64], np.zeros(dim_salida * 64, dtype=np.float32))
    nodos.append(helper.make_node("Gemm", inputs=[prev, "w_gemm"], outputs=["huella"], name="bottleneck"))

    grafo = helper.make_graph(nodos, "grafo_prueba", [inp], [out], initializer=[peso_final])
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(modelo, str(ruta))
    return ruta


def test_inspeccionar_onnx_distingue_r34_r100_y_ecapa_por_estructura(tmp_path: Path):
    """Aunque r34 y r100 tengan dim=256 y se llamen igual, `inspeccionar_onnx` los distingue."""
    onnx_ecapa = _crear_onnx_sintetico(tmp_path / "ecapa-int8.onnx", dim_salida=192, num_nodos_conv=20)
    onnx_r34 = _crear_onnx_sintetico(tmp_path / "voxblink-int8.onnx", dim_salida=256, num_nodos_conv=39)
    onnx_r100 = _crear_onnx_sintetico(tmp_path / "voxblink-r100-int8.onnx", dim_salida=256, num_nodos_conv=105)

    info_e = inspeccionar_onnx(onnx_ecapa)
    assert info_e.version == "1"
    assert info_e.variante == "ecapa"
    assert info_e.dim_embedding == 192
    assert info_e.origen_deteccion == "estructura_grafo"

    info_34 = inspeccionar_onnx(onnx_r34)
    assert info_34.version == "2"
    assert info_34.variante == "r34"
    assert info_34.arquitectura == "voxblink2-simam-resnet34"
    assert info_34.dim_embedding == 256
    assert info_34.num_nodos_conv == 39

    info_100 = inspeccionar_onnx(onnx_r100)
    assert info_100.version == "3"
    assert info_100.variante == "r100"
    assert info_100.arquitectura == "voxblink2-simam-resnet100"
    assert info_100.dim_embedding == 256
    assert info_100.num_nodos_conv == 105


def test_inscribir_y_verificar_metadatos_onnx(tmp_path: Path):
    """Verifica la escritura de `metadata_props` y la detección de falsificaciones/incoherencias."""
    ruta = _crear_onnx_sintetico(tmp_path / "voxblink-int8.onnx", dim_salida=256, num_nodos_conv=105)
    inscribir_metadatos_onnx(ruta, info_por_variante("r100"))

    info = inspeccionar_onnx(ruta)
    assert info.version == "3"
    assert info.variante == "r100"
    assert info.origen_deteccion == "metadata_props"
    assert info.bloques == (6, 16, 24, 3)

    # También funciona con el lector binario sin librería `onnx`
    info_bin = _inspeccionar_onnx_sin_libreria(ruta, info.sha256_onnx, ruta.stat().st_size)
    assert info_bin.version == "3"
    assert info_bin.variante == "r100"
    assert info_bin.origen_deteccion == "protobuf_bytes"

    # Si alguien inscribe metadatos de r100 en un grafo que en realidad es r34 (39 Conv), debe fallar
    ruta_falsa = _crear_onnx_sintetico(tmp_path / "falso.onnx", dim_salida=256, num_nodos_conv=39)
    inscribir_metadatos_onnx(ruta_falsa, info_por_variante("r100"))
    with pytest.raises(IncompatibilidadModeloError, match="nodos Conv"):
        inspeccionar_onnx(ruta_falsa)


def test_validar_compatibilidad_indice_detecta_choque_r34_vs_r100(tmp_path: Path):
    """Detecta índices de igual dimensión (256) pero distinta versión de modelo (r34 vs r100)."""
    ruta_idx = tmp_path / "indice.npz"
    X = np.ones((10, 256), dtype=np.float32)
    y = np.array(["actor-1"] * 5 + ["actor-2"] * 5)
    clips = np.array([f"actor-1/{i}.mp3" for i in range(5)] + [f"actor-2/{i}.mp3" for i in range(5)])

    info_r34 = info_por_version("2")
    info_r100 = info_por_version("3")
    info_ecapa = info_por_version("1")

    np.savez(
        ruta_idx,
        vectores=X,
        etiquetas=y,
        clips=clips,
        separado=False,
        modelo_version=info_r34.version,
        modelo_arquitectura=info_r34.arquitectura,
        modelo_variante=info_r34.variante,
        modelo_sha256="abc123456789",
    )
    datos = np.load(ruta_idx)

    # Compatible con r34
    meta = validar_compatibilidad_indice(datos, info_modelo=info_r34, version_esperada="2")
    assert meta["modelo_version"] == "2"

    # Incompatible con r100 aunque ambos tengan dim=256
    with pytest.raises(IncompatibilidadModeloError, match="Incompatibilidad crítica"):
        validar_compatibilidad_indice(datos, info_modelo=info_r100)

    # Incompatible con versión esperada "3"
    with pytest.raises(IncompatibilidadModeloError, match="Incompatibilidad de versión"):
        validar_compatibilidad_indice(datos, version_esperada="3")

    # Incompatible con ECAPA (dim 192 vs 256)
    with pytest.raises(IncompatibilidadModeloError, match="dim"):
        validar_compatibilidad_indice(datos, info_modelo=info_ecapa)


def test_validar_artefactos_web_integros_y_corruptos(tmp_path: Path):
    """Comprueba la validación cruzada de `lda.bin`, `centroides.bin`, `trozos.bin` y `actores.json`."""
    datos_dir = tmp_path / "datos"
    datos_dir.mkdir()
    dim = 256
    n_actores = 3
    n_trozos = 7

    mu = np.zeros(dim, dtype="<f4")
    W = np.eye(dim, dtype="<f4")
    np.concatenate([mu, W.ravel()]).tofile(datos_dir / "lda.bin")

    C = np.zeros((n_actores, dim), dtype="<f4")
    C[:, 0] = 1.0  # vectores unitarios
    C.tofile(datos_dir / "centroides.bin")

    inicio = np.array([0, 2, 5, n_trozos], dtype="<u4")
    escala = np.full(n_trozos, 1.0 / 127.0, dtype="<f4")
    q8 = np.ones((n_trozos, dim), dtype=np.int8)
    with open(datos_dir / "trozos.bin", "wb") as f:
        f.write(inicio.tobytes())
        f.write(escala.tobytes())
        f.write(q8.tobytes())

    web_meta = {
        "dim": dim,
        "version_modelo": "3",
        "arquitectura_modelo": "voxblink2-simam-resnet100",
        "variante_modelo": "r100",
        "version_lda": "1.0",
        "actores": [{"id": f"a{i}", "nombre": f"Actor {i}"} for i in range(n_actores)],
    }
    (datos_dir / "actores.json").write_text(json.dumps(web_meta), encoding="utf-8")

    onnx_r100 = _crear_onnx_sintetico(tmp_path / "voxblink-int8.onnx", dim_salida=256, num_nodos_conv=105)
    res = validar_artefactos_web(datos_dir, ruta_onnx=onnx_r100, version_esperada="3")
    assert res["dim"] == 256
    assert res["num_actores"] == 3
    assert res["num_trozos"] == 7

    # Si se intenta validar contra un ONNX de ResNet34 (versión 2), debe rechazarlo
    onnx_r34 = _crear_onnx_sintetico(tmp_path / "voxblink-r34.onnx", dim_salida=256, num_nodos_conv=39)
    with pytest.raises(IncompatibilidadModeloError, match="versión"):
        validar_artefactos_web(datos_dir, ruta_onnx=onnx_r34, version_esperada="3")

    # Si se corrompe el tamaño de centroides.bin, debe detectarlo
    np.zeros(10, dtype="<f4").tofile(datos_dir / "centroides.bin")
    with pytest.raises(IncompatibilidadModeloError, match="centroides.bin"):
        validar_artefactos_web(datos_dir)
