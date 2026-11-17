"""Tests automatizados para el servidor FastAPI de EstaVoz (backend/app.py)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from app import app


@pytest.fixture(scope="module")
def cliente():
    """Cliente de pruebas con el ciclo de vida (lifespan) de FastAPI activado."""
    with TestClient(app) as test_client:
        yield test_client


def _crear_wav_en_memoria(duracion_seg: float = 4.0, sr: int = 16000, f0: float = 180.0) -> bytes:
    """Genera un archivo WAV en memoria con señal pseudo-vocal periódica."""
    t = np.arange(int(sr * duracion_seg), dtype=np.float32) / sr
    audio = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_app_salud(cliente):
    """Verifica el endpoint de salud y la carga íntegra de los actores."""
    res = cliente.get("/api/salud")
    assert res.status_code == 200
    datos = res.json()
    assert datos["ok"] is True
    assert datos["modo_indice"] in ("web", "npz")
    info = datos["info_modelo"]
    assert info["dim"] == 256
    assert info["num_actores"] >= 836
    assert info["version_modelo"] == "3"
    assert info["variante_modelo"] == "r100"


def test_app_sirve_frontend_estatico(cliente):
    """Verifica que el servidor sirve correctamente la aplicación web en la raíz."""
    res = cliente.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Estavoz" in res.text
    assert "voxblink-int8.onnx" in res.text


def test_app_identificar_rechaza_audio_demasiado_corto(cliente):
    """Verifica que audios menores a 2 segundos devuelven error 400 controlado."""
    wav_corto = _crear_wav_en_memoria(duracion_seg=1.0)
    res = cliente.post(
        "/api/identificar",
        files={"audio": ("corto.wav", wav_corto, "audio/wav")},
    )
    assert res.status_code == 400
    assert "corto" in res.json()["detail"].lower()


def test_app_identificar_audio_valido(cliente):
    """Verifica la inferencia extremo a extremo con el modelo ONNX y clasificación en 3 estados."""
    wav_valido = _crear_wav_en_memoria(duracion_seg=6.0, f0=150.0)
    res = cliente.post(
        "/api/identificar",
        files={"audio": ("prueba.wav", wav_valido, "audio/wav")},
    )
    assert res.status_code == 200
    datos = res.json()
    assert datos["estado"] in ("identificado", "dudoso", "desconocido")
    assert isinstance(datos["porcentaje"], int)
    assert 0 <= datos["porcentaje"] <= 100
    assert "resultados" in datos
    assert len(datos["resultados"]) == 3
    primer_candidato = datos["resultados"][0]
    assert "slug" in primer_candidato
    assert "parecido" in primer_candidato
    assert "nombre" in primer_candidato


def test_app_buscar_sugerencias(cliente):
    """Verifica el buscador contextual de obras, personajes y actores ('¿Qué estás viendo?')."""
    res = cliente.get("/api/buscar?q=Toy+Story")
    assert res.status_code == 200
    sugerencias = res.json()
    assert isinstance(sugerencias, list)
    assert len(sugerencias) > 0
    primera = sugerencias[0]
    assert "tipo" in primera
    assert "texto" in primera
    assert "num_voces" in primera
    assert "toy story" in primera["texto"].lower()


def test_app_identificar_con_pista_contextual(cliente):
    """Verifica que la pista contextual restringe el espacio de búsqueda y activa con_contexto."""
    wav_valido = _crear_wav_en_memoria(duracion_seg=6.0, f0=150.0)
    res = cliente.post(
        "/api/identificar?pista=Toy+Story",
        files={"audio": ("prueba_contextual.wav", wav_valido, "audio/wav")},
    )
    assert res.status_code == 200
    datos = res.json()
    assert datos["estado"] in ("identificado", "dudoso", "desconocido")
    assert "contexto" in datos
    ctx = datos["contexto"]
    assert ctx["pista"] == "Toy Story"
    assert "num_candidatos" in ctx
    assert 0 < ctx["num_candidatos"] < 200
    assert len(datos["resultados"]) > 0
    assert "porcentaje" in datos["resultados"][0]


def test_app_detalle_actor_y_filmografia_filtrable(cliente):
    """Verifica `/api/actor/{actor_id}` para explorar en qué más sale un doblador (pelis, series, juegos...)."""
    res_buscar = cliente.get("/api/buscar?q=Toy+Story")
    assert res_buscar.status_code == 200
    ident = cliente.post(
        "/api/identificar",
        files={"audio": ("actor.wav", _crear_wav_en_memoria(duracion_seg=4.0, f0=140.0), "audio/wav")},
    )
    assert ident.status_code == 200
    slug = ident.json()["resultados"][0]["slug"]

    res_actor = cliente.get(f"/api/actor/{slug}?limite=20")
    assert res_actor.status_code == 200
    det = res_actor.json()
    assert det["id"] == slug
    assert "filmografia" in det
    assert "total" in det["filmografia"]
    assert "conteo_tipos" in det["filmografia"]
    assert isinstance(det["filmografia"]["items"], list)

    res_404 = cliente.get("/api/actor/000000-actor-inexistente")
    assert res_404.status_code == 404
