"""Pruebas unitarias de las mejoras de filmografía, parsing sin BS4, remuestreo y AS-Norm."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RAIZ = Path(__file__).resolve().parent.parent
for carpeta in ("backend", "scraper"):
    ruta = str(RAIZ / carpeta)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

import eldoblaje  # noqa: E402
from filmografia import (  # noqa: E402
    clasificar_tipo_obra,
    elegir_trabajos_destacados,
    filtrar_papeles_actor,
    nombre_original,
    resumen_actor_desde_papeles,
)
from rechazo import aplicar_as_norm, calcular_estadisticas_cohorte  # noqa: E402
from voz import SAMPLE_RATE, _a_mono_16k, cargar_audio_bytes  # noqa: E402


def test_clasificar_tipo_obra_y_nombre_original():
    assert clasificar_tipo_obra("THE BATMAN") == "pelicula"
    assert clasificar_tipo_obra("BREAKING BAD [Serie TV]") == "serie"
    assert clasificar_tipo_obra("LA JAULA (1x05)", "Sofiane (1x05)") == "serie"
    assert clasificar_tipo_obra("TOY STORY [Animación]") == "animacion"
    assert clasificar_tipo_obra("DRAGON BALL Z [Anime]") == "animacion"
    assert clasificar_tipo_obra("GOD OF WAR [Videojuego]") == "videojuego"
    assert clasificar_tipo_obra("PLANETA TIERRA [Documental]") == "documental"

    assert nombre_original("Bale, Christian") == "Christian Bale"
    assert nombre_original("(VOZ)") == ""
    assert nombre_original("DESCONOCIDO") == ""


def test_elegir_trabajos_destacados_diversifica_formatos():
    filas = [
        ["LA JAULA [1ª temporada]", "Voces adicionales", 2024, "(DESCONOCIDO)"],
        ["EL CABALLERO OSCURO", "Bruce Wayne / Batman", 2008, "Bale, Christian"],
        ["BREAKING BAD [Serie TV]", "Walter White", 2010, "Cranston, Bryan"],
        ["TOY STORY [Animación]", "Buzz Lightyear (voz)", 1995, "(VOZ)"],
        ["THE LAST OF US [Videojuego]", "Joel", 2013, "(VOZ)"],
        ["AMERICAN PSYCHO", "Patrick Bateman", 2000, "Bale, Christian"],
    ]
    popularidad = {"el caballero oscuro": 40, "breaking bad": 35, "toy story": 30, "the last of us": 20}
    res = resumen_actor_desde_papeles(filas, popularidad_obras=popularidad, max_trabajos=6)

    destacados = res["trabajos"]
    assert len(destacados) >= 4
    joined = "\n".join(destacados)
    assert "EL CABALLERO OSCURO" in joined
    assert "BREAKING BAD" in joined
    assert "TOY STORY" in joined
    assert "THE LAST OF US" in joined
    assert res["resumen_tipos"]["pelicula"] == 2
    assert res["resumen_tipos"]["videojuego"] == 1
    assert res["vozh"] == [{"nombre": "Christian Bale", "n": 2}]


def test_filtrar_papeles_actor_por_tipo_y_texto():
    filas = [
        ["INCEPTION", "Cobb", 2010, "DiCaprio, Leonardo"],
        ["GOD OF WAR [Videojuego]", "Kratos", 2018, "Judge, Christopher"],
        ["ARCANE [Serie de Animación]", "Jayce", 2021, "(VOZ)"],
    ]
    solo_juegos = filtrar_papeles_actor(filas, tipo="videojuego")
    assert solo_juegos["total"] == 1
    assert solo_juegos["items"][0]["personaje"] == "Kratos"

    busq = filtrar_papeles_actor(filas, q="dicaprio")
    assert busq["total"] == 1
    assert busq["items"][0]["obra"] == "INCEPTION"


def test_parser_eldoblaje_conserva_filas_sin_actor_original():
    html_ejemplo = """
    <html><body><table>
      <tr>
        <td width="161">(VOZ)</td>
        <td width="335"><a href="FichaPelicula.asp?id=999">TOY STORY [Animación]</a></td>
        <td width="166">Buzz Lightyear</td>
        <td width="38">1995</td>
      </tr>
      <tr>
        <td width="161"><a href="FichaActorOriginal.asp?id=12">ALLEN, TIM</a></td>
        <td width="335"><a href="FichaPelicula.asp?id=1000">GALAXY QUEST</a></td>
        <td width="166">Jason Nesmith</td>
        <td width="38">1999</td>
      </tr>
    </table>
    <a href="Saludos/52672.mp3">Saludo oficial</a>
    </body></html>
    """
    trabajos = eldoblaje._parsear_trabajos_html(html_ejemplo, con_id=True)
    assert len(trabajos) == 2
    assert trabajos[0]["obra"] == "TOY STORY [Animación]"
    assert trabajos[0]["actor_original"] == ""
    assert trabajos[0]["personaje"] == "Buzz Lightyear"
    assert trabajos[1]["actor_original"] == "Allen, Tim"

    muestras = eldoblaje._muestras_html(html_ejemplo)
    assert muestras == [{"ruta": "Saludos/52672.mp3", "titulo": "Saludo oficial"}]


def test_remuestreo_y_cargar_audio_bytes_preserva_frecuencia():
    sr_origen = 48000
    t = np.arange(sr_origen, dtype=np.float32) / sr_origen
    tono_est = np.stack([0.5 * np.sin(2 * np.pi * 1000.0 * t), 0.5 * np.sin(2 * np.pi * 1000.0 * t)], axis=1)

    buf = io.BytesIO()
    sf.write(buf, tono_est, sr_origen, format="WAV")
    out = cargar_audio_bytes(buf.getvalue())

    assert out.dtype == np.float32
    assert len(out) == SAMPLE_RATE
    espectro = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1.0 / SAMPLE_RATE)
    freq_pico = freqs[int(np.argmax(espectro))]
    assert abs(freq_pico - 1000.0) <= 5.0
    assert len(_a_mono_16k(tono_est, sr_origen, max_seg=1)) == SAMPLE_RATE


def test_as_norm_mejora_discriminacion_contra_imanes():
    rng = np.random.default_rng(7)
    C = rng.normal(size=(30, 32)).astype(np.float32)
    # Actor 0 es un "imán" con vecinos altos en promedio, Actor 1 es más específico
    C[0] = C[2:12].mean(axis=0)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    mu_a, sd_a = calcular_estadisticas_cohorte(C, k=15)
    assert mu_a.shape == (30,)
    assert np.all(sd_a >= 1e-4)

    brutas = np.full(30, 0.10, dtype=np.float32)
    brutas[0] = 0.34
    brutas[1] = 0.335
    norm = aplicar_as_norm(brutas, mu_a=mu_a, sd_a=sd_a, k=15, peso=0.04)
    # El imán (con mu_a alta) debe penalizarse respecto al actor específico con puntuación bruta casi igual
    assert (norm[1] - norm[0]) > (brutas[1] - brutas[0])


def test_incorporar_voces_poda_intrusos_y_alias(monkeypatch):
    import voz
    from incorporar_voces import _claves_nombre_actor, _es_entrevista_solitaria, podar_intrusos_estado

    assert "lluis posada" in _claves_nombre_actor("Luis Posada")
    assert "ricard solans" in _claves_nombre_actor("Ricardo Solans")
    assert "pere molina" in _claves_nombre_actor("Pedro Molina")
    assert "sergi zamora" in _claves_nombre_actor("Sergio Zamora")
    assert "joan carles gustems" in _claves_nombre_actor("Juan Carlos Gustems")
    assert "albert trifol" in _claves_nombre_actor("Alberto Trifol")
    assert "m jesus lleonart" in _claves_nombre_actor("María Jesús Lleonart")
    assert "francisco javier garcia saenz" in _claves_nombre_actor("Javier García-Sáenz")
    assert "fer delgado" in _claves_nombre_actor("Fernando (fer) Delgado")
    assert "fernando delgado" in _claves_nombre_actor("Fernando (fer) Delgado")
    assert "rafa calvo" in _claves_nombre_actor("Rafael Calvo")
    assert "gaby jimenez" in _claves_nombre_actor("Gabriel Jiménez")
    assert "mercedes montala" in _claves_nombre_actor("Mercè Montalà")

    assert _es_entrevista_solitaria("109. Especial Wicked, con Sara Heras", "sara heras") is True
    assert _es_entrevista_solitaria("Oliver y Benji. Entrevista a Jorge Saudinos", "jorge saudinos") is True
    assert _es_entrevista_solitaria("Charleta con PABLO DOMÍNGUEZ LAGARES", "pablo dominguez lagares") is True
    assert _es_entrevista_solitaria("T3xC19, amb Carles Lladó i Josep Maria Mas", "carles llado") is False
    assert _es_entrevista_solitaria("108. Especial Bluey, con Lucía Pérez y Daniela Portugués", "lucia perez") is False

    # Fallback a ffmpeg cuando soundfile.read no puede decodificar un contenedor alternativo (p. ej. m4a)
    llamadas_ff = []
    esperado = np.ones(160, dtype=np.float32) * 0.25
    monkeypatch.setattr(voz, "_decodificar_con_ffmpeg", lambda ruta, separar=False, max_seg=None: (llamadas_ff.append(ruta), esperado)[1])
    out_fb = cargar_audio_bytes(b"NO_ES_WAV_NI_MP3_M4A_FAKE")
    assert len(llamadas_ff) == 1
    assert np.allclose(out_fb, esperado)

    dim = 16
    c0 = np.eye(1, dim, 0, dtype=np.float32)[0]
    limpios = np.tile(c0, (5, 1)).astype(np.float32)
    intruso = -c0[None, :].astype(np.float32)
    estado = {
        "actores": [{"id": "1-prueba", "nombre": "Actor Prueba"}],
        "C": c0[None, :].copy(),
        "trozos_por_actor": [np.vstack([limpios, intruso])],
    }
    podados = podar_intrusos_estado(estado, umbral_outlier=0.20, min_trozos=4)
    assert podados == 1
    assert len(estado["trozos_por_actor"][0]) == 5


