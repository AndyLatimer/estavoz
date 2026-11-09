# Estavoz

Escucha unos segundos de una película, serie o videojuego y dice qué actor de doblaje está hablando.
Web: https://srharrison-estavoz.static.hf.space

## Cómo funciona

1. **En el navegador** (`frontend/`): el móvil graba 10 s y calcula la *huella de voz* de cada trozo de 6 s
   con un modelo ONNX que corre en el propio dispositivo (el audio no sale de él). La huella pasa por una
   proyección LDA y se compara con la huella media de cada doblador.
2. **"¿Qué estás viendo?"** (opcional): limita la comparación a los dobladores que tienen esa obra,
   personaje o actor original en sus trabajos (`datos/busqueda.json`).
3. **"¿Acertamos?"**: si el usuario responde, se guarda en Supabase solo la huella (nunca el audio);
   cada noche esas respuestas se incorporan a las huellas de los dobladores.

### Modelo de voz
SimAM-ResNet100 entrenado con VoxBlink2 (100.000+ voces de YouTube en muchos idiomas) y afinado con VoxCeleb2
(`backend/voxblink.py`, variante `r100`, pesos en Hugging Face `bzikst/voxblink2_samresnet100_ft`).
Se exporta a ONNX con el fbank de Kaldi dentro y comprimido a 8 bits (`despliegue/exportar_voxblink.py`, ~52 MB).
En la evaluación reproducible sobre los 836 dobladores (`backend/sistema_evaluacion.py`), con muestras móviles simuladas acierta un 81,8 % Top-1 y 89,5 % Top-3 en búsqueda abierta (76,3 % Top-1 en 6 s; 83,5 % en 12 s; 86,7 % en 18 s), y un 94,2 % Top-1 (97,9 % Top-3) al usar "¿Qué estás viendo?", rechazando además el 95,0 % de las voces desconocidas (`backend/rechazo.py`).

### Por qué acierta con el móvil
- La LDA se entrena con cada clip en versión limpia y en versión "grabada con el móvil delante de la tele"
  (`backend/simular_movil.py`: altavoz pequeño, ecos de habitaciones reales y ruidos reales de OpenSLR 28).
- Además de los clips de eldoblaje.com, hay voz extraída de entrevistas en podcasts (`scraper/podcasts.py`),
  quedándose solo con los trozos donde la voz coincide con la del doblador.

## Datos (no están en git)

`data/` contiene clips, fotos, fichas, trabajos e índice. Hay una copia privada en Hugging Face:
```bash
HF_TOKEN=... .venv/bin/python despliegue/copia_datos.py SrHarrison/estavoz-datos bajar   # recuperar
HF_TOKEN=... .venv/bin/python despliegue/copia_datos.py SrHarrison/estavoz-datos subir   # guardar
```

Para regenerarlos desde cero:
```bash
.venv/bin/python scraper/eldoblaje.py listar       # lista de actores de doblaje
.venv/bin/python scraper/eldoblaje.py fichas       # fichas y clips de voz
.venv/bin/python scraper/eldoblaje.py fotos
.venv/bin/python scraper/eldoblaje.py trabajos     # lista completa de trabajos
.venv/bin/python scraper/podcasts.py buscar && .venv/bin/python scraper/podcasts.py descargar
ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx .venv/bin/python scraper/podcasts.py extraer
```

## Construir y publicar la web

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

.venv/bin/python despliegue/exportar_voxblink.py                                   # modelo para el navegador
ESTAVOZ_ONNX=frontend/modelo/voxblink-int8.onnx .venv/bin/python backend/indexar.py --aumentar   # huellas
.venv/bin/python despliegue/exportar_web.py                                         # datos de la web
HF_TOKEN=... .venv/bin/python despliegue/subir_a_huggingface.py SrHarrison/estavoz  # publicar
```

## Medir el acierto

- `backend/sistema_evaluacion.py` ejecuta la evaluación reproducible completa (global, por actor, grabación, fuente y duración; con/sin búsqueda contextual; simulación de voces desconocidas en 3 estados; informes JSON/CSV y detección de regresiones):
  ```bash
  .venv/bin/python backend/sistema_evaluacion.py --salida data/evaluaciones/estandar
  .venv/bin/python backend/sistema_evaluacion.py --solo-galeria-independiente --salida data/evaluaciones/independiente
  ```
- `backend/experimentar.py` compara variantes con fragmentos de clips de eldoblaje: `lda`, `movil`, `aumentado`, `duracion`, `modelos`.

## Base de datos de respuestas

`despliegue/supabase.sql` crea la tabla. La web solo puede añadir filas (clave pública en `frontend/config.js`).
`.github/workflows/aprender.yml` las recoge cada noche (necesita los secretos `HF_TOKEN`, `SUPABASE_URL`,
`SUPABASE_SERVICE_KEY` y las variables `HF_SPACE`, `HF_DATOS`).

## Licencias de terceros

- onnxruntime-web (MIT), coi-serviceworker (MIT): `frontend/lib/`.
- Fuentes Fraunces y JetBrains Mono (SIL Open Font License): `frontend/fuentes/`.
- Ecos y ruidos: OpenSLR 28 (Apache 2.0).
