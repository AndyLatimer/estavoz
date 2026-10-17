# Estavoz

Escucha unos segundos de una película o serie y dice qué actor de doblaje habla.

## Cómo funciona
1. `data/actores/<slug>/` tiene `ficha.json` y una carpeta `clips/` con audios de ese actor.
2. `backend/indexar.py` calcula la "huella de voz" de cada clip (modelo ECAPA de SpeechBrain).
3. `backend/app.py` recibe la grabación del móvil, calcula su huella y devuelve los 3 actores más parecidos.
4. `frontend/index.html` es la web con el botón "Escuchar".

## Datos de eldoblaje.com
```bash
.venv/bin/python scraper/eldoblaje.py listar      # lista de actores de doblaje
.venv/bin/python scraper/eldoblaje.py fichas      # fichas y clips (se puede cortar y reanudar)
```

## Uso
```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend/indexar.py          # tras añadir o cambiar actores
.venv/bin/uvicorn backend.app:app --port 8000
```
Abre http://localhost:8000 (el micrófono solo funciona en localhost o con https).

## Pendiente
- Ampliar el índice a todos los actores de eldoblaje.com con muestras de voz.
- Separar la voz de la música/efectos (Demucs) para audio de películas.
- Subirlo a un servidor con https.
