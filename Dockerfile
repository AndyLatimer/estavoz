# Imagen para Hugging Face Spaces (Docker). También sirve en cualquier servidor.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces ejecuta el contenedor con el usuario 1000
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH TORCH_HOME=/home/app/.cache/torch
WORKDIR /home/app/estavoz

RUN pip install --no-cache-dir --user torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app backend backend
COPY --chown=app frontend frontend
COPY --chown=app data/indice.npz data/fichas.json data/

# Descarga los modelos al construir la imagen, no en la primera visita
RUN python -c "import sys; sys.path.insert(0, 'backend'); import voz; voz.modelo(); voz.separador()"

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
