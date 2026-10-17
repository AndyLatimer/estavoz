"""Sube Estavoz a un Space de Hugging Face (https gratis).

Necesita un token de escritura de https://huggingface.co/settings/tokens en HF_TOKEN.
Uso: HF_TOKEN=hf_... .venv/bin/python despliegue/subir_a_huggingface.py <usuario>/estavoz

Sube el código, el índice de huellas y las fichas. Los clips de audio NO se suben.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parent.parent
CABECERA = """---
title: Estavoz
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Estavoz: escucha unos segundos de una película o serie y te dice qué actor de doblaje habla.
"""


def main():
    if len(sys.argv) != 2 or "HF_TOKEN" not in os.environ:
        sys.exit(__doc__)
    repo = sys.argv[1]
    for f in ("data/indice.npz", "data/fichas.json"):
        if not (RAIZ / f).exists():
            sys.exit(f"Falta {f}: ejecuta antes backend/indexar.py")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in ("Dockerfile", "requirements.txt"):
            shutil.copy(RAIZ / f, tmp / f)
        for d in ("backend", "frontend"):
            shutil.copytree(RAIZ / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "evaluar.py"))
        (tmp / "data").mkdir()
        for f in ("indice.npz", "fichas.json"):
            shutil.copy(RAIZ / "data" / f, tmp / "data" / f)
        (tmp / "README.md").write_text(CABECERA, encoding="utf-8")
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp,
                          commit_message="Actualizar Estavoz")
    print(f"Subido. En unos minutos estará en https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
