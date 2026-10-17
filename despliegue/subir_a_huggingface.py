"""Sube la web de Estavoz a un Space estático de Hugging Face (https y alojamiento gratis).

Necesita un token de escritura de https://huggingface.co/settings/tokens en HF_TOKEN.
Uso: HF_TOKEN=hf_... .venv/bin/python despliegue/subir_a_huggingface.py <usuario>/estavoz

Sube solo frontend/ (web, modelo y huellas). Los clips de audio NO se suben.
Antes: indexar con ESTAVOZ_ONNX=frontend/modelo/ecapa-int8.onnx y despliegue/exportar_web.py.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parent.parent
WEB = RAIZ / "frontend"
CABECERA = """---
title: Estavoz
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: static
pinned: false
---

Estavoz: escucha unos segundos de una película o serie y te dice qué actor de doblaje habla.
"""


def main():
    if len(sys.argv) != 2 or "HF_TOKEN" not in os.environ:
        sys.exit(__doc__)
    repo = sys.argv[1]
    for f in ("modelo/ecapa-int8.onnx", "datos/huellas.bin", "datos/actores.json"):
        if not (WEB / f).exists():
            sys.exit(f"Falta frontend/{f}")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp) / "web"
        shutil.copytree(WEB, tmp)
        (tmp / "README.md").write_text(CABECERA, encoding="utf-8")
        # delete_patterns: quita fotos y datos viejos que ya no estén en esta versión
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp, delete_patterns=["fotos/*", "datos/*"],
                          commit_message="Actualizar Estavoz")
    usuario, nombre = repo.split("/")
    print(f"Subido: https://huggingface.co/spaces/{repo}")
    print(f"Directo: https://{usuario.lower()}-{nombre.lower()}.static.hf.space")


if __name__ == "__main__":
    main()
