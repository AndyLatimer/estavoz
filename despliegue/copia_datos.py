"""Copia de seguridad privada de data/ en un dataset de Hugging Face (solo lo ve su dueño).

Uso: HF_TOKEN=hf_... .venv/bin/python despliegue/copia_datos.py <usuario>/estavoz-datos [subir|bajar]

Guarda clips, fotos, fichas, índice y huellas calculadas. No sube los episodios completos de
podcast (data/podcasts/audio): se pueden volver a descargar y lo útil ya está en los clips.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RAIZ = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2 or "HF_TOKEN" not in os.environ:
        sys.exit(__doc__)
    repo, accion = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "subir")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if accion == "bajar":
        snapshot_download(repo, repo_type="dataset", local_dir=RAIZ / "data", token=os.environ["HF_TOKEN"])
        print("Datos restaurados en data/")
        return
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=RAIZ / "data", commit_message="Copia de datos",
                            ignore_patterns=["podcasts/audio/**", "entrevistas/audio/**", "ruidos/**", "cache_huellas/**",
                                             "cache_indice_vb/**", "cache_indice_r100/**", "indice_r34.npz", "indice_antes_voces.npz", "voces/tmp/*", "fichas_r34.json", "**/*.part", "respuestas.json"])
    print(f"Copia subida a https://huggingface.co/datasets/{repo} (privado)")


if __name__ == "__main__":
    main()
