"""Copia de seguridad privada de data/ en un dataset de Hugging Face (solo lo ve su dueño).

Uso: HF_TOKEN=hf_... .venv/bin/python despliegue/copia_datos.py <usuario>/estavoz-datos [subir|bajar|bajar-web]

Guarda clips, fotos, fichas, índice y huellas calculadas. Con `bajar-web` descarga solo
`indice.npz`, `fichas.json`, `actores/*/foto.jpg` y `actores/*/trabajos.json` (~20 MB en vez
de todos los audios) para regenerar la web en segundos.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RAIZ = Path(__file__).resolve().parent.parent
PATRONES_WEB = ["indice.npz", "fichas.json", "actores/*/foto.jpg", "actores/*/trabajos.json"]


def main():
    if len(sys.argv) < 2 or "HF_TOKEN" not in os.environ:
        sys.exit(__doc__)
    repo, accion = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "subir")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if accion in ("bajar", "bajar-web"):
        patrones = PATRONES_WEB if accion == "bajar-web" else None
        snapshot_download(repo, repo_type="dataset", local_dir=RAIZ / "data", token=os.environ["HF_TOKEN"], allow_patterns=patrones)
        print("Datos restaurados en data/")
        return
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=RAIZ / "data", commit_message="Copia de datos",
                            ignore_patterns=["podcasts/audio/**", "entrevistas/audio/**", "ruidos/**", "cache_huellas/**",
                                             "cache_indice_vb/**", "cache_indice_r100/**", "indice_r34.npz", "indice_antes_voces.npz", "voces/tmp/*", "fichas_r34.json", "**/*.part", "respuestas.json"])
    print(f"Copia subida a https://huggingface.co/datasets/{repo} (privado)")


if __name__ == "__main__":
    main()
