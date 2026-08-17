"""
Script 04: descarga los datasets de Roboflow en formato YOLO.

Requiere una API key de Roboflow guardada en un archivo .env en la raiz del
proyecto:

    ROBOFLOW_API_KEY=tu_api_key_aqui

Uso:
    python scripts/04_load_dataset.py                 # descarga todos los datasets
    python scripts/04_load_dataset.py plastic-paper-metal   # solo uno

Para agregar un dataset nuevo basta con anadir una entrada en DATASETS: los
valores de workspace, project y version salen del snippet "show download code"
de la pagina del dataset en Roboflow.
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from roboflow import Roboflow

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"

# Registro de datasets. La clave es el nombre de la carpeta destino dentro de datasets/.
DATASETS = {
    "plastic-paper-metal": {
        "workspace": "danielfdd2004-unitec-edu",
        "project": "plastic-paper-metal",
        # Version 2 = 2234 imagenes (1956 train / 185 valid / 93 test) con augmentations.
        # La version 1 es un export "Roboflow Instant [Eval]" de solo 118 imagenes.
        "version": 2,
    },
}

# Roboflow genera un zip distinto por formato, pero yolov8/yolov9/yolov11 producen
# exactamente la misma estructura (labels .txt + data.yaml). Si un formato tiene el
# export roto en el storage de Roboflow, probamos con el siguiente.
FORMATS = ["yolov8", "yolov9", "yolov11"]

MAX_ATTEMPTS = 6
RETRY_DELAY_SECONDS = 15


def export_link_is_valid(api_key: str, workspace: str, project: str, version: int, fmt: str) -> bool:
    """Comprueba que el zip del export exista de verdad en el storage de Roboflow.

    La API puede devolver un link "listo" apuntando a un objeto que nunca se
    escribio; en ese caso el download falla con BadZipFile tras varios minutos.
    """
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={api_key}"
    data = requests.get(url, timeout=30).json()
    link = data.get("export", {}).get("link")
    if not link:
        return False
    head = requests.head(link, allow_redirects=True, timeout=30)
    return head.status_code == 200


def download_dataset(api_key: str, name: str, config: dict) -> Path:
    workspace = config["workspace"]
    project_slug = config["project"]
    version_number = config["version"]
    location = DATASETS_DIR / name

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(workspace).project(project_slug).version(version_number)

    for fmt in FORMATS:
        print(f"[..] {name}: pidiendo export en formato {fmt}")
        version.export(fmt)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if export_link_is_valid(api_key, workspace, project_slug, version_number, fmt):
                break
            if attempt == MAX_ATTEMPTS:
                print(f"[!!] {name}: el export {fmt} sigue sin estar en el storage de Roboflow")
                break
            print(f"[..] {name}: export {fmt} aun no disponible, reintentando ({attempt}/{MAX_ATTEMPTS})")
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            continue

        try:
            version.download(fmt, location=str(location), overwrite=True)
        except zipfile.BadZipFile:
            zip_path = location / "roboflow.zip"
            if zip_path.exists():
                zip_path.unlink()
            print(f"[!!] {name}: el zip de {fmt} llego corrupto, probando otro formato")
            continue

        print(f"[OK] {name}: descargado en {location}")
        return location

    sys.exit(
        f"\n[!!] No se pudo descargar '{name}': ningun formato ({', '.join(FORMATS)}) tiene un "
        f"export valido en Roboflow.\n"
        f"     Regenera la version {version_number} desde la web de Roboflow "
        f"(Versions > Generate New Version) y vuelve a ejecutar este script."
    )


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Falta ROBOFLOW_API_KEY en el archivo .env de la raiz del proyecto")

    parser = argparse.ArgumentParser(description="Descarga datasets de Roboflow")
    parser.add_argument(
        "datasets",
        nargs="*",
        help=f"nombres a descargar: {', '.join(DATASETS)} (por defecto: todos)",
    )
    args = parser.parse_args()

    names = args.datasets or list(DATASETS)
    desconocidos = [n for n in names if n not in DATASETS]
    if desconocidos:
        sys.exit(f"Dataset(s) desconocido(s): {', '.join(desconocidos)}. Opciones: {', '.join(DATASETS)}")

    for name in names:
        download_dataset(api_key, name, DATASETS[name])


if __name__ == "__main__":
    main()
