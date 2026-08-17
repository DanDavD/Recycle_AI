"""
Script 04: descarga los datasets de Roboflow en formato YOLO.

Requiere una API key de Roboflow en un archivo .env en la raiz del proyecto
(copia la plantilla: copy .env.example .env):

    ROBOFLOW_API_KEY=tu_api_key_aqui

Uso:
    python scripts/04_load_dataset.py                       # descarga todos los del registro
    python scripts/04_load_dataset.py plastic-paper-metal    # solo uno
    python scripts/04_load_dataset.py --snippet snippet.txt  # dataset nuevo desde el snippet
    python scripts/04_load_dataset.py --snippet -            # pega el snippet y Ctrl+Z, Enter

Para agregar un dataset al registro permanente basta con anadir una entrada en
DATASETS: workspace, project y version salen del snippet "Show download code"
de la pagina del dataset en Roboflow (boton Download Dataset > Show download code).
Ese mismo snippet se puede pasar tal cual con --snippet sin tocar el codigo.
"""

import argparse
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import requests
import yaml
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


# --- Lectura del snippet de Roboflow ----------------------------------------


def parse_snippet(text: str) -> dict:
    """Extrae workspace / project / version (y api_key) del snippet de Roboflow.

    Acepta el bloque de Python que da "Show download code", del estilo:

        from roboflow import Roboflow
        rf = Roboflow(api_key="xxxxxxxx")
        project = rf.workspace("mi-workspace").project("mi-proyecto")
        version = project.version(2)
        dataset = version.download("yolov8")
    """
    def buscar(patron: str):
        match = re.search(patron, text)
        return match.group(1) if match else None

    workspace = buscar(r"\.workspace\(\s*[\"']([^\"']+)[\"']\s*\)")
    project = buscar(r"\.project\(\s*[\"']([^\"']+)[\"']\s*\)")
    version = buscar(r"\.version\(\s*(\d+)\s*\)")
    api_key = buscar(r"api_key\s*=\s*[\"']([^\"']+)[\"']")
    fmt = buscar(r"\.download\(\s*[\"']([^\"']+)[\"']")

    faltantes = [
        nombre
        for nombre, valor in (("workspace", workspace), ("project", project), ("version", version))
        if not valor
    ]
    if faltantes:
        sys.exit(
            f"[!!] No pude leer {', '.join(faltantes)} del snippet.\n"
            f"     Pega el bloque completo de 'Show download code' (el que incluye\n"
            f"     rf.workspace(...).project(...) y project.version(N))."
        )

    config = {"workspace": workspace, "project": project, "version": int(version)}
    if api_key:
        config["api_key"] = api_key
    if fmt:
        config["format"] = fmt
    return config


def read_snippet(origen: str) -> str:
    if origen == "-":
        print("Pega el snippet de Roboflow y termina con Ctrl+Z + Enter (Windows) o Ctrl+D:\n")
        return sys.stdin.read()
    ruta = Path(origen)
    if not ruta.exists():
        sys.exit(f"[!!] No encuentro el archivo del snippet: {ruta}")
    return ruta.read_text(encoding="utf-8")


# --- Descarga ----------------------------------------------------------------


def export_link_is_valid(api_key: str, workspace: str, project: str, version: int, fmt: str) -> bool:
    """Comprueba que el zip del export exista de verdad en el storage de Roboflow.

    La API puede devolver un link "listo" apuntando a un objeto que nunca se
    escribio; en ese caso el download falla con BadZipFile tras varios minutos.
    """
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={api_key}"
    try:
        data = requests.get(url, timeout=30).json()
        link = data.get("export", {}).get("link")
        if not link:
            return False
        return requests.head(link, allow_redirects=True, timeout=30).status_code == 200
    except requests.RequestException as exc:
        print(f"[..] error de red consultando el export ({exc.__class__.__name__}), se reintenta")
        return False


def wait_for_export(api_key: str, workspace: str, project: str, version: int, fmt: str) -> bool:
    """Espera a que el zip del export aparezca en el storage. True si aparecio."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if export_link_is_valid(api_key, workspace, project, version, fmt):
            return True
        if attempt == MAX_ATTEMPTS:
            print(f"[!!] el export {fmt} sigue sin estar en el storage de Roboflow")
            return False
        print(f"[..] export {fmt} aun no disponible, reintentando ({attempt}/{MAX_ATTEMPTS})")
        time.sleep(RETRY_DELAY_SECONDS)
    return False


def download_dataset(api_key: str, name: str, config: dict) -> Path:
    workspace = config["workspace"]
    project_slug = config["project"]
    version_number = config["version"]
    location = DATASETS_DIR / name

    # Si el snippet traia un formato concreto lo probamos primero.
    formats = list(FORMATS)
    preferido = config.get("format")
    if preferido and preferido in formats:
        formats.remove(preferido)
        formats.insert(0, preferido)

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(workspace).project(project_slug).version(version_number)

    for fmt in formats:
        print(f"[..] {name}: pidiendo export en formato {fmt}")
        version.export(fmt)

        if not wait_for_export(api_key, workspace, project_slug, version_number, fmt):
            continue

        try:
            version.download(fmt, location=str(location), overwrite=True)
        except (zipfile.BadZipFile, RuntimeError) as exc:
            zip_path = location / "roboflow.zip"
            if zip_path.exists():
                zip_path.unlink()
            print(f"[!!] {name}: fallo el zip de {fmt} ({exc.__class__.__name__}), probando otro formato")
            continue

        if not (location / "data.yaml").exists():
            print(f"[!!] {name}: el export {fmt} se bajo pero no trae data.yaml, probando otro formato")
            continue

        print(f"[OK] {name}: descargado en {location}")
        return location

    sys.exit(
        f"\n[!!] No se pudo descargar '{name}': ningun formato ({', '.join(formats)}) tiene un "
        f"export valido en Roboflow.\n"
        f"     Regenera la version {version_number} desde la web de Roboflow "
        f"(Versions > Generate New Version) y vuelve a ejecutar este script."
    )


def describe_dataset(location: Path) -> None:
    """Resumen de lo descargado: clases e imagenes por split."""
    data_yaml = location / "data.yaml"
    if not data_yaml.exists():
        return

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print(f"     clases ({data.get('nc')}): {data.get('names')}")
    for split in ("train", "valid", "val", "test"):
        images_dir = location / split / "images"
        if not images_dir.is_dir():
            continue
        n_img = sum(1 for _ in images_dir.iterdir())
        labels_dir = location / split / "labels"
        n_lbl = sum(1 for _ in labels_dir.iterdir()) if labels_dir.is_dir() else 0
        print(f"     {split:<6}: {n_img} imagenes / {n_lbl} labels")


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga datasets de Roboflow")
    parser.add_argument(
        "datasets",
        nargs="*",
        help=f"nombres del registro a descargar: {', '.join(DATASETS)} (por defecto: todos)",
    )
    parser.add_argument(
        "--snippet",
        metavar="ARCHIVO",
        help="ruta a un archivo con el snippet 'Show download code' de Roboflow, o '-' para pegarlo por stdin",
    )
    parser.add_argument(
        "--name",
        help="nombre de la carpeta destino en datasets/ (por defecto: el slug del proyecto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if args.snippet:
        if args.datasets:
            sys.exit("[!!] Usa --snippet o nombres del registro, no las dos cosas a la vez.")
        config = parse_snippet(read_snippet(args.snippet))
        # El snippet trae la key incrustada; sirve de respaldo si aun no hay .env.
        api_key = api_key or config.pop("api_key", None)
        config.pop("api_key", None)
        name = args.name or config["project"]
        print(
            f"[OK] Snippet leido: workspace={config['workspace']} "
            f"project={config['project']} version={config['version']}"
        )
        trabajos = [(name, config)]
    else:
        if args.name:
            sys.exit("[!!] --name solo aplica junto con --snippet.")
        names = args.datasets or list(DATASETS)
        desconocidos = [n for n in names if n not in DATASETS]
        if desconocidos:
            sys.exit(
                f"[!!] Dataset(s) desconocido(s): {', '.join(desconocidos)}.\n"
                f"     Opciones del registro: {', '.join(DATASETS)}\n"
                f"     Para uno nuevo:  python scripts/04_load_dataset.py --snippet -"
            )
        trabajos = [(n, DATASETS[n]) for n in names]

    if not api_key:
        sys.exit(
            "[!!] Falta ROBOFLOW_API_KEY.\n"
            "     Crea el .env copiando la plantilla:  copy .env.example .env\n"
            "     y pega ahi tu Private API Key (app.roboflow.com > Settings > API Keys)."
        )

    for name, config in trabajos:
        location = download_dataset(api_key, name, config)
        describe_dataset(location)

    print("\nSiguiente paso:")
    print(f"    python scripts/05_train_model.py --dataset {trabajos[0][0]}")


if __name__ == "__main__":
    main()
