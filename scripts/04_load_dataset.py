"""
Script 04: Descarga los datasets de Roboflow en formato YOLOv8.

Descarga todos los datasets configurados (papel, plástico, metal, botellas, latas)
utilizando la API key de Roboflow guardada en .env.

Uso:
    python scripts/04_load_dataset.py                       # descarga todos los datasets
    python scripts/04_load_dataset.py metal-cans            # solo uno
    python scripts/04_load_dataset.py --snippet snippet.txt  # dataset nuevo desde snippet
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
NEGATIVES_DIR = DATASETS_DIR / "negatives"

# Registro central de datasets para el proyecto Recycle_AI
DATASETS = {
    # 1. Dataset base (2826 imágenes)
    "plastic-glass-metal-paper": {
        "workspace": "example-wg2gf",
        "project": "plastic-glass-metal-paper",
        "version": 2,
    },
    # 2. Dataset diversidad potato (515 imágenes)
    "paper-plastic-metal-potato": {
        "workspace": "potato-jkpgv",
        "project": "paper-plastic-metal",
        "version": 1,
    },
    # 3. Papel y cartón (paper-rjcig)
    "paper-rjcig": {
        "workspace": "material-identification",
        "project": "paper-rjcig",
        "version": 1,
    },
    # 4. Plástico botellas (plastic-bottles)
    "plastic-bottles": {
        "workspace": "waste-rq8p9",
        "project": "plastic-bottles-uu8v9",
        "version": 1,
    },
    # 5. Plástico variado y latas (trash-detection)
    "trash-detection": {
        "workspace": "trash-dataset-for-oriented-bounded-box",
        "project": "trash-detection-1fjjc",
        "version": 1,
    },
    # 6. Metal/latas — refuerzo para balancear la clase metal
    "metal-detect-can": {
        "workspace": "mark-nyjbc",
        "project": "detect-can-ddp9h",
        "version": 1,
    },
    # 7. Metal (METAL, Can, Cardboard)
    "metal-dpyt1": {
        "workspace": "gbitirme",
        "project": "metal-dpyt1",
        "version": 1,
    },
    # 8. Reciclables (plastic, glass, metal) — aporta metal extra
    "metal-recyclable-items": {
        "workspace": "durio",
        "project": "recyclable-items-0vzpm",
        "version": 1,
    },
    # 9. Plástico — botellas dedicadas (refuerzo equilibrado)
    "plastic-bottles-7nk9f": {
        "workspace": "timileyin",
        "project": "plastic-bottles-7nk9f",
        "version": 1,
    },
    # 10. Plástico general (botellas, bolsas)
    "plastic-detection-ctqd5": {
        "workspace": "viraj-gi9zk",
        "project": "plastic-detection-ctqd5",
        "version": 2,
    },
    # 11. Plástico — bolsas/botellas/detergente (waste_classification)
    "plastic-waste-classification": {
        "workspace": "rsbpproject",
        "project": "waste_classification-koank",
        "version": 1,
    },
    # 12. Plástico — botella/bolsa (bluewaste, mezclado con glass que se descarta solo)
    "plastic-bluewaste": {
        "workspace": "omary-mkuu",
        "project": "bluewaste",
        "version": 2,
    },
}

# Registro opcional de fuentes de negativos (fondos, madera, tela, desechos ajenos)
NEGATIVE_DATASETS = {
    # Desechos orgánicos y generales para entrenar rechazo de fondo
    "garbage-negatives": {
        "workspace": "dhafar-sami",
        "project": "all-classes-trash-dataset80-3",
        "version": 3,
    },
}


def descargar_dataset(rf: Roboflow, workspace: str, project: str, folder_name: str, target_version: int = 1) -> bool:
    dest = DATASETS_DIR / folder_name
    if (dest / "data.yaml").exists():
        print(f"✅ Ya descargado previamente: {folder_name}")
        return True

    print(f"\n[..] Conectando a Roboflow: {workspace}/{project}...")
    try:
        proj = rf.workspace(workspace).project(project)

        # 1. Intentar descargar la versión indicada
        try:
            print(f"-> Probando versión {target_version}...")
            proj.version(target_version).download("yolov8", location=str(dest))
            if (dest / "data.yaml").exists():
                print(f"✅ {folder_name} descargado con éxito.")
                return True
        except Exception as e_v1:
            print(f"   (Versión {target_version} no disponible directamente: {e_v1}. Buscando versiones publicadas...)")

        # 2. Si target_version no está lista, buscar la versión más reciente con proj.versions()
        lista_versiones = proj.versions() if callable(getattr(proj, "versions", None)) else proj.versions
        nums = []
        for item in lista_versiones:
            val = getattr(item, "version", None) or getattr(item, "version_number", None)
            if val is not None:
                nums.append(int(val))
            elif isinstance(item, (int, str)):
                try:
                    nums.append(int(item))
                except Exception:
                    pass

        if nums:
            ultima = max(nums)
            print(f"-> Descargando versión más reciente: {ultima}...")
            proj.version(ultima).download("yolov8", location=str(dest))
            if (dest / "data.yaml").exists():
                print(f"✅ {folder_name} descargado con éxito.")
                return True
        else:
            print(f"⚠️ No se encontraron versiones generadas en {project}.")
    except Exception as e:
        print(f"⚠️ Error al procesar {folder_name}: {e}")

    return False


def parse_snippet(text: str) -> dict:
    def buscar(patron: str):
        match = re.search(patron, text)
        return match.group(1) if match else None

    workspace = buscar(r"\.workspace\(\s*[\"']([^\"']+)[\"']\s*\)")
    project = buscar(r"\.project\(\s*[\"']([^\"']+)[\"']\s*\)")
    version = buscar(r"\.version\(\s*(\d+)\s*\)")
    api_key = buscar(r"api_key\s*=\s*[\"']([^\"']+)[\"']")

    faltantes = [
        n for n, v in (("workspace", workspace), ("project", project), ("version", version)) if not v
    ]
    if faltantes:
        sys.exit(f"[!!] No pude leer {', '.join(faltantes)} del snippet.")

    config = {"workspace": workspace, "project": project, "version": int(version)}
    if api_key:
        config["api_key"] = api_key
    return config


def extract_images_to_negatives(source_dir: Path, target_dir: Path = NEGATIVES_DIR, max_images: int = 500) -> int:
    """Copia imágenes de un dataset a la carpeta de negativos ignorando sus etiquetas originales."""
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            if target_dir in path.parents:
                continue
            dst = target_dir / f"{source_dir.name}_{path.stem}{path.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(path, dst)
                count += 1
                if count >= max_images:
                    break
    return count


def main():
    parser = argparse.ArgumentParser(description="Descarga datasets de Roboflow para Recycle_AI")
    parser.add_argument("datasets", nargs="*", help="nombres a descargar (por defecto: todos los registrados)")
    parser.add_argument("--snippet", help="ruta al archivo con el snippet de Roboflow, o '-' para pegarlo")
    parser.add_argument(
        "--download-negatives",
        action="store_true",
        help="descarga datasets registrados en NEGATIVE_DATASETS y extrae sus imágenes a datasets/negatives/",
    )
    parser.add_argument(
        "--extract-negatives",
        metavar="DATASET_O_CARPETA",
        help="extrae sólo imágenes de una carpeta o dataset existente hacia datasets/negatives/ (sin etiquetas)",
    )
    parser.add_argument(
        "--max-negatives-extract",
        type=int,
        default=500,
        help="máximo de imágenes a extraer a datasets/negatives/ (default: 500)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY", "loIOyQL5vSDizjI7ZiT8")

    # Opción de extracción directa de negativos
    if args.extract_negatives:
        origen = Path(args.extract_negatives)
        if not origen.is_dir():
            origen = DATASETS_DIR / args.extract_negatives
        if not origen.is_dir():
            sys.exit(f"[ERROR] No se encuentra la carpeta origen: {args.extract_negatives}")
        total = extract_images_to_negatives(origen, NEGATIVES_DIR, args.max_negatives_extract)
        print(f"✅ Se extrajeron {total} imágenes a {NEGATIVES_DIR}")
        return

    if args.snippet:
        if args.snippet == "-":
            print("Pega el snippet de Roboflow (Ctrl+D para terminar):\n")
            text = sys.stdin.read()
        else:
            text = Path(args.snippet).read_text(encoding="utf-8")
        cfg = parse_snippet(text)
        api_key = cfg.get("api_key") or api_key
        rf = Roboflow(api_key=api_key)
        descargar_dataset(rf, cfg["workspace"], cfg["project"], cfg["project"], cfg["version"])
        return

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    # Descarga de negativos registrados si se solicitó con --download-negatives
    if args.download_negatives:
        print(f"\n[..] Descargando {len(NEGATIVE_DATASETS)} dataset(s) de negativos...")
        for name, cfg in NEGATIVE_DATASETS.items():
            ok = descargar_dataset(rf, cfg["workspace"], cfg["project"], name, cfg.get("version", 1))
            if ok:
                cant = extract_images_to_negatives(DATASETS_DIR / name, NEGATIVES_DIR, args.max_negatives_extract)
                print(f"   -> Extraídas {cant} imágenes negativas de '{name}' a {NEGATIVES_DIR}")
        print(f"✅ Proceso de descarga de negativos completado. Imágenes listas en {NEGATIVES_DIR}.\n")
        if not args.datasets:
            return

    names = args.datasets or list(DATASETS.keys())
    print(f"Iniciando descarga de {len(names)} dataset(s)...")

    exitosos = 0
    for name in names:
        if name not in DATASETS:
            print(f"⚠️ Dataset '{name}' no está en el registro. Opciones: {', '.join(DATASETS.keys())}")
            continue
        cfg = DATASETS[name]
        ok = descargar_dataset(rf, cfg["workspace"], cfg["project"], name, cfg.get("version", 1))
        if ok:
            exitosos += 1

    print(f"\n🎉 Descarga finalizada: {exitosos}/{len(names)} datasets listos.")
    print("Siguiente paso:")
    print("    python scripts/combine_datasets.py\n")


if __name__ == "__main__":
    main()
