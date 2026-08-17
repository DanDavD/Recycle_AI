"""
Script 05: entrena el modelo YOLOv8 de clasificacion de reciclaje.

Requiere que el dataset ya este descargado con scripts/04_load_dataset.py.

Uso:
    python scripts/05_train_model.py                          # valores por defecto
    python scripts/05_train_model.py --epochs 150 --batch 8   # ajustes manuales
    python scripts/05_train_model.py --dataset plastic-paper-metal

Los pesos finales se copian a models/<nombre_run>.pt y el run completo
(graficas, matriz de confusion, metricas) queda en runs/detect/<nombre_run>/.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "runs"

DEFAULT_DATASET = "plastic-paper-metal"
# yolov8n: el mas rapido, pensado para correr en vivo con la camara.
# Si quieres mas precision y aguanta la VRAM, prueba yolov8s.pt.
DEFAULT_WEIGHTS = "yolov8n.pt"

# Defaults calibrados para la RTX 3060 Laptop (6 GB de VRAM).
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 4  # en Windows subir esto suele dar problemas con el DataLoader


def find_data_yaml(dataset_name: str) -> Path:
    """Devuelve el data.yaml del dataset y corrige sus rutas si hace falta.

    Roboflow escribe rutas relativas ('../train/images') que solo funcionan si
    ejecutas desde dentro de la carpeta del dataset. Las reescribimos a rutas
    absolutas para poder lanzar el entrenamiento desde cualquier sitio.
    """
    dataset_dir = DATASETS_DIR / dataset_name
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        sys.exit(
            f"No encuentro {data_yaml}\n"
            f"Descarga el dataset primero:  python scripts/04_load_dataset.py {dataset_name}"
        )

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cambiado = False
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        split_dir = dataset_dir / split.replace("val", "valid") / "images"
        if not split_dir.exists():
            # algunos exports usan 'val' en vez de 'valid'
            split_dir = dataset_dir / split / "images"
        if split_dir.exists() and data[split] != str(split_dir):
            data[split] = str(split_dir)
            cambiado = True

    if cambiado:
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"[OK] Rutas de {data_yaml.name} convertidas a absolutas")

    print(f"[OK] Dataset: {dataset_name} | clases ({data.get('nc')}): {data.get('names')}")
    return data_yaml


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] Entrenando en GPU: {name} ({vram:.1f} GB)")
        return "0"
    print("[!!] No hay GPU disponible, se entrenara en CPU (mucho mas lento)")
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de reciclaje")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="carpeta dentro de datasets/")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="pesos base (yolov8n.pt, yolov8s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="baja a 8 si da error de memoria (CUDA OOM)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--name", default=None, help="nombre del run (por defecto: <dataset>_<weights>)")
    parser.add_argument("--resume", action="store_true", help="reanuda el ultimo entrenamiento interrumpido")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = find_data_yaml(args.dataset)
    device = pick_device()
    run_name = args.name or f"{args.dataset}_{Path(args.weights).stem}"

    # Si los pesos base estan en la raiz del proyecto usamos esa copia y evitamos
    # que ultralytics los vuelva a descargar.
    weights = args.weights
    local_weights = ROOT_DIR / weights
    if local_weights.exists():
        weights = str(local_weights)

    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(RUNS_DIR / "detect"),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        patience=25,     # corta si no mejora en 25 epochs
        amp=True,        # precision mixta: menos VRAM y mas velocidad
        cache=False,     # con 2234 imagenes cabe en RAM, pero cache=True se come 8+ GB
        plots=True,
        seed=0,
    )

    metrics = model.val()
    print("\n=== Resultados en validacion ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

    best = RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
    if best.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        destino = MODELS_DIR / f"{run_name}.pt"
        shutil.copy2(best, destino)
        print(f"\n[OK] Mejor modelo copiado a {destino}")
        print(f"[OK] Graficas y metricas en {best.parent.parent}")


if __name__ == "__main__":
    main()
