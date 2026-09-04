"""
Script 05: Entrena el modelo YOLOv8 de clasificación de reciclaje.

Requiere que el dataset ya esté descargado y combinado con:
    python scripts/04_load_dataset.py
    python scripts/combine_datasets.py

Uso:
    # 1. Entrenar modelo nuevo desde cero:
    python scripts/05_train_model.py --dataset combined --epochs 100

    # 2. Añadir más aprendizaje sobre el modelo actual (Fine-Tuning):
    python scripts/05_train_model.py --dataset combined --epochs 100 --fine-tune

Los pesos finales se copian a models/<run>.pt y el reporte completo
queda en runs/detect/<run>/.
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

DEFAULT_DATASET = "combined"
DEFAULT_WEIGHTS = "yolov8n.pt"

# Defaults calibrados para entrenamiento
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 4

# Data Augmentation óptimo para basureros inteligentes
DEFAULT_DEGREES = 180.0
DEFAULT_FLIPUD = 0.5


def find_latest_trained_weights() -> Path | None:
    """Busca el modelo .pt entrenado más reciente en models/."""
    if not MODELS_DIR.exists():
        return None
    candidatos = [
        p for p in MODELS_DIR.glob("*.pt")
        if p.name not in ("yolov8n.pt", "yolov8s.pt", "yolo26n.pt")
    ]
    if not candidatos:
        return None
    candidatos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidatos[0]


def get_next_model_path(base_name: str) -> Path:
    """Genera un nombre sin sobreescribir checkpoints anteriores (ej. combined_yolov8n3.pt)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"{base_name}.pt"
    if not target.exists():
        return target
    counter = 2
    while True:
        target = MODELS_DIR / f"{base_name}{counter}.pt"
        if not target.exists():
            return target
        counter += 1


def find_data_yaml(dataset_name: str) -> Path:
    """Devuelve el data.yaml del dataset y corrige sus rutas si hace falta."""
    dataset_dir = DATASETS_DIR / dataset_name
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        sys.exit(
            f"No encuentro {data_yaml}\n"
            f"Prepara el dataset primero corriendo:\n"
            f"    python scripts/04_load_dataset.py\n"
            f"    python scripts/combine_datasets.py"
        )

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cambiado = False
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        split_dir = dataset_dir / split.replace("val", "valid") / "images"
        if not split_dir.exists():
            split_dir = dataset_dir / split / "images"
        if split_dir.exists() and data[split] != str(split_dir):
            data[split] = str(split_dir)
            cambiado = True

    if cambiado:
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"[OK] Rutas de {data_yaml.name} convertidas a absolutas")

    nc = data.get("nc") or (len(data.get("names", [])) if isinstance(data.get("names"), (list, dict)) else "?")
    print(f"[OK] Dataset: {dataset_name} | clases ({nc}): {data.get('names')}")
    return data_yaml


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] Entrenando en GPU: {name} ({vram:.1f} GB)")
        return "0"
    print("[!!] No hay GPU disponible, se entrenará en CPU (más lento)")
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de reciclaje YOLOv8")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="carpeta dentro de datasets/ (def: combined)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="pesos base (yolov8n.pt o ruta a un .pt previo)")
    parser.add_argument(
        "--fine-tune",
        "--from-latest",
        action="store_true",
        help="Continúa entrenando a partir del modelo más reciente en models/ (Fine-Tuning)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="baja a 8 si da error de memoria VRAM")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--degrees",
        type=float,
        default=DEFAULT_DEGREES,
        help="rango de rotación aleatoria en grados",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=DEFAULT_FLIPUD,
        help="probabilidad de volteo vertical (objetos al revés)",
    )
    parser.add_argument("--name", default=None, help="nombre del run")
    parser.add_argument("--resume", action="store_true", help="reanuda el último entrenamiento interrumpido")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = find_data_yaml(args.dataset)
    device = pick_device()

    weights = args.weights
    if args.fine_tune or weights == "latest":
        latest_pt = find_latest_trained_weights()
        if latest_pt:
            weights = str(latest_pt)
            print(f"\n[OK] 🚀 Modo Fine-Tuning: Continuando a partir de {latest_pt.name}")
        else:
            print(f"[!!] No hay modelos previos en models/, entrenando desde base: {DEFAULT_WEIGHTS}")
            weights = DEFAULT_WEIGHTS

    run_name = args.name or f"{args.dataset}_{Path(weights).stem}"
    print(f"[OK] Augmentation: rotación +-{args.degrees:g}° | volteo vertical p={args.flipud:g}")

    if args.resume:
        weights_resume = RUNS_DIR / "detect" / run_name / "weights" / "last.pt"
        if not weights_resume.exists():
            sys.exit(f"No encuentro {weights_resume} para reanudar.")
        print(f"[OK] Reanudando desde {weights_resume}")
        weights = str(weights_resume)
    else:
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
        patience=25,     # detiene si no mejora en 25 epochs
        amp=True,        # precisión mixta para acelerar en GPU
        cache=False,
        plots=True,
        seed=0,
        degrees=args.degrees,
        flipud=args.flipud,
    )

    metrics = model.val()
    print("\n=== Resultados en validación ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

    best = RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
    if best.exists():
        destino = get_next_model_path(run_name)
        shutil.copy2(best, destino)
        print(f"\n✅ [EXITO] Mejor modelo guardado en: {destino}")
        print(f"📊 Gráficas y métricas guardadas en: {best.parent.parent}\n")


if __name__ == "__main__":
    main()
