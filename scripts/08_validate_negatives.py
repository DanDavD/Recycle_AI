"""
Script 08: Validación de Falsos Positivos sobre Imágenes Negativas.

Evalúa un modelo entrenado contra un banco de imágenes que NO contienen
materiales reciclables (madera, tela, restos orgánicos, fondo de la estación, etc.)
y calcula la tasa de falsos positivos (False Positive Rate - FPR).

Uso:
    # 1. Validar modelo más reciente contra datasets/negatives:
    python scripts/08_validate_negatives.py

    # 2. Validar un modelo específico con fotos de prueba particulares:
    python scripts/08_validate_negatives.py --model models/combined_yolov8n.pt --source fotos_prueba/ --conf 0.40

    # 3. Modo estricto para CI/CD (falla si hay algún falso positivo):
    python scripts/08_validate_negatives.py --strict
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
NEGATIVES_DIR = ROOT_DIR / "datasets" / "negatives"
RUNS_DIR = ROOT_DIR / "runs" / "detect" / "eval_negatives"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa la supresión de falsos positivos en imágenes negativas")
    parser.add_argument(
        "--model",
        default=None,
        help="ruta al modelo .pt (por defecto: el más reciente en models/)",
    )
    parser.add_argument(
        "--source",
        default=str(NEGATIVES_DIR),
        help="carpeta o imagen a evaluar (default: datasets/negatives)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="umbral de confianza mínima para considerar detección (default: 0.40)",
    )
    parser.add_argument(
        "--save-dir",
        default=str(RUNS_DIR),
        help="directorio donde guardar imágenes con falsos positivos detectados",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="retorna código de error si se detecta cualquier falso positivo",
    )
    return parser.parse_args()


def collect_images(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path] if source_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if source_path.is_dir():
        return [
            p for p in sorted(source_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []


def main():
    args = parse_args()

    # 1. Determinar modelo
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_trained_weights()

    if not model_path or not model_path.exists():
        sys.exit(
            f"[ERROR] No se encontró el modelo en {model_path}.\n"
            "Indica la ruta con --model models/tu_modelo.pt o entrena primero con 05_train_model.py"
        )

    # 2. Recolectar imágenes
    source_path = Path(args.source)
    images = collect_images(source_path)
    if not images:
        sys.exit(
            f"[ERROR] No se encontraron imágenes en {source_path}.\n"
            "Asegúrate de colocar fotos (.jpg, .png, etc.) en datasets/negatives/ o especificar --source"
        )

    print("\n" + "=" * 68)
    print(" 🧪 EVALUACIÓN DE SUPRESIÓN DE FALSOS POSITIVOS (RECHAZO DE OBJETOS)")
    print("=" * 68)
    print(f"📦 Modelo:            {model_path.name}")
    print(f"📁 Directorio prueba: {source_path}")
    print(f"🎯 Umbral confianza:  {args.conf:.2f}")
    print(f"🖼️  Total imágenes:    {len(images)}")
    print("=" * 68 + "\n")

    model = YOLO(str(model_path))
    class_names = model.names

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    false_positives = []
    class_fp_counts = Counter()
    confidences = []

    for img_path in images:
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        r = results[0]
        boxes = r.boxes

        if len(boxes) > 0:
            # Hubo falso positivo
            detections = []
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf_val = float(b.conf[0].item())
                cls_name = class_names.get(cls_id, str(cls_id))
                class_fp_counts[cls_name] += 1
                confidences.append(conf_val)
                detections.append(f"{cls_name} ({conf_val:.2f})")

            false_positives.append((img_path, detections))

            # Guardar visualización anotada del falso positivo para inspección
            annotated_frame = r.plot()
            out_name = save_dir / f"FP_{img_path.stem}.jpg"
            cv2.imwrite(str(out_name), annotated_frame)

    total_images = len(images)
    total_fp = len(false_positives)
    fpr = (total_fp / total_images * 100) if total_images > 0 else 0

    print("-" * 68)
    print(" 📊 RESULTADOS DE LA EVALUACIÓN")
    print("-" * 68)
    print(f"Total imágenes analizadas:           {total_images}")
    print(f"Imágenes con falsos positivos (FP):  {total_fp}")
    print(f"Tasa de falsos positivos (FPR):      {fpr:.2f}%")
    print(f"Tasa de rechazo correcto (TNR):      {100.0 - fpr:.2f}%")

    if total_fp > 0:
        print("\n⚠️ Desglose de falsas alarmas por clase asignada erróneamente:")
        for cname, count in class_fp_counts.most_common():
            print(f"   - {cname:<10}: {count} detecciones")

        print(f"\n📈 Confianza de los falsos positivos:")
        print(f"   - Mínima: {min(confidences):.2f}")
        print(f"   - Media:  {sum(confidences)/len(confidences):.2f}")
        print(f"   - Máxima: {max(confidences):.2f}")

        print(f"\n🖼️  Imágenes anotadas de los falsos positivos guardadas en:\n   {save_dir}\n")
    else:
        print("\n🎉 ¡Excelente! Ninguna imagen generó falsos positivos por encima del umbral.\n")

    print("=" * 68)

    if args.strict and total_fp > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
