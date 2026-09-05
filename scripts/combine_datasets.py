import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"

TARGET_CLASSES = ["metal", "paper", "plastic"]

ALIASES = {
    # --- PLÁSTICO ---
    "plastics": "plastic", "pet": "plastic", "plastic bottle": "plastic",
    "plastic-bottle": "plastic", "plastic_bottle": "plastic", "bottle": "plastic",
    "bottles": "plastic", "plastic bag": "plastic", "plastic-bag": "plastic",
    "plastic_bag": "plastic", "bag": "plastic", "plastic cup": "plastic",
    "cup": "plastic", "plastic container": "plastic", "container": "plastic",
    "hdpe": "plastic", "plastic wrapper": "plastic", "wrapper": "plastic", "straw": "plastic",

    # --- PAPEL Y CARTÓN ---
    "papers": "paper", "cardboard": "paper", "paper/cardboard": "paper",
    "paper-cardboard": "paper", "paper bag": "paper", "paper cup": "paper",
    "carton": "paper", "box": "paper", "boxes": "paper", "newspaper": "paper", "magazine": "paper",

    # --- METAL ---
    "metals": "metal", "can": "metal", "cans": "metal", "drink can": "metal",
    "food can": "metal", "tin can": "metal", "tin_can": "metal", "aluminum": "metal",
    "aluminium": "metal", "aluminum can": "metal", "tin": "metal", "foil": "metal",
    "metal cap": "metal", "bottle cap": "metal", "cap": "metal", "metal cans": "metal", "metal-cans": "metal",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class(name: str) -> str | None:
    key = name.strip().lower()
    if key in TARGET_CLASSES:
        return key
    if key in ALIASES:
        return ALIASES[key]
    for k, v in ALIASES.items():
        if k in key:
            return v
    return None


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Convierte un polígono de segmentación [x1, y1, x2, y2, ...] a bounding box [xc, yc, w, h] normalizado."""
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    xmin = max(0.0, min(xs))
    xmax = min(1.0, max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(1.0, max(ys))
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0.001 or h <= 0.001:
        return None
    xc = xmin + w / 2.0
    yc = ymin + h / 2.0
    return xc, yc, w, h


def find_source_datasets() -> list:
    if not DATASETS_DIR.is_dir():
        return []
    return [
        e.name
        for e in sorted(DATASETS_DIR.iterdir())
        if e.is_dir() and e.name not in ("combined", "negatives") and (e / "data.yaml").exists()
    ]


def load_negative_images(negatives_dir: Path) -> list:
    """Carga todas las imágenes de fondo/objetos ajenos de la carpeta de negativos sin cajas."""
    if not negatives_dir.is_dir():
        return []
    neg_images = []
    for path in sorted(negatives_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # Si está en una subcarpeta (ej. wood, fabric, organic), usar el nombre de la subcarpeta
            cat = path.parent.name if path.parent != negatives_dir else "bg"
            neg_images.append((f"neg_{cat}", path, []))
    return neg_images


def load_class_map(dataset_dir: Path) -> dict:
    with open(dataset_dir / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return {idx: normalize_class(n) for idx, n in enumerate(names)}


def split_dirs(dataset_dir: Path):
    for split in ("train", "valid", "val", "test"):
        i_dir, l_dir = dataset_dir / split / "images", dataset_dir / split / "labels"
        if i_dir.is_dir() and l_dir.is_dir():
            yield split, i_dir, l_dir


def print_distribution_table(dataset_dir: Path):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print("⚠️ Primero ejecuta la combinación de datasets.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names", {})
    class_map = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}

    splits = ["train", "valid", "test"]
    split_counts = {s: Counter() for s in splits}
    split_images = {s: 0 for s in splits}
    split_negatives = {s: 0 for s in splits}

    for s in splits:
        img_d, lbl_d = dataset_dir / s / "images", dataset_dir / s / "labels"
        if img_d.exists():
            split_images[s] = len([f for f in img_d.iterdir() if f.is_file()])
        if lbl_d.exists():
            for lbl in lbl_d.glob("*.txt"):
                content = lbl.read_text(encoding="utf-8").strip()
                if not content:
                    split_negatives[s] += 1
                else:
                    for line in content.splitlines():
                        if line.strip():
                            split_counts[s][int(line.split()[0])] += 1

    total_counts = Counter()
    for s in splits:
        total_counts.update(split_counts[s])
    total_objs = sum(total_counts.values())
    total_imgs = sum(split_images.values())
    total_negs = sum(split_negatives.values())

    print("\n" + "=" * 66)
    print(" 📊 DISTRIBUCIÓN FINAL DEL DATASET COMBINADO")
    print("=" * 66)
    print(f"{'Categoría':<12} | {'Train':<7} | {'Val':<6} | {'Test':<6} | {'TOTAL':<7} | {'% Total':<8}")
    print("-" * 66)
    icons = {"metal": "🥫", "paper": "📄", "plastic": "🧴"}
    for cid, cname in sorted(class_map.items()):
        tot = total_counts[cid]
        pct = (tot / total_objs * 100) if total_objs > 0 else 0
        print(f"{icons.get(cname.lower(), '📦')} {cname.capitalize():<9} | {split_counts['train'][cid]:<7} | {split_counts['valid'][cid]:<6} | {split_counts['test'][cid]:<6} | {tot:<7} | {pct:>6.1f}%")
    print("-" * 66)
    print(f"{'TOTAL OBJETOS':<12} | {sum(split_counts['train'].values()):<7} | {sum(split_counts['valid'].values()):<6} | {sum(split_counts['test'].values()):<6} | {total_objs:<7} | 100.0%")
    print("=" * 66)

    if total_negs > 0:
        neg_pct = (total_negs / total_imgs * 100) if total_imgs > 0 else 0
        print(f"🚫 Negativos (fondo/ajenos): {total_negs} imágenes ({neg_pct:.1f}% del total de imágenes)")
        print(f"   Train: {split_negatives['train']:<5} | Val: {split_negatives['valid']:<5} | Test: {split_negatives['test']:<5}")
        print("=" * 66)

    print(f"🖼️  Total imágenes (positivas + negativas): {total_imgs}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina datasets de Roboflow en uno solo con balanceo y soporte de negativos")
    parser.add_argument("--output", default="combined", help="carpeta destino dentro de datasets/")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="desactiva el balanceo estricto (usa todos los datos disponibles sin recortar)",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="fuerza un tope manual de cajas por clase (por defecto: el mínimo entre las clases)",
    )
    parser.add_argument(
        "--negatives-dir",
        default=str(DATASETS_DIR / "negatives"),
        help="carpeta con imágenes negativas (fondo/objetos ajenos sin etiquetas)",
    )
    parser.add_argument(
        "--negatives-ratio",
        type=float,
        default=0.10,
        help="proporción de negativos respecto al total de imágenes positivas (default: 0.10 i.e. 10%%)",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=None,
        help="tope máximo de imágenes negativas a incluir",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_names = find_source_datasets()
    if not dataset_names:
        print("[ERROR] No hay datasets en datasets/. Descárgalos primero con: python scripts/04_load_dataset.py")
        sys.exit(1)

    print(f"[OK] Combinando: {', '.join(dataset_names)}")
    target_indices = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

    pool = []
    for name in dataset_names:
        ds_dir = DATASETS_DIR / name
        class_map = load_class_map(ds_dir)
        for split, images_dir, labels_dir in split_dirs(ds_dir):
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                if not lbl_path.is_file():
                    continue
                valid_boxes = []
                with open(lbl_path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        target_class = class_map.get(int(parts[0]))
                        if target_class is None:
                            continue
                        
                        # Si es bounding box estándar: 1 clase + 4 coordenadas
                        if len(parts) == 5:
                            try:
                                xc, yc, w, h = (float(v) for v in parts[1:5])
                                if w > 0.001 and h > 0.001:
                                    valid_boxes.append(f"{target_indices[target_class]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            except ValueError:
                                continue
                        # Si es polígono de segmentación: 1 clase + n puntos (x1 y1 x2 y2 ...)
                        elif len(parts) > 5:
                            try:
                                coords = [float(p) for p in parts[1:]]
                                bbox = polygon_to_bbox(coords)
                                if bbox is not None:
                                    xc, yc, w, h = bbox
                                    valid_boxes.append(f"{target_indices[target_class]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            except ValueError:
                                continue
                if valid_boxes:
                    pool.append((name, img_path, valid_boxes))

    random.seed(args.seed)
    random.shuffle(pool)

    # Contar cajas positivas antes de balancear
    counts = {c: 0 for c in TARGET_CLASSES}
    for _, _, boxes in pool:
        for b in boxes:
            counts[TARGET_CLASSES[int(b.split()[0])]] += 1
    print(f"\n[INFO] Cajas encontradas antes de balanceo: {counts}")

    # Balanceo por tope uniforme
    if not args.no_balance:
        target_per_class = args.target_per_class or (min(counts.values()) if counts.values() else 0)
        print(f"[INFO] Balanceando clases a tope uniforme: {target_per_class} cajas por clase")
        balanced_pool = []
        cur = {c: 0 for c in TARGET_CLASSES}
        for tag, img, boxes in pool:
            cls_in_img = [TARGET_CLASSES[int(b.split()[0])] for b in boxes]
            if all(cur[c] >= target_per_class for c in cls_in_img):
                continue
            balanced_pool.append((tag, img, boxes))
            for c in cls_in_img:
                cur[c] += 1
        pool = balanced_pool
        print(f"[INFO] Cajas balanceadas finales: {cur}")
    else:
        print("[INFO] Balanceo desactivado (--no-balance): usando todas las imágenes positivas disponibles.")

    num_positives = len(pool)
    print(f"[INFO] Total imágenes positivas seleccionadas: {num_positives}")

    # Cargar y seleccionar imágenes negativas (fondo/objetos ajenos sin etiquetas)
    negatives_dir = Path(args.negatives_dir)
    neg_pool = load_negative_images(negatives_dir)
    selected_negatives = []
    if neg_pool:
        target_negatives = int(num_positives * args.negatives_ratio)
        if args.max_negatives is not None:
            target_negatives = min(target_negatives, args.max_negatives)
        random.shuffle(neg_pool)
        selected_negatives = neg_pool[:target_negatives] if target_negatives < len(neg_pool) else neg_pool
        actual_ratio = (len(selected_negatives) / num_positives * 100) if num_positives > 0 else 0
        print(f"[INFO] Incorporando {len(selected_negatives)} imágenes negativas desde {negatives_dir} ({actual_ratio:.1f}% de positivos)")
    else:
        print(f"[INFO] No se encontraron imágenes negativas en {negatives_dir} (opcional para reducir falsos positivos)")

    # Separar splits (train, valid, test) para positivos
    n_pos_test = int(num_positives * args.test_frac)
    n_pos_val = int(num_positives * args.val_frac)
    train_pos = pool[n_pos_test + n_pos_val:]
    val_pos = pool[n_pos_test:n_pos_test + n_pos_val]
    test_pos = pool[:n_pos_test]

    # Separar splits de manera idéntica para negativos
    n_neg = len(selected_negatives)
    n_neg_test = int(n_neg * args.test_frac)
    n_neg_val = int(n_neg * args.val_frac)
    train_neg = selected_negatives[n_neg_test + n_neg_val:]
    val_neg = selected_negatives[n_neg_test:n_neg_test + n_neg_val]
    test_neg = selected_negatives[:n_neg_test]

    train_samples = train_pos + train_neg
    val_samples = val_pos + val_neg
    test_samples = test_pos + test_neg

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    out_dir = DATASETS_DIR / args.output
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for s_name, samples in [("train", train_samples), ("valid", val_samples), ("test", test_samples)]:
        i_out, l_out = out_dir / s_name / "images", out_dir / s_name / "labels"
        i_out.mkdir(parents=True, exist_ok=True)
        l_out.mkdir(parents=True, exist_ok=True)
        for i, (tag, src_img, boxes) in enumerate(samples):
            ext = src_img.suffix.lower()
            dst = f"{tag}_{src_img.stem}_{i:05d}"
            shutil.copy2(src_img, i_out / f"{dst}{ext}")
            lbl_file = l_out / f"{dst}.txt"
            if boxes:
                with open(lbl_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(boxes) + "\n")
            else:
                # Fondo o negativo sin objetos: archivo de etiquetas vacío (YOLOv8 background image)
                open(lbl_file, "w", encoding="utf-8").close()

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(TARGET_CLASSES),
                "names": TARGET_CLASSES,
            },
            f,
        )
    n_total_imgs = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"\n[OK] Dataset combinado exitosamente en {out_dir} con {n_total_imgs} imágenes.")

    # Imprimir automáticamente la tabla de distribución
    print_distribution_table(out_dir)

    print("Siguiente paso:")
    print(f"    python scripts/05_train_model.py --dataset {args.output} --epochs 100\n")


if __name__ == "__main__":
    main()
