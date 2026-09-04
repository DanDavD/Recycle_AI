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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def find_source_datasets() -> list:
    if not DATASETS_DIR.is_dir():
        return []
    return [
        e.name
        for e in sorted(DATASETS_DIR.iterdir())
        if e.is_dir() and e.name != "combined" and (e / "data.yaml").exists()
    ]


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

    for s in splits:
        img_d, lbl_d = dataset_dir / s / "images", dataset_dir / s / "labels"
        if img_d.exists():
            split_images[s] = len([f for f in img_d.iterdir() if f.is_file()])
        if lbl_d.exists():
            for lbl in lbl_d.glob("*.txt"):
                with open(lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            split_counts[s][int(line.split()[0])] += 1

    total_counts = Counter()
    for s in splits:
        total_counts.update(split_counts[s])
    total_objs = sum(total_counts.values())

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
    print(f"\n🖼️  Total imágenes: {sum(split_images.values())}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina datasets de Roboflow en uno solo")
    parser.add_argument("--output", default="combined", help="carpeta destino dentro de datasets/")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
                        if target_class is not None:
                            valid_boxes.append(f"{target_indices[target_class]} {' '.join(parts[1:])}")
                if valid_boxes:
                    pool.append((name, img_path, valid_boxes))

    random.seed(args.seed)
    random.shuffle(pool)

    # Mostrar conteo total
    counts = {c: 0 for c in TARGET_CLASSES}
    for _, _, boxes in pool:
        for b in boxes:
            counts[TARGET_CLASSES[int(b.split()[0])]] += 1
    print(f"\n[INFO] Cajas totales por clase: {counts}")

    n_total = len(pool)
    n_test = int(n_total * args.test_frac)
    n_val = int(n_total * args.val_frac)

    out_dir = DATASETS_DIR / args.output
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for s_name, samples in [("train", pool[n_test + n_val:]), ("valid", pool[n_test:n_test + n_val]), ("test", pool[:n_test])]:
        i_out, l_out = out_dir / s_name / "images", out_dir / s_name / "labels"
        i_out.mkdir(parents=True, exist_ok=True)
        l_out.mkdir(parents=True, exist_ok=True)
        for i, (tag, src_img, boxes) in enumerate(samples):
            ext = src_img.suffix.lower()
            dst = f"{tag}_{src_img.stem}_{i:05d}"
            shutil.copy2(src_img, i_out / f"{dst}{ext}")
            with open(l_out / f"{dst}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(boxes) + "\n")

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": TARGET_CLASSES,
            },
            f,
        )
    print(f"\n[OK] Dataset combinado exitosamente en {out_dir} con {n_total} imágenes.")

    # Imprimir automáticamente la tabla de distribución
    print_distribution_table(out_dir)

    print("Siguiente paso:")
    print(f"    python scripts/05_train_model.py --dataset {args.output} --epochs 100\n")


if __name__ == "__main__":
    main()
