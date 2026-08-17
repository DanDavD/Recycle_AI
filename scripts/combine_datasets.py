"""
Combina varios datasets de Roboflow (ya descargados con 04_load_dataset.py) en
un solo dataset para entrenar, quedandose solo con las clases plastic/paper/metal.

Que hace:
  - Lee cada dataset en datasets/<nombre>/ (busca su data.yaml).
  - Mapea los nombres de clase de cada dataset a plastic/paper/metal (ver ALIASES).
    Cualquier clase que no mapee (ej. "glass") se descarta.
  - Por cada imagen, quita del label las cajas de las clases descartadas. Si a una
    imagen no le queda ninguna caja (ej. una foto que solo tenia vidrio), se excluye
    completa.
  - Junta todo en un solo pool, lo baraja (seed fija) y lo vuelve a partir en
    train/valid/test.
  - Escribe datasets/combined/{train,valid,test}/{images,labels}/ + data.yaml.

Uso:
    python scripts/combine_datasets.py
    python scripts/combine_datasets.py --datasets plastic-paper-metal paper-plastic-metal-potato
    python scripts/combine_datasets.py --output combined --val-frac 0.1 --test-frac 0.1

Luego entrenar con:
    python scripts/05_train_model.py --dataset combined
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"

TARGET_CLASSES = ["metal", "paper", "plastic"]

# Nombres de clase de cada dataset externo -> una de TARGET_CLASSES.
# Cualquier nombre que no aparezca aqui ni en TARGET_CLASSES se descarta
# (asi es como se excluye "glass" sin tener que tocar el codigo por dataset).
ALIASES = {
    "plastics": "plastic",
    "pet": "plastic",
    "papers": "paper",
    "cardboard": "paper",
    "paper/cardboard": "paper",
    "metals": "metal",
    "can": "metal",
    "cans": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "tin": "metal",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_class(name: str) -> str | None:
    key = name.strip().lower()
    if key in TARGET_CLASSES:
        return key
    return ALIASES.get(key)


def find_source_datasets() -> list:
    if not DATASETS_DIR.is_dir():
        return []
    found = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if not entry.is_dir() or entry.name == "combined":
            continue
        if (entry / "data.yaml").exists():
            found.append(entry.name)
    return found


def load_class_map(dataset_dir: Path) -> dict:
    """class_idx (del dataset) -> nombre normalizado en TARGET_CLASSES, o None si se descarta."""
    with open(dataset_dir / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    class_map = {}
    dropped = []
    for idx, raw_name in enumerate(names):
        target = normalize_class(raw_name)
        class_map[idx] = target
        if target is None:
            dropped.append(raw_name)
    if dropped:
        print(f"     clases descartadas en {dataset_dir.name}: {dropped}")
    return class_map


def split_dirs(dataset_dir: Path):
    """Genera (images_dir, labels_dir) de cada split presente (train/valid/val/test)."""
    for split in ("train", "valid", "val", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            yield images_dir, labels_dir


def collect_examples(dataset_name: str) -> list:
    """Devuelve [(imagen, lineas_de_label_ya_remapeadas), ...] para un dataset."""
    dataset_dir = DATASETS_DIR / dataset_name
    class_map = load_class_map(dataset_dir)

    examples = []
    total_seen = 0
    for images_dir, labels_dir in split_dirs(dataset_dir):
        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            total_seen += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            kept_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                target = class_map.get(int(parts[0]))
                if target is None:
                    continue
                new_idx = TARGET_CLASSES.index(target)
                kept_lines.append(" ".join([str(new_idx), *parts[1:]]))

            if kept_lines:
                examples.append((image_path, kept_lines))

    print(f"[OK] {dataset_name}: {len(examples)}/{total_seen} imagenes con al menos una caja valida")
    return examples


def write_split(name: str, examples: list, output_dir: Path) -> None:
    images_out = output_dir / name / "images"
    labels_out = output_dir / name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for tag, image_path, lines in examples:
        dest_name = f"{tag}__{image_path.name}"
        shutil.copy2(image_path, images_out / dest_name)
        (labels_out / f"{Path(dest_name).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def write_data_yaml(output_dir: Path) -> None:
    data = {
        "train": str(output_dir / "train" / "images"),
        "val": str(output_dir / "valid" / "images"),
        "test": str(output_dir / "test" / "images"),
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina datasets de Roboflow en uno solo")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="carpetas dentro de datasets/ a combinar (por defecto: todas las que tengan data.yaml)",
    )
    parser.add_argument("--output", default="combined", help="carpeta destino dentro de datasets/")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.datasets or find_source_datasets()
    if not dataset_names:
        sys.exit(
            "[!!] No hay datasets descargados en datasets/.\n"
            "     Corre primero:  python scripts/04_load_dataset.py"
        )
    print(f"[OK] Combinando: {', '.join(dataset_names)}")

    pool = []
    for name in dataset_names:
        for image_path, lines in collect_examples(name):
            pool.append((name, image_path, lines))

    if not pool:
        sys.exit("[!!] Ninguna imagen quedo con clases validas (plastic/paper/metal). Revisa ALIASES.")

    random.Random(args.seed).shuffle(pool)

    n_total = len(pool)
    n_val = int(n_total * args.val_frac)
    n_test = int(n_total * args.test_frac)
    n_train = n_total - n_val - n_test

    output_dir = DATASETS_DIR / args.output
    if output_dir.exists():
        shutil.rmtree(output_dir)

    write_split("train", pool[:n_train], output_dir)
    write_split("valid", pool[n_train:n_train + n_val], output_dir)
    write_split("test", pool[n_train + n_val:], output_dir)
    write_data_yaml(output_dir)

    print(f"\n[OK] Dataset combinado en {output_dir}")
    print(f"     clases: {TARGET_CLASSES}")
    print(f"     train: {n_train} | valid: {n_val} | test: {n_test} | total: {n_total}")
    print(f"\nSiguiente paso:\n    python scripts/05_train_model.py --dataset {args.output}")


if __name__ == "__main__":
    main()
