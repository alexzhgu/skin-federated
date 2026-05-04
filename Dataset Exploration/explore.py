"""
Phase 1: Inventory & Class Taxonomy
"""

from pathlib import Path
import json
import os
import kagglehub
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Resolving dataset paths via kagglehub")
DATASET_1_PATH = Path(kagglehub.dataset_download(
    "anieetorudofia/skin-diseases-cancer-comprehensive-dataset"))
DATASET_1_NAME = "comprehensive"

DATASET_2_PATH = Path(kagglehub.dataset_download(
    "muhammadabdulsami/massive-skin-disease-balanced-dataset"))
DATASET_2_NAME = "balanced"

print(f"  {DATASET_1_NAME}: {DATASET_1_PATH}")
print(f"  {DATASET_2_NAME}: {DATASET_2_PATH}")

OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYWORDS = {"train", "test", "val", "valid", "validation"}

def count_images(directory: Path) -> int:
    count = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTS:
                    count += 1
    except (PermissionError, FileNotFoundError):
        pass
    return count


def find_dataset_root(root: Path, max_depth: int = 4) -> Path:
    current = root
    for _ in range(max_depth):
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if not subdirs:
            return current
        if any(d.name.lower() in SPLIT_KEYWORDS for d in subdirs):
            return current
        if any(count_images(d) > 0 for d in subdirs):
            return current
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        return current
    return current


def inventory_dataset(root: Path, name: str) -> pd.DataFrame:
    """
    Walk dataset directory

    Expected structures:
        root/class_name/img.jpg
        root/{train,test,val}/class_name/img.jpg
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    root = find_dataset_root(root)
    print(f"  resolved {name} root -> {root}")

    top_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    has_splits = any(d.name.lower() in SPLIT_KEYWORDS for d in top_dirs)

    rows = []
    if has_splits:
        for split_dir in top_dirs:
            class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            for class_dir in tqdm(class_dirs, desc=f"{name}/{split_dir.name}"):
                rows.append({
                    "dataset": name,
                    "split": split_dir.name,
                    "class": class_dir.name,
                    "count": count_images(class_dir),
                })
    else:
        for class_dir in tqdm(top_dirs, desc=name):
            rows.append({
                "dataset": name,
                "split": "all",
                "class": class_dir.name,
                "count": count_images(class_dir),
            })

    return pd.DataFrame(rows)


def normalise_class_name(name: str) -> str:
    n = name.lower().strip()
    for sep in ["_", "-", ".", "/"]:
        n = n.replace(sep, " ")
    return " ".join(n.split())


def compare_class_taxonomies(df1: pd.DataFrame, df2: pd.DataFrame,
                              name1: str, name2: str) -> dict:
    classes_1 = set(df1["class"].unique())
    classes_2 = set(df2["class"].unique())

    norm_1 = {normalise_class_name(c) for c in classes_1}
    norm_2 = {normalise_class_name(c) for c in classes_2}

    return {
        "overlap_normalised": sorted(norm_1 & norm_2),
        f"only_in_{name1}": sorted(norm_1 - norm_2),
        f"only_in_{name2}": sorted(norm_2 - norm_1),
        f"raw_classes_{name1}": sorted(classes_1),
        f"raw_classes_{name2}": sorted(classes_2),
    }


def plot_class_distribution(df: pd.DataFrame, dataset_name: str, out_path: Path):
    counts = df.groupby("class")["count"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.25)))
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of images")
    ax.set_title(f"{dataset_name}  -  {counts.sum():,} images across {len(counts)} classes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def summarise(df: pd.DataFrame, name: str) -> None:
    total = df["count"].sum()
    classes = df["class"].nunique()
    splits = sorted(df["split"].unique())
    per_class = df.groupby("class")["count"].sum()
    imbalance = per_class.max() / max(per_class.min(), 1)

    print(f"\n=== {name} ===")
    print(f"Total images:        {total:,}")
    print(f"Classes:             {classes}")
    print(f"Splits:              {splits}")
    print(f"Smallest class:      {per_class.idxmin()} ({per_class.min():,})")
    print(f"Largest class:       {per_class.idxmax()} ({per_class.max():,})")
    print(f"Imbalance (max/min): {imbalance:.1f}x")

def main():
    print("Inventorying dataset 1")
    df1 = inventory_dataset(DATASET_1_PATH, DATASET_1_NAME)
    df1.to_csv(OUTPUT_DIR / f"inventory_{DATASET_1_NAME}.csv", index=False)

    print("\nInventorying dataset 2")
    df2 = inventory_dataset(DATASET_2_PATH, DATASET_2_NAME)
    df2.to_csv(OUTPUT_DIR / f"inventory_{DATASET_2_NAME}.csv", index=False)

    summarise(df1, DATASET_1_NAME)
    summarise(df2, DATASET_2_NAME)

    taxonomy = compare_class_taxonomies(df1, df2, DATASET_1_NAME, DATASET_2_NAME)
    with open(OUTPUT_DIR / "class_taxonomy_comparison.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"\n=== Class taxonomy comparison ===")
    print(f"Shared (after normalisation): {len(taxonomy['overlap_normalised'])}")
    print(f"Only in {DATASET_1_NAME}:       {len(taxonomy[f'only_in_{DATASET_1_NAME}'])}")
    print(f"Only in {DATASET_2_NAME}:       {len(taxonomy[f'only_in_{DATASET_2_NAME}'])}")

    plot_class_distribution(df1, DATASET_1_NAME, OUTPUT_DIR / f"class_dist_{DATASET_1_NAME}.png")
    plot_class_distribution(df2, DATASET_2_NAME, OUTPUT_DIR / f"class_dist_{DATASET_2_NAME}.png")

    print(f"\nOutputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()