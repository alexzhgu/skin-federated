"""
Stratified sample of the dataset into a reusable manifest.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Config =====
PER_CLASS = 500
MIN_PER_CLASS = 50
SEED = 42


NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.1

DATASET_KAGGLE = "muhammadabdulsami/massive-skin-disease-balanced-dataset"
SPLIT_KEYWORDS = {"train", "test", "val", "valid", "validation"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def find_dataset_root(root: Path, max_depth: int = 4) -> Path:
    current = root
    for _ in range(max_depth):
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if not subdirs:
            return current
        if any(d.name.lower() in SPLIT_KEYWORDS for d in subdirs):
            return current
        if any(any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
               for d in subdirs):
            return current
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        return current
    return current


def collect_filepaths(root: Path) -> pd.DataFrame:
    root = find_dataset_root(root)
    print(f"  resolved root -> {root}")

    rows = []
    top_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    has_splits = any(d.name.lower() in SPLIT_KEYWORDS for d in top_dirs)

    if has_splits:
        for split_dir in top_dirs:
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    for f in class_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                            rows.append({"filepath": str(f), "class": class_dir.name})
    else:
        for class_dir in top_dirs:
            for f in class_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    rows.append({"filepath": str(f), "class": class_dir.name})
    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, per_class, min_per_class, seed) -> pd.DataFrame:
    full_counts = df["class"].value_counts()
    keep_classes = full_counts[full_counts >= min_per_class].index
    dropped = full_counts[full_counts < min_per_class]
    if len(dropped):
        print(f"  Dropping {len(dropped)} classes below MIN_PER_CLASS={min_per_class}: "
              f"{dict(dropped)}")
    df = df[df["class"].isin(keep_classes)].copy()

    if per_class is None:
        sampled = df.copy()
    else:
        parts = []
        short_classes = []
        for cls, group in df.groupby("class", sort=False):
            n = min(len(group), per_class)
            if len(group) < per_class:
                short_classes.append((cls, len(group)))
            parts.append(group.sample(n=n, random_state=seed))
        sampled = pd.concat(parts, ignore_index=True)
        if short_classes:
            print(f"  {len(short_classes)} classes had fewer than PER_CLASS={per_class} "
                  f"images and used everything available: {dict(short_classes)}")

    return sampled.reset_index(drop=True)


def plot_class_proportions(full_df: pd.DataFrame, sampled_df: pd.DataFrame, out_path: Path):
    full_p = full_df["class"].value_counts(normalize=True)
    samp_p = sampled_df["class"].value_counts(normalize=True)
    classes = sorted(set(full_p.index) | set(samp_p.index))
    full_p = full_p.reindex(classes, fill_value=0)
    samp_p = samp_p.reindex(classes, fill_value=0)

    max_abs_diff = (full_p - samp_p).abs().max()

    fig, ax = plt.subplots(figsize=(12, max(6, len(classes) * 0.25)))
    y = np.arange(len(classes))
    ax.barh(y - 0.2, full_p.values, height=0.4, label="full")
    ax.barh(y + 0.2, samp_p.values, height=0.4, label="sampled")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Proportion of dataset")
    ax.set_title(f"Class proportions: full vs sampled  "
                 f"(max abs diff = {max_abs_diff:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Max abs proportion difference (full vs sampled): {max_abs_diff:.4f}")


def dirichlet_partition(df: pd.DataFrame, num_clients: int, alpha: float, seed: int):
    """
    Hsu et al.-style partitioning: for each class, draw a Dirichlet(alpha) over
    clients, then split that class's indices accordingly.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(df["class"].unique())
    counts = np.zeros((num_clients, len(classes)), dtype=int)

    for ci, cls in enumerate(classes):
        idxs = df.index[df["class"] == cls].to_numpy().copy()
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for k in range(num_clients):
            counts[k, ci] = len(splits[k])

    return counts, classes


def plot_partition_heatmap(counts: np.ndarray, classes: list, out_path: Path,
                            alpha: float):
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.3), 6))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels([f"client {k}" for k in range(counts.shape[0])])
    ax.set_xlabel("Class")
    ax.set_title(f"Worst-case Dirichlet partition (alpha={alpha}): images per (client, class)")
    fig.colorbar(im, ax=ax, label="images")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    zero_cells = (counts == 0).sum()
    total_cells = counts.size
    min_per_client = counts.sum(axis=1).min()
    max_per_client = counts.sum(axis=1).max()
    print(f"  Partition: {zero_cells}/{total_cells} (client, class) cells are empty "
          f"({zero_cells / total_cells:.1%})")
    print(f"  Smallest client has {min_per_client:,} images, "
          f"largest has {max_per_client:,}")


def main():
    print("Resolving dataset...")
    root = Path(kagglehub.dataset_download(DATASET_KAGGLE))
    df = collect_filepaths(root)
    print(f"  {len(df):,} images across {df['class'].nunique()} classes")

    print(f"\nSampling: PER_CLASS={PER_CLASS}, MIN_PER_CLASS={MIN_PER_CLASS}, seed={SEED}")
    sampled = stratified_sample(df, PER_CLASS, MIN_PER_CLASS, SEED)
    print(f"  Sampled {len(sampled):,} images across {sampled['class'].nunique()} classes")

    full_counts = df["class"].value_counts().rename("full_count")
    sampled_counts = sampled["class"].value_counts().rename("sampled_count")
    manifest = sampled.copy()
    manifest = manifest.merge(full_counts, left_on="class", right_index=True)
    manifest = manifest.merge(sampled_counts, left_on="class", right_index=True)
    manifest.to_csv(OUTPUT_DIR / "sample_manifest.csv", index=False)
    print(f"  Wrote manifest: {OUTPUT_DIR / 'sample_manifest.csv'}")

    print("\nVerification:")
    plot_class_proportions(df, sampled, OUTPUT_DIR / "sample_class_proportions.png")

    print(f"\nWorst-case partition simulation (clients={NUM_CLIENTS}, alpha={DIRICHLET_ALPHA}):")
    counts, classes = dirichlet_partition(sampled, NUM_CLIENTS, DIRICHLET_ALPHA, SEED)
    plot_partition_heatmap(counts, classes, OUTPUT_DIR / "sample_partition_heatmap.png",
                           DIRICHLET_ALPHA)

    print(f"\nOutputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()