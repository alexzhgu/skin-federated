"""
Stratified sample of the dataset into a reusable manifest.
Knobs at the top:
    PER_CLASS                target images per class (None = use all)
    MIN_PER_CLASS            drop classes with fewer than this many images
    SEED                     reproducibility
Outputs (in ./eda_outputs/):
    sample_manifest.csv                 filepath, class, full_count, sampled_count
    sample_class_proportions.png        full vs sampled proportions per class
    sample_partition_alpha_<a>.png      one heatmap per Dirichlet alpha
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
PER_CLASS = 500           # change to taste; None = no cap
MIN_PER_CLASS = 50        # drop classes below this in the source data
SEED = 42
# Federated partition simulation (for the worst-case heatmap)
NUM_CLIENTS = 8
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0]   # full experimental sweep
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
def shorten_label(name: str, max_len: int = 22) -> str:
    """Drop common filler suffixes and truncate."""
    for filler in [" And Other Bacterial Infections", " And Other Viral Infections",
                   " And Other Fungal Infections", " And Other Infestations And Bites",
                   " And Other Benign Tumors", " And Other Connective Tissue Diseases",
                   " And Other Hair Diseases", " And Other Nail Disease",
                   " And Other Contact Dermatitis", " And Disorders Of Pigmentation",
                   " And Other Stds Photos", " And Other Malignant Lesions",
                   " And Related Diseases", " Photos"]:
        name = name.replace(filler, "")
    if len(name) > max_len:
        name = name[:max_len - 1] + "..."
    return name
def plot_partition_heatmap(sampled_df: pd.DataFrame, alpha: float, num_clients: int,
                            seed: int, out_path: Path):
    """One heatmap for a single alpha value."""
    counts, classes = dirichlet_partition(sampled_df, num_clients, alpha, seed)
    short_classes = [shorten_label(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.4), 5))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(short_classes)))
    ax.set_xticklabels(short_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"client {k}" for k in range(num_clients)], fontsize=8)
    ax.set_title(f"Dirichlet partition (alpha={alpha})   "
                 f"({(counts == 0).sum()}/{counts.size} empty cells, "
                 f"min client = {counts.sum(axis=1).min():,} imgs)")
    fig.colorbar(im, ax=ax, label="images")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return counts
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
    print(f"\nDirichlet partition simulation across alphas={DIRICHLET_ALPHAS} "
          f"(clients={NUM_CLIENTS}):")
    for alpha in DIRICHLET_ALPHAS:
        out_path = OUTPUT_DIR / f"sample_partition_alpha_{alpha}.png"
        counts = plot_partition_heatmap(sampled, alpha, NUM_CLIENTS, SEED, out_path)
        zero_pct = (counts == 0).sum() / counts.size
        min_client = counts.sum(axis=1).min()
        print(f"  alpha={alpha}: {zero_pct:.1%} empty cells, "
              f"smallest client = {min_client:,} images -> {out_path.name}")
    print(f"\nOutputs in: {OUTPUT_DIR.resolve()}")
if __name__ == "__main__":
    main()