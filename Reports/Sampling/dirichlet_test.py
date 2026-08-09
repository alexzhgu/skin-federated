"""
Standalone Dirichlet partition visualisation.

Loads ./eda_outputs/sample_manifest.csv and produces a stacked heatmap
of (client, class) image counts across multiple alpha values, with one
shared x-axis at the bottom.

Outputs (in ./eda_outputs/):
    sample_partition_heatmap.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== Config =====
MANIFEST = Path("./eda_outputs/sample_manifest.csv")
OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0]
SEED = 42


def dirichlet_partition(df: pd.DataFrame, num_clients: int, alpha: float, seed: int):
    """Hsu et al.-style partitioning."""
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


def plot_partition_stacked(sampled_df: pd.DataFrame, alphas: list, num_clients: int,
                            seed: int, out_path: Path):
    """Stacked heatmaps sharing one x-axis (class labels only at the bottom)."""
    classes = sorted(sampled_df["class"].unique())
    short_classes = [shorten_label(c) for c in classes]

    fig, axes = plt.subplots(len(alphas), 1, sharex=True,
                              figsize=(max(10, len(classes) * 0.4),
                                       2.5 * len(alphas)))
    if len(alphas) == 1:
        axes = [axes]

    all_counts = []
    for ax, alpha in zip(axes, alphas):
        counts, _ = dirichlet_partition(sampled_df, num_clients, alpha, seed)
        all_counts.append((alpha, counts))
        im = ax.imshow(counts, aspect="auto", cmap="viridis")
        ax.set_yticks(range(num_clients))
        ax.set_yticklabels([f"c{k}" for k in range(num_clients)], fontsize=7)
        ax.set_title(f"alpha = {alpha}   "
                     f"({(counts == 0).sum()}/{counts.size} empty cells, "
                     f"min client = {counts.sum(axis=1).min():,} imgs)",
                     fontsize=10)
        fig.colorbar(im, ax=ax, label="images")

    axes[-1].set_xticks(range(len(short_classes)))
    axes[-1].set_xticklabels(short_classes, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return all_counts


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}. Run sample.py first.")

    print(f"Loading manifest: {MANIFEST}")
    sampled = pd.read_csv(MANIFEST)[["filepath", "class"]]
    print(f"  {len(sampled):,} images across {sampled['class'].nunique()} classes")

    print(f"\nDirichlet partition simulation across alphas={DIRICHLET_ALPHAS} "
          f"(clients={NUM_CLIENTS}):")
    out_path = OUTPUT_DIR / "sample_partition_heatmap.png"
    all_counts = plot_partition_stacked(sampled, DIRICHLET_ALPHAS, NUM_CLIENTS, SEED, out_path)
    for alpha, counts in all_counts:
        zero_pct = (counts == 0).sum() / counts.size
        min_client = counts.sum(axis=1).min()
        print(f"  alpha={alpha}: {zero_pct:.1%} empty cells, "
              f"smallest client = {min_client:,} images")
    print(f"  Stacked heatmap -> {out_path}")


if __name__ == "__main__":
    main()