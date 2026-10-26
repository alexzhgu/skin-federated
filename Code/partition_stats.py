"""Partition statistics for the federated experiments.

The partition and client-sampling code is copied from federated.py so the
results match the training runs exactly without needing torch. Only the
manifest's labels are read; no images are loaded.

Outputs:
    partition_stats.csv     one row per (partition, K, seed)
    partition_heatmaps.png  client-by-class counts, K = 10, seed 42
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "sub_manifest.csv"
FRAC = 0.5
ROUNDS = 100


# ---- copied from federated.py -------------------------------------------
def load_manifest(path, path_col="filepath", label_col="class"):
    df = pd.read_csv(path)
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_to_idx[l] for l in df[label_col]], dtype=np.int64)
    splits = df["split"].to_numpy()
    train_idx = np.flatnonzero(splits == "train")
    return labels, classes, train_idx


def partition_iid(labels, num_clients, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [list(shard) for shard in np.array_split(idx, num_clients)]


def partition_dirichlet(labels, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    client_idx = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for client, shard in enumerate(np.split(c_idx, cuts)):
            client_idx[client].extend(shard.tolist())
    for k in range(num_clients):
        rng.shuffle(client_idx[k])
    return client_idx
# --------------------------------------------------------------------------


labels, classes, train_idx = load_manifest(MANIFEST)
y = labels[train_idx]
C = len(classes)


def client_counts(partition, K, alpha, seed):
    parts = (partition_iid(y, K, seed) if partition == "iid"
             else partition_dirichlet(y, K, alpha, seed))
    parts = [np.array(p, dtype=int) for p in parts if len(p) >= 2]  # same filter as training
    return np.stack([np.bincount(y[p], minlength=C) for p in parts])


def round_coverage(M, seed):
    """Replays the training script's client sampling and counts how many of the
    30 classes the selected clients hold between them in each round."""
    rng = np.random.default_rng(seed)
    K = M.shape[0]
    m = max(1, int(FRAC * K))
    cov = [int((M[rng.choice(K, size=m, replace=False)].sum(0) > 0).sum())
           for _ in range(ROUNDS)]
    return np.array(cov)


configs = [("iid", None, 10), ("dirichlet", 1.0, 10), ("dirichlet", 0.5, 10),
           ("dirichlet", 0.1, 10), ("dirichlet", 0.1, 5), ("dirichlet", 0.1, 50)]
rows = []
for partition, alpha, K in configs:
    for seed in (42, 43, 44, 45, 46):
        M = client_counts(partition, K, alpha, seed)
        sizes = M.sum(1)
        present = (M > 0).sum(1)
        cov = round_coverage(M, seed)
        rows.append({
            "partition": "IID" if partition == "iid" else f"alpha={alpha}",
            "K": K, "seed": seed, "clients": M.shape[0],
            "size_min": sizes.min(), "size_median": int(np.median(sizes)), "size_max": sizes.max(),
            "classes_min": present.min(), "classes_median": float(np.median(present)),
            "classes_max": present.max(),
            "empty_cells": int((M == 0).sum()), "total_cells": M.size,
            "round_cov_mean": round(cov.mean(), 1), "round_cov_min": cov.min(),
            "rounds_all_30": int((cov == C).sum()),
        })
stats = pd.DataFrame(rows)
stats.to_csv("partition_stats.csv", index=False)
print(stats.to_string(index=False))

fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
vmax = max(client_counts(p, 10, a, 42).max() for p, a, _ in configs[:4])
for ax, (partition, alpha, K) in zip(axes, configs[:4]):
    M = client_counts(partition, K, alpha, 42)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title("IID" if partition == "iid" else f"$\\alpha$ = {alpha}", fontsize=11)
    ax.set_ylabel("Client")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(range(1, M.shape[0] + 1), fontsize=7)
    ax.set_xticks(range(C))
    ax.set_xticklabels(range(1, C + 1), fontsize=7)
axes[-1].set_xlabel("Class index")
fig.colorbar(im, ax=axes, shrink=0.6, label="Training images")
fig.savefig("partition_heatmaps.png", dpi=200)