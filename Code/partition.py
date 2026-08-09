"""
Client partitioning for federated experiments.

Splits a training dataframe across K simulated clients under either an IID
shuffle or a Dirichlet label-skew partition (Hsu et al., 2019).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PARTITIONS = {
    "iid":       None,
    "alpha_1.0": 1.0,
    "alpha_0.5": 0.5,
    "alpha_0.1": 0.1,
}


def iid_partition(df: pd.DataFrame, num_clients: int, seed: int) -> list[np.ndarray]:
    """Uniform random split"""
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(df))
    rng.shuffle(idxs)
    return [np.sort(a) for a in np.array_split(idxs, num_clients)]


def dirichlet_partition(df: pd.DataFrame, num_clients: int, alpha: float,
                        seed: int, label_col: str = "class") -> list[np.ndarray]:
    """Per-class Dirichlet split"""
    rng = np.random.default_rng(seed)
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in sorted(df[label_col].unique()):
        idxs = np.flatnonzero(df[label_col].to_numpy() == cls)
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idxs, cuts)):
            client_idx[k].extend(chunk.tolist())

    return [np.sort(np.array(c, dtype=int)) for c in client_idx]


def partition(df: pd.DataFrame, num_clients: int, alpha: float | None,
              seed: int, label_col: str = "class") -> list[np.ndarray]:
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError(
            "df must have a 0..n-1 RangeIndex; call reset_index(drop=True) first. "
            "Returned indices are positional and are passed to torch Subset."
        )
    if alpha is None:
        clients = iid_partition(df, num_clients, seed)
    else:
        clients = dirichlet_partition(df, num_clients, alpha, seed, label_col)

    total = sum(len(c) for c in clients)
    assert total == len(df), f"partition lost images: {total} != {len(df)}"
    assert len(np.unique(np.concatenate(clients))) == len(df), "duplicate assignment"
    return clients


def summarise(df: pd.DataFrame, clients: list[np.ndarray],
              label_col: str = "class") -> dict:
    classes = sorted(df[label_col].unique())
    counts = np.zeros((len(clients), len(classes)), dtype=int)
    labels = df[label_col].to_numpy()
    cls_to_col = {c: j for j, c in enumerate(classes)}
    for k, idx in enumerate(clients):
        for c, n in zip(*np.unique(labels[idx], return_counts=True)):
            counts[k, cls_to_col[c]] = n

    sizes = counts.sum(axis=1)
    return {
        "counts": counts,
        "classes": classes,
        "client_sizes": sizes.tolist(),
        "min_client": int(sizes.min()),
        "max_client": int(sizes.max()),
        "size_ratio": float(sizes.max() / max(1, sizes.min())),
        "empty_cells": int((counts == 0).sum()),
        "total_cells": int(counts.size),
        "classes_per_client": (counts > 0).sum(axis=1).tolist(),
    }


if __name__ == "__main__":
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else "split_manifest.csv"
    train_df = (pd.read_csv(manifest)
                  .query("split == 'train'")
                  .reset_index(drop=True))
    print(f"{len(train_df):,} train images, {train_df['class'].nunique()} classes\n")

    for name, alpha in PARTITIONS.items():
        clients = partition(train_df, num_clients=10, alpha=alpha, seed=42)
        s = summarise(train_df, clients)
        print(f"{name:>10}: "
              f"clients {s['min_client']:,}-{s['max_client']:,} imgs "
              f"(ratio {s['size_ratio']:.1f}x), "
              f"{s['empty_cells']}/{s['total_cells']} empty cells, "
              f"classes/client {min(s['classes_per_client'])}-{max(s['classes_per_client'])}")
