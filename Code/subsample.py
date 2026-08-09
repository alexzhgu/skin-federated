"""
Build a subsampled manifest for the federated experiments.

Takes the full split_manifest.csv (with a 'split' column of train/val/test),
keeps the 30 largest classes, and samples down to a target image count while
preserving the original split assignment per image. Sampling is stratified so
each class keeps its share, and the train/val/test ratio is held roughly fixed.

Output: a new manifest with the same columns (filepath, class, split), ready to
feed to the baseline, partition.py, and train_federated.py unchanged.

Usage:
    python subsample.py --manifest runs/20260707_163655/split_manifest.csv \
                        --out sub_manifest.csv --classes 30 --total 20000
"""

import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="sub_manifest.csv")
    ap.add_argument("--classes", type=int, default=30)
    ap.add_argument("--total", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.manifest)
    for col in ("filepath", "class", "split"):
        if col not in df.columns:
            raise SystemExit(f"manifest missing column: {col}")

    # pick the N largest classes (defensible: state the criterion in the thesis)
    keep_classes = df["class"].value_counts().head(args.classes).index.tolist()
    df = df[df["class"].isin(keep_classes)].copy()
    print(f"kept {len(keep_classes)} classes, {len(df):,} images before sampling")

    per_class = args.total // args.classes
    print(f"target ~{per_class} images/class ({per_class * args.classes:,} total)")

    out_parts = []
    for cls in keep_classes:
        sub = df[df["class"] == cls]
        # sample within each split so the train/val/test ratio is preserved
        for split in ("train", "val", "test"):
            rows = sub[sub["split"] == split]
            frac = len(rows) / len(sub) if len(sub) else 0
            n = min(len(rows), round(per_class * frac))
            if n > 0:
                out_parts.append(rows.sample(n=n, random_state=int(rng.integers(1e9))))

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nwrote {len(out):,} images to {args.out}")
    print(out.groupby("split").size().to_string())
    print(f"classes: {out['class'].nunique()}, "
          f"min class size: {out['class'].value_counts().min()}")
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
