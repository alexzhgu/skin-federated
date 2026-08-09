"""
Perceptual-hash duplicate / leakage check for the skin disease splits.

Reads a split_manifest.csv (columns: filepath, class, split) and looks for
near-duplicate images using an orientation-canonical pHash. Reports:
  - exact pHash collisions (identical or near-identical images)
  - collisions that cross train/val/test  -> split leakage
  - collisions that cross class labels     -> label noise

The orientation-canonical hash (min pHash over the 8 flip/rotation variants)
means flipped or rotated augmentation copies collide with their source, which a
folder browse or a plain pHash both miss.

Dependency: pip install imagehash

Usage:
    python phash_check.py --manifest runs/<ts>/split_manifest.csv
    python phash_check.py --manifest ... --max-hamming 4   # also catch fuzzy near-dupes
    python phash_check.py --manifest ... --fast            # plain pHash, ~8x faster, no flip/rotate invariance
"""

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse

import pandas as pd
from PIL import Image, ImageFile
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True

HASH_SIZE = 8            # 8x8 DCT -> 64-bit hash
CANONICAL = True         # toggled off by --fast via _worker's closure below


def phash_int(img, hash_size=HASH_SIZE):
    return int(str(imagehash.phash(img, hash_size=hash_size)), 16)


def canonical_phash(path, canonical=True):
    """pHash as a 64-bit int. If canonical, minimise over the 8 dihedral
    orientations so flipped/rotated copies map to the same value."""
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    if not canonical:
        return phash_int(img)
    variants = []
    for base in (img, img.transpose(Image.ROTATE_90),
                 img.transpose(Image.ROTATE_180), img.transpose(Image.ROTATE_270)):
        variants.append(base)
        variants.append(base.transpose(Image.FLIP_LEFT_RIGHT))
    return min(phash_int(v) for v in variants)


# module-level flag set in main(), read by the worker (picklable this way)
_USE_CANONICAL = True


def _worker(item):
    idx, path = item
    return idx, canonical_phash(path, _USE_CANONICAL)


def hamming(a, b):
    return bin(a ^ b).count("1")


def banded_pairs(hashes, max_hamming, nbits=64):
    """Yield candidate index pairs within max_hamming using pigeonhole banding:
    two hashes within d bits must share one of d+1 equal-width bands."""
    nbands = max_hamming + 1
    band = nbits // nbands
    mask = (1 << band) - 1
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        for b in range(nbands):
            buckets[(b, (h >> (b * band)) & mask)].append(i)
    seen = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                pair = (members[a], members[c])
                if pair in seen:
                    continue
                seen.add(pair)
                if hamming(hashes[pair[0]], hashes[pair[1]]) <= max_hamming:
                    yield pair


def main():
    global _USE_CANONICAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-hamming", type=int, default=0,
                    help="0 = exact collisions only; >0 also runs fuzzy near-dup search")
    ap.add_argument("--fast", action="store_true",
                    help="plain pHash (no flip/rotation invariance), roughly 8x faster")
    args = ap.parse_args()
    _USE_CANONICAL = not args.fast

    manifest = Path(args.manifest)
    df = pd.read_csv(manifest)
    for col in ("filepath", "class", "split"):
        if col not in df.columns:
            raise SystemExit(f"manifest missing column: {col}")

    mode = "plain" if args.fast else "orientation-canonical"
    print(f"hashing {len(df):,} images ({mode} pHash, {args.workers} workers)...")

    hashes = [None] * len(df)
    failed = 0
    tasks = list(enumerate(df["filepath"].tolist()))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, (idx, h) in enumerate(ex.map(_worker, tasks, chunksize=64), 1):
            hashes[idx] = h
            if h is None:
                failed += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{len(tasks):,}")
    if failed:
        print(f"  {failed:,} images failed to hash (unreadable)")

    df["phash"] = hashes
    df = df[df["phash"].notna()].reset_index(drop=True)

    # ---- exact collisions ----
    groups = {h: list(idx) for h, idx in df.groupby("phash").indices.items()}
    dup_groups = {h: idx for h, idx in groups.items() if len(idx) > 1}

    dup_images = sum(len(v) for v in dup_groups.values())
    cross_split_groups, cross_class_groups = [], 0
    for h, idx in dup_groups.items():
        sub = df.iloc[idx]
        if sub["split"].nunique() > 1:
            cross_split_groups.append(idx)
        if sub["class"].nunique() > 1:
            cross_class_groups += 1

    print("\n=== exact pHash collisions ===")
    print(f"  duplicate groups:      {len(dup_groups):,}")
    print(f"  images in a duplicate: {dup_images:,} ({dup_images / max(1, len(df)):.1%} of dataset)")
    print(f"  groups crossing split: {len(cross_split_groups):,}   <-- LEAKAGE")
    print(f"  groups crossing class: {cross_class_groups:,}   <-- label noise")

    # write the leaking groups out for inspection
    if cross_split_groups:
        rows = []
        for gid, idx in enumerate(cross_split_groups):
            for _, r in df.iloc[idx].iterrows():
                rows.append({"group": gid, "filepath": r["filepath"],
                             "class": r["class"], "split": r["split"]})
        out = manifest.parent / "phash_leak_groups.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  wrote {len(cross_split_groups):,} cross-split groups -> {out}")
        print("\n  first few leaking groups:")
        for idx in cross_split_groups[:5]:
            print(df.iloc[idx][["class", "split", "filepath"]].to_string(index=False))
            print()

    # ---- optional fuzzy near-dupes ----
    if args.max_hamming > 0:
        arr = df["phash"].tolist()
        pairs = list(banded_pairs(arr, args.max_hamming, nbits=HASH_SIZE * HASH_SIZE))
        near_cross = sum(1 for a, b in pairs if df.iloc[a]["split"] != df.iloc[b]["split"])
        print(f"\n=== fuzzy near-dupes (hamming <= {args.max_hamming}) ===")
        print(f"  candidate pairs:      {len(pairs):,}")
        print(f"  pairs crossing split: {near_cross:,}   <-- LEAKAGE")


if __name__ == "__main__":
    main()
