"""
Plateau round per configuration from per-round validation logs.

Averages val_acc over seeds by round, then reports the first evaluated round
where the seed mean comes within --tol of its maximum.

Usage:
    python plateau_round.py --fed-dir all_runs --tol 0.005
"""

from pathlib import Path
import argparse
import re

import pandas as pd

TAG_RE = re.compile(
    r"^(?P<algo>fedavg|fedprox)"
    r"_(?P<partition>iid|dirichlet)"
    r"(?:_a(?P<alpha>[\d.]+))?"
    r"(?:_mu(?P<mu>[\d.]+))?"
    r"(?:_c(?P<clients>\d+))?"
    r"_seed(?P<seed>\d+)$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed-dir", default="all_runs")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()

    groups = {}
    for log in sorted(Path(args.fed_dir).glob("*_log.csv")):
        m = TAG_RE.match(log.name[: -len("_log.csv")])
        if not m:
            continue
        d = m.groupdict()
        key = (d["algo"], d["mu"] or "-", "IID" if d["partition"] == "iid"
               else f"a={d['alpha']}", f"c{d['clients'] or 10}")
        groups.setdefault(key, []).append(
            pd.read_csv(log).set_index("round")["val_acc"])

    print(f"plateau = first round where seed-mean val acc >= max - {args.tol}\n")
    for key in sorted(groups):
        mean = pd.concat(groups[key], axis=1).mean(axis=1)
        peak = mean.max()
        plateau = int(mean.index[mean >= peak - args.tol][0])
        algo, mu, cond, k = key
        print(f"  {algo:<8} mu={mu:<6} {cond:<7} {k:<4} n={len(groups[key])}  "
              f"peak {peak:.4f} at round {int(mean.idxmax()):>3}  plateau round {plateau:>3}")


if __name__ == "__main__":
    main()
