"""
Rounds-to-target and communication-to-target per federated run.

Reads every <tag>_log.csv (round, val_acc, val_macro_f1, cumulative_gb) in a
run directory, finds the first evaluated round where validation accuracy
reaches the target, and reports per run and per configuration. Runs that never
reach the target are reported as not reached.

Resolution is limited by --eval-every in federated.py (default 2 rounds).

Outputs (in --out-dir):
    rounds_to_target_runs.csv   one row per run
    rounds_to_target_grid.csv   one row per configuration, over seeds

Usage:
    python rounds_to_target.py --fed-dir all_runs --target 0.85
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

TAG_RE = re.compile(
    r"^(?P<algo>fedavg|fedprox)"
    r"_(?P<partition>iid|dirichlet)"
    r"(?:_a(?P<alpha>[\d.]+))?"
    r"(?:_mu(?P<mu>[\d.]+))?"
    r"(?:_c(?P<clients>\d+))?"
    r"_seed(?P<seed>\d+)$"
)


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        return None
    d = m.groupdict()
    return {
        "algo": d["algo"],
        "partition": d["partition"],
        "alpha": float(d["alpha"]) if d["alpha"] else np.nan,
        "mu": float(d["mu"]) if d["mu"] else np.nan,
        "clients": int(d["clients"]) if d["clients"] else 10,
        "seed": int(d["seed"]),
    }


def first_reach(curve, target):
    hit = curve[curve["val_acc"] >= target]
    if hit.empty:
        return None, None
    return int(hit["round"].iloc[0]), float(hit["cumulative_gb"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed-dir", default="all_runs")
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    fed_dir, out_dir = Path(args.fed_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in sorted(fed_dir.glob("*_log.csv")):
        tag = log.name[: -len("_log.csv")]
        cfg = parse_tag(tag)
        if cfg is None:
            print(f"  skipping unparseable tag: {tag}")
            continue
        curve = pd.read_csv(log)
        rnd, gb = first_reach(curve, args.target)
        rows.append({
            "tag": tag, **cfg,
            "best_val_acc": curve["val_acc"].max(),
            "rounds_run": int(curve["round"].iloc[-1]),
            "round_to_target": rnd,
            "gb_to_target": gb,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no parseable logs in {fed_dir}")

    keys = ["algo", "mu", "partition", "alpha", "clients", "seed"]
    df = df.sort_values(keys)
    df.to_csv(out_dir / "rounds_to_target_runs.csv", index=False)

    grid = (df.groupby(keys[:-1], dropna=False)
              .agg(n=("seed", "size"),
                   n_reached=("round_to_target", lambda s: int(s.notna().sum())),
                   rounds_mean=("round_to_target", "mean"),
                   rounds_std=("round_to_target", "std"),
                   gb_mean=("gb_to_target", "mean"),
                   best_val_mean=("best_val_acc", "mean"))
              .reset_index())
    grid.to_csv(out_dir / "rounds_to_target_grid.csv", index=False)

    print(f"target val acc {args.target}  ({len(df)} runs from {fed_dir})\n")
    print("=== per run ===")
    for _, r in df.iterrows():
        cond = "IID" if r["partition"] == "iid" else f"a={r['alpha']:g}"
        mu = f"mu={r['mu']:g}" if not np.isnan(r["mu"]) else ""
        hit = (f"round {int(r['round_to_target']):>3}  {r['gb_to_target']:7.2f} GB"
               if pd.notna(r["round_to_target"]) else "not reached")
        print(f"  {r['algo']:<8}{mu:<10} {cond:<7} c{r['clients']:<3} s{r['seed']}  "
              f"best {r['best_val_acc']:.4f}  {hit}")

    print("\n=== per configuration (mean over seeds that reached target) ===")
    for _, r in grid.iterrows():
        cond = "IID" if r["partition"] == "iid" else f"a={r['alpha']:g}"
        mu = f"mu={r['mu']:g}" if not np.isnan(r["mu"]) else ""
        if r["n_reached"] > 0:
            std = 0.0 if np.isnan(r["rounds_std"]) else r["rounds_std"]
            hit = (f"{r['rounds_mean']:6.1f} +/- {std:4.1f} rounds  "
                   f"{r['gb_mean']:7.2f} GB")
        else:
            hit = "not reached"
        print(f"  {r['algo']:<8}{mu:<10} {cond:<7} c{r['clients']:<3} "
              f"reached {r['n_reached']}/{r['n']}  {hit}")


if __name__ == "__main__":
    main()
