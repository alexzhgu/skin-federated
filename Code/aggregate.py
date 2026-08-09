"""
Aggregate federated run outputs into a results table and figures.

Reads fed_runs/*_report.txt and fed_runs/*_log.csv, parses the run
configuration from the filename tag, and writes tables and figures.

Outputs (in ./results/):
    summary.csv              one row per run
    grid.csv                 mean/std per (algo, partition, mu) cell
    paired.csv               per-seed FedAvg vs FedProx differences
    fig_accuracy_vs_alpha.png
    fig_curves.png
    fig_accuracy_vs_comms.png

Usage:
    python aggregate.py
    python aggregate.py --fed-dir fed_runs --baseline runs/20260712_101500
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fedavg_dirichlet_a0.1_c10_seed42  /  fedprox_iid_mu0.001_c50_seed42
# _c is optional: runs predating the client-count tag default to 10.
TAG_RE = re.compile(
    r"^(?P<algo>fedavg|fedprox)"
    r"_(?P<partition>iid|dirichlet)"
    r"(?:_a(?P<alpha>[\d.]+))?"
    r"(?:_mu(?P<mu>[\d.]+))?"
    r"(?:_c(?P<clients>\d+))?"
    r"_seed(?P<seed>\d+)$"
)

IID_ALPHA = 10.0


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


def read_report(path):
    """Parse the key/value lines at the head of a *_report.txt."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, val = parts
        key = re.sub(r"\(.*\)", "", key)
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = np.nan  # e.g. round_to_target None
    return out


def collect(fed_dir: Path):
    rows, curves = [], {}
    for report in sorted(fed_dir.glob("*_report.txt")):
        tag = report.name[: -len("_report.txt")]
        cfg = parse_tag(tag)
        if cfg is None:
            print(f"  skipping unparseable tag: {tag}")
            continue
        rec = {"tag": tag, **cfg, **read_report(report)}

        log = fed_dir / f"{tag}_log.csv"
        if log.exists():
            curve = pd.read_csv(log)
            curves[tag] = curve
            rec["final_val_acc"] = curve["val_acc"].iloc[-1]
            rec["best_val_acc"] = curve["val_acc"].max()
            rec["best_round"] = int(curve.loc[curve["val_acc"].idxmax(), "round"])
            rec["rounds_run"] = int(curve["round"].iloc[-1])
        rows.append(rec)

    df = pd.DataFrame(rows)
    if len(df):
        df["alpha_plot"] = df["alpha"].fillna(IID_ALPHA)
        df["label"] = df.apply(
            lambda r: f"{r['algo']}"
            + (f" mu={r['mu']:g}" if not np.isnan(r["mu"]) else ""), axis=1
        )
    return df, curves


def build_grid(df):
    g = (df.groupby(["algo", "mu", "partition", "alpha_plot", "clients"], dropna=False)
           .agg(n=("test_acc", "size"),
                acc_mean=("test_acc", "mean"),
                acc_std=("test_acc", "std"),
                f1_mean=("test_macro_f1", "mean"),
                f1_std=("test_macro_f1", "std"),
                comms_mean=("total_comms_gb", "mean"))
           .reset_index()
           .sort_values(["algo", "mu", "clients", "alpha_plot"]))
    return g


def build_paired(df):
    """FedAvg vs FedProx at matched seed and partition."""
    avg = df[df["algo"] == "fedavg"]
    prox = df[df["algo"] == "fedprox"]
    if avg.empty or prox.empty:
        return pd.DataFrame()
    merged = prox.merge(
        avg[["partition", "alpha_plot", "clients", "seed", "test_acc", "test_macro_f1"]],
        on=["partition", "alpha_plot", "clients", "seed"],
        suffixes=("_prox", "_avg"),
    )
    merged["acc_diff"] = merged["test_acc_prox"] - merged["test_acc_avg"]
    merged["f1_diff"] = merged["test_macro_f1_prox"] - merged["test_macro_f1_avg"]
    return merged[["partition", "alpha_plot", "clients", "mu", "seed",
                   "test_acc_avg", "test_acc_prox", "acc_diff", "f1_diff"]]


def fig_accuracy_vs_alpha(grid, baseline_acc, out_path, clients=10):
    grid = grid[grid["clients"] == clients]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (algo, mu), sub in grid.groupby(["algo", "mu"], dropna=False):
        sub = sub.sort_values("alpha_plot")
        label = algo + (f" (mu={mu:g})" if not np.isnan(mu) else "")
        ax.errorbar(sub["alpha_plot"], sub["acc_mean"],
                    yerr=sub["acc_std"].fillna(0),
                    marker="o", capsize=3, label=label)
    if baseline_acc is not None:
        ax.axhline(baseline_acc, ls="--", c="k", lw=1,
                   label=f"centralised ({baseline_acc:.3f})")
    ax.set_xscale("log")
    ticks = sorted(grid["alpha_plot"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([("IID" if t == IID_ALPHA else f"{t:g}") for t in ticks])
    ax.set_xlabel("Dirichlet concentration parameter alpha")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs data heterogeneity (mean +/- std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fig_accuracy_vs_nodes(grid, out_path, alpha=0.1):
    sub = grid[(grid["alpha_plot"] == alpha) & (grid["clients"].notna())]
    if sub["clients"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (algo, mu), g in sub.groupby(["algo", "mu"], dropna=False):
        g = g.sort_values("clients")
        label = algo + (f" (mu={mu:g})" if not np.isnan(mu) else "")
        ax.errorbar(g["clients"], g["acc_mean"], yerr=g["acc_std"].fillna(0),
                    marker="o", capsize=3, label=label)
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy vs client count (alpha = {alpha:g}, mean +/- std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def fig_curves(df, curves, out_path):
    partitions = sorted(df["alpha_plot"].unique())
    fig, axes = plt.subplots(1, len(partitions), figsize=(4 * len(partitions), 3.8),
                             sharey=True, squeeze=False)
    for ax, a in zip(axes[0], partitions):
        sub = df[df["alpha_plot"] == a]
        for label, grp in sub.groupby("label"):
            aligned = []
            for tag in grp["tag"]:
                if tag in curves:
                    c = curves[tag]
                    aligned.append(c.set_index("round")["val_acc"])
            if not aligned:
                continue
            stacked = pd.concat(aligned, axis=1)
            mean, std = stacked.mean(axis=1), stacked.std(axis=1).fillna(0)
            ax.plot(mean.index, mean.values, label=label, lw=1.4)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
        ax.set_title("IID" if a == IID_ALPHA else f"alpha = {a:g}", fontsize=10)
        ax.set_xlabel("Communication round")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Validation accuracy")
    axes[0][-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fig_accuracy_vs_comms(df, curves, out_path):
    partitions = sorted(df["alpha_plot"].unique())
    fig, axes = plt.subplots(1, len(partitions), figsize=(4 * len(partitions), 3.8),
                             sharey=True, squeeze=False)
    for ax, a in zip(axes[0], partitions):
        sub = df[df["alpha_plot"] == a]
        for label, grp in sub.groupby("label"):
            tag = grp["tag"].iloc[0]  # one representative seed
            if tag not in curves:
                continue
            c = curves[tag]
            ax.plot(c["cumulative_gb"], c["val_acc"], label=label, lw=1.4)
        ax.set_title("IID" if a == IID_ALPHA else f"alpha = {a:g}", fontsize=10)
        ax.set_xlabel("Cumulative communication (GB)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Validation accuracy")
    axes[0][-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed-dir", default="fed_runs")
    ap.add_argument("--baseline", default=None,
                    help="baseline run dir containing test_results.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    fed_dir = Path(args.fed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, curves = collect(fed_dir)
    if df.empty:
        raise SystemExit(f"no parseable runs found in {fed_dir}")
    print(f"parsed {len(df)} runs, {len(curves)} with curves")

    baseline_acc = None
    if args.baseline:
        import json
        bpath = Path(args.baseline) / "test_results.json"
        if bpath.exists():
            baseline_acc = json.load(open(bpath))["test_acc"]
            print(f"baseline test acc: {baseline_acc:.4f}")
        else:
            print(f"  baseline not found at {bpath}")

    df.sort_values(["algo", "mu", "alpha_plot", "seed"]).to_csv(
        out_dir / "summary.csv", index=False)

    grid = build_grid(df)
    grid.to_csv(out_dir / "grid.csv", index=False)
    print("\n=== grid (mean +/- std over seeds) ===")
    for _, r in grid.iterrows():
        cond = "IID" if r["alpha_plot"] == IID_ALPHA else f"a={r['alpha_plot']:g}"
        mu = f" mu={r['mu']:g}" if not np.isnan(r["mu"]) else ""
        nc = f"c{int(r['clients'])}" if not pd.isna(r["clients"]) else "c?"
        std = 0.0 if np.isnan(r["acc_std"]) else r["acc_std"]
        flag = "  <-- single seed" if r["n"] < 2 else ""
        print(f"  {r['algo']:<8}{mu:<12} {cond:<8} {nc:<5} "
              f"acc {r['acc_mean']:.4f} +/- {std:.4f}  (n={int(r['n'])}){flag}")

    paired = build_paired(df)
    if not paired.empty:
        paired.to_csv(out_dir / "paired.csv", index=False)
        print("\n=== fedprox - fedavg, paired by seed ===")
        for (part, a, mu, nc), sub in paired.groupby(
                ["partition", "alpha_plot", "mu", "clients"]):
            cond = "IID" if a == IID_ALPHA else f"a={a:g}"
            d = sub["acc_diff"]
            print(f"  {cond:<8} c{int(nc):<4} mu={mu:<8g} mean diff {d.mean():+.4f} "
                  f"(std {d.std() if len(d) > 1 else 0:.4f}, n={len(d)})")

    fig_accuracy_vs_alpha(grid, baseline_acc, out_dir / "fig_accuracy_vs_alpha.png")
    fig_curves(df, curves, out_dir / "fig_curves.png")
    fig_accuracy_vs_comms(df, curves, out_dir / "fig_accuracy_vs_comms.png")
    if fig_accuracy_vs_nodes(grid, out_dir / "fig_accuracy_vs_nodes.png"):
        print("wrote node-sweep figure")
    print(f"\nwrote tables and figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()