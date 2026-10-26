"""
Aggregate federated run outputs into a results table and figures.

Reads *_report.txt and *_log.csv from one or more run folders, parses the run
configuration from the filename tag, and writes tables and figures. When the
same configuration and seed appear in more than one folder, the later folder wins.

Outputs (in ./results/):
    summary.csv              one row per run
    grid.csv                 mean/std per (algo, partition, mu, clients) cell
    paired.csv               per-seed FedAvg vs FedProx differences
    fig_accuracy_vs_alpha.png
    fig_curves.png           val curves per round, one panel per partition (K = 10)
    fig_accuracy_vs_comms.png  val curves per GB, one panel per partition (K = 10)
    fig_accuracy_vs_nodes.png  test acc vs client count at alpha = 0.1
    panels/curves_<cond>_c<K>.png, panels/comms_<cond>_c<K>.png
                             one figure per partition and client count

Usage:
    python aggregate.py --fed-dir fed_runs post_analysis \
        --baseline runs/<seed42_dir> runs/<seed43_dir> runs/<seed44_dir>
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fedavg_dirichlet_a0.1_c10_seed42  /  fedprox_iid_mu0.001_c50_seed42
# _c is optional: runs predating the client-count tag default to 10.
# Tags with extra parts (e.g. _scratch) do not match and are skipped.
TAG_RE = re.compile(
    r"^(?P<algo>fedavg|fedprox)"
    r"_(?P<partition>iid|dirichlet)"
    r"(?:_a(?P<alpha>[\d.]+))?"
    r"(?:_mu(?P<mu>[\d.]+))?"
    r"(?:_c(?P<clients>\d+))?"
    r"_seed(?P<seed>\d+)$"
)

IID_ALPHA = 10.0
COLOURS = {None: "tab:blue", 0.001: "tab:orange", 0.01: "tab:green",
           0.1: "tab:red", 1.0: "tab:purple"}
DPI = 200


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


def method_label(algo, mu):
    return "FedAvg" if algo == "fedavg" else rf"FedProx ($\mu$ = {mu:g})"


def colour(algo, mu):
    return COLOURS[None] if algo == "fedavg" else COLOURS.get(float(mu), "tab:gray")


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


def config_key(cfg):
    return (cfg["algo"], cfg["partition"],
            None if np.isnan(cfg["alpha"]) else cfg["alpha"],
            None if np.isnan(cfg["mu"]) else cfg["mu"],
            cfg["clients"], cfg["seed"])


def collect(fed_dirs):
    found = {}
    for fed_dir in fed_dirs:
        reports = sorted(fed_dir.glob("*_report.txt"))
        print(f"  {fed_dir}: {len(reports)} reports")
        for report in reports:
            tag = report.name[: -len("_report.txt")]
            cfg = parse_tag(tag)
            if cfg is None:
                print(f"    skipping unparseable tag: {tag}")
                continue
            rec = {"tag": tag, "source": str(fed_dir), **cfg, **read_report(report)}
            curve = None
            log = fed_dir / f"{tag}_log.csv"
            if log.exists():
                curve = pd.read_csv(log)
                rec["final_val_acc"] = curve["val_acc"].iloc[-1]
                rec["best_val_acc"] = curve["val_acc"].max()
                rec["best_round"] = int(curve.loc[curve["val_acc"].idxmax(), "round"])
                rec["rounds_run"] = int(curve["round"].iloc[-1])
            key = config_key(cfg)
            if key in found:
                print(f"    {tag}: replaces {found[key][0]['tag']} from {found[key][0]['source']}")
            found[key] = (rec, curve)

    rows = [rec for rec, _ in found.values()]
    curves = {rec["tag"]: curve for rec, curve in found.values() if curve is not None}
    df = pd.DataFrame(rows)
    if len(df):
        df["alpha_plot"] = df["alpha"].fillna(IID_ALPHA)
    return df, curves


def build_grid(df):
    g = (df.groupby(["algo", "mu", "partition", "alpha_plot", "clients"], dropna=False)
           .agg(n=("test_acc", "size"),
                seeds=("seed", lambda s: " ".join(str(x) for x in sorted(s))),
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


def fig_accuracy_vs_alpha(grid, baseline, out_path, clients=10):
    grid = grid[grid["clients"] == clients]
    alphas = sorted(grid["alpha_plot"].unique())
    pos = {a: i for i, a in enumerate(alphas)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if baseline is not None:
        m, s, n = baseline
        ax.axhline(m, ls="--", c="k", lw=1,
                   label=rf"Centralised ({m:.3f} $\pm$ {s:.3f})")
        if n > 1:
            ax.axhspan(m - s, m + s, color="k", alpha=0.08)
    offset = 0.0
    for (algo, mu), sub in grid.groupby(["algo", "mu"], dropna=False):
        sub = sub.sort_values("alpha_plot")
        yerr = sub["acc_std"].fillna(0)
        style = dict(marker="o", capsize=3, color=colour(algo, mu),
                     label=method_label(algo, mu))
        if len(sub) > 1:
            ax.errorbar(sub["alpha_plot"].map(pos), sub["acc_mean"], yerr=yerr, **style)
        else:
            offset += 0.12
            ax.errorbar(sub["alpha_plot"].map(pos) + offset, sub["acc_mean"],
                        yerr=yerr, ls="none", **style)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(["IID" if a == IID_ALPHA else f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Dirichlet concentration $\alpha$")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def fig_accuracy_vs_nodes(grid, out_path, alpha=0.1):
    sub = grid[grid["alpha_plot"] == alpha]
    if sub["clients"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (algo, mu), g in sub.groupby(["algo", "mu"], dropna=False):
        if g["clients"].nunique() < 2:
            continue
        g = g.sort_values("clients")
        ax.errorbar(g["clients"], g["acc_mean"], yerr=g["acc_std"].fillna(0),
                    marker="o", capsize=3, color=colour(algo, mu),
                    label=method_label(algo, mu))
    ax.set_xlabel("Number of clients $K$")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return True


def cond_title(alpha, clients):
    cond = "IID" if alpha == IID_ALPHA else rf"$\alpha$ = {alpha:g}"
    return f"{cond}, K = {int(clients)}"


def cond_file(alpha, clients):
    cond = "iid" if alpha == IID_ALPHA else f"a{alpha:g}"
    return f"{cond}_c{int(clients)}"


def seed_mean_curve(grp, curves):
    """Mean and std of val_acc over seeds, indexed by round, with cumulative GB."""
    series = [curves[t].set_index("round") for t in grp["tag"] if t in curves]
    if not series:
        return None
    acc = pd.concat([s["val_acc"] for s in series], axis=1)
    gb = series[0]["cumulative_gb"].reindex(acc.index)
    return acc.mean(axis=1), acc.std(axis=1).fillna(0), gb, len(series)


def curve_ylim(df, curves):
    lo, hi = 1.0, 0.0
    for _, grp in df.groupby(["algo", "mu", "alpha_plot", "clients"], dropna=False):
        res = seed_mean_curve(grp, curves)
        if res is None:
            continue
        mean, std, _, _ = res
        lo, hi = min(lo, (mean - std).min()), max(hi, (mean + std).max())
    return max(0.0, lo - 0.02), min(1.0, hi + 0.02)


def draw_panel(ax, sub, curves, xaxis):
    for (algo, mu), grp in sub.groupby(["algo", "mu"], dropna=False):
        res = seed_mean_curve(grp, curves)
        if res is None:
            continue
        mean, std, gb, n = res
        x = mean.index if xaxis == "round" else gb.values
        c = colour(algo, mu)
        ax.plot(x, mean.values, color=c, lw=1.4,
                label=f"{method_label(algo, mu)}, {n} seed{'s' if n > 1 else ''}")
        if n > 1:
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("Communication round" if xaxis == "round"
                  else "Estimated cumulative payload (GB)")
    ax.grid(alpha=0.3)


def fig_combined(df, curves, out_path, xaxis, ylim, clients=10):
    """One panel per partition at a fixed client count."""
    df = df[df["clients"] == clients]
    partitions = sorted(df["alpha_plot"].unique())
    fig, axes = plt.subplots(1, len(partitions), figsize=(4 * len(partitions), 3.8),
                             sharey=True, squeeze=False)
    for ax, a in zip(axes[0], partitions):
        draw_panel(ax, df[df["alpha_plot"] == a], curves, xaxis)
        ax.set_title(cond_title(a, clients), fontsize=10)
        ax.set_ylim(*ylim)
    axes[0][0].set_ylabel("Validation accuracy")
    axes[0][-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def fig_panels(df, curves, out_dir, ylim):
    """One figure per (partition, client count), for rounds and for GB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (a, k), sub in df.groupby(["alpha_plot", "clients"]):
        for xaxis, prefix in (("round", "curves"), ("gb", "comms")):
            fig, ax = plt.subplots(figsize=(6.5, 4))
            draw_panel(ax, sub, curves, xaxis)
            ax.set_ylabel("Validation accuracy")
            ax.set_ylim(*ylim)
            ax.legend(fontsize=8, loc="lower right")
            plt.tight_layout()
            path = out_dir / f"{prefix}_{cond_file(a, k)}.png"
            plt.savefig(path, dpi=DPI)
            plt.close()
            written.append(path.name)
    return written


def load_baseline(dirs):
    accs = []
    for b in dirs:
        bpath = Path(b) / "test_results.json"
        if bpath.exists():
            acc = json.load(open(bpath))["test_acc"]
            accs.append(acc)
            print(f"  baseline {b}: {acc:.4f}")
        else:
            print(f"  baseline not found at {bpath}")
    if not accs:
        return None
    sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    return float(np.mean(accs)), sd, len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed-dir", nargs="+", default=["fed_runs"],
                    help="one or more run folders; later folders win on duplicates")
    ap.add_argument("--baseline", nargs="+", default=None,
                    help="baseline run dirs, each containing test_results.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reading runs")
    df, curves = collect([Path(d) for d in args.fed_dir])
    if df.empty:
        raise SystemExit(f"no parseable runs found in {args.fed_dir}")
    print(f"parsed {len(df)} runs, {len(curves)} with curves")

    baseline = None
    if args.baseline:
        print("\nreading baseline")
        baseline = load_baseline(args.baseline)
        if baseline is not None:
            m, s, n = baseline
            print(f"  centralised mean {m:.4f} +/- {s:.4f} (n={n})")

    df.sort_values(["algo", "mu", "alpha_plot", "clients", "seed"]).to_csv(
        out_dir / "summary.csv", index=False)

    grid = build_grid(df)
    grid.to_csv(out_dir / "grid.csv", index=False)
    print("\n=== grid (mean +/- std over seeds) ===")
    for _, r in grid.iterrows():
        cond = "IID" if r["alpha_plot"] == IID_ALPHA else f"a={r['alpha_plot']:g}"
        mu = f" mu={r['mu']:g}" if not np.isnan(r["mu"]) else ""
        std = 0.0 if np.isnan(r["acc_std"]) else r["acc_std"]
        flag = "  <-- single seed" if r["n"] < 2 else ""
        print(f"  {r['algo']:<8}{mu:<12} {cond:<8} c{int(r['clients']):<4} "
              f"acc {r['acc_mean']:.4f} +/- {std:.4f}  seeds [{r['seeds']}]{flag}")

    paired = build_paired(df)
    if not paired.empty:
        paired.to_csv(out_dir / "paired.csv", index=False)
        print("\n=== fedprox - fedavg, paired by seed (points) ===")
        for (part, a, mu, nc), sub in paired.groupby(
                ["partition", "alpha_plot", "mu", "clients"]):
            cond = "IID" if a == IID_ALPHA else f"a={a:g}"
            d = sub["acc_diff"] * 100
            sd = d.std() if len(d) > 1 else 0.0
            print(f"  {cond:<8} c{int(nc):<4} mu={mu:<8g} mean {d.mean():+.2f} "
                  f"+/- {sd:.2f}, ahead on {(d > 0).sum()} of {len(d)}")

    fig_accuracy_vs_alpha(grid, baseline, out_dir / "fig_accuracy_vs_alpha.png")
    ylim = curve_ylim(df, curves)
    fig_combined(df, curves, out_dir / "fig_curves.png", "round", ylim)
    fig_combined(df, curves, out_dir / "fig_accuracy_vs_comms.png", "gb", ylim)
    panels = fig_panels(df, curves, out_dir / "panels", ylim)
    print(f"\nwrote {len(panels)} single-panel figures to {out_dir / 'panels'}")
    if fig_accuracy_vs_nodes(grid, out_dir / "fig_accuracy_vs_nodes.png"):
        print("wrote node-sweep figure")
    print(f"wrote tables and figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()