"""Merge the post_analysis runs with the August summary.csv and print every
number that changes in Chapters 5 and 6, plus a client drift figure.

Usage:
    python update_numbers.py summary.csv post_analysis

Outputs:
    summary_merged.csv   August runs plus new runs, one row per run
    drift_curves.png     mean client drift per evaluation round
"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:
    stats = None

SUMMARY = sys.argv[1] if len(sys.argv) > 1 else "summary.csv"
NEW_DIR = sys.argv[2] if len(sys.argv) > 2 else "post_analysis"
TAG_RE = re.compile(r"(fedavg|fedprox)_(iid|dirichlet)(?:_a([\d.]+))?(?:_mu([\d.]+))?"
                    r"(?:_scratch)?_c(\d+)_seed(\d+)$")


def read_new_run(report_path):
    tag = os.path.basename(report_path).replace("_report.txt", "")
    m = TAG_RE.match(tag)
    if m is None:
        raise ValueError(f"unrecognised tag: {tag}")
    algo, partition, alpha, mu, clients, seed = m.groups()
    rep = {}
    with open(report_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("test_acc", "test_macro_f1", "total_comms_gb"):
                rep[parts[0]] = float(parts[1])
            if line.startswith("round_to_target"):
                val = line.split()[-1]
                rep["round_to_target"] = np.nan if val == "None" else float(val)
    log = pd.read_csv(report_path.replace("_report.txt", "_log.csv"))
    best = log.loc[log["val_acc"].idxmax()]
    return {
        "tag": tag, "algo": algo, "partition": partition,
        "alpha": float(alpha) if alpha else np.nan,
        "mu": float(mu) if mu else np.nan,
        "clients": int(clients), "seed": int(seed),
        "test_acc": rep["test_acc"], "test_macro_f1": rep["test_macro_f1"],
        "round_to_target": rep.get("round_to_target", np.nan),
        "total_comms_gb": rep["total_comms_gb"],
        "final_val_acc": log["val_acc"].iloc[-1],
        "best_val_acc": best["val_acc"], "best_round": int(best["round"]),
        "rounds_run": int(log["round"].iloc[-1]),
    }, log


old = pd.read_csv(SUMMARY)
new_rows, logs = [], {}
for path in sorted(glob.glob(os.path.join(NEW_DIR, "*_report.txt"))):
    row, log = read_new_run(path)
    new_rows.append(row)
    logs[row["tag"]] = (row, log)
new = pd.DataFrame(new_rows)
key = ["algo", "partition", "alpha", "mu", "clients", "seed"]
merged = (pd.concat([old, new], ignore_index=True)
          .drop_duplicates(subset=key, keep="last"))
merged.to_csv("summary_merged.csv", index=False)
print(f"{len(old)} August runs + {len(new)} new runs -> {len(merged)} rows\n")


def select(algo, alpha, clients=10, mu=None):
    d = merged[(merged.algo == algo) & (merged.partition == "dirichlet")
               & np.isclose(merged.alpha, alpha) & (merged.clients == clients)]
    if mu is not None:
        d = d[np.isclose(d.mu, mu)]
    return d.set_index("seed").sort_index()


def ms(x, scale=1.0, nd=4):
    x = np.asarray(x, dtype=float) * scale
    sd = x.std(ddof=1) if len(x) > 1 else float("nan")
    return f"{x.mean():.{nd}f} +/- {sd:.{nd}f} (n={len(x)})"


def paired(a, b, label):
    seeds = sorted(set(a.index) & set(b.index))
    d = (b.loc[seeds, "test_acc"] - a.loc[seeds, "test_acc"]) * 100
    line = " ".join(f"{s}:{v:+.2f}" for s, v in zip(seeds, d))
    out = f"  {label}: {line}\n    mean {d.mean():+.2f} +/- {d.std(ddof=1):.2f} points, " \
          f"FedProx ahead on {(d > 0).sum()} of {len(d)} seeds"
    if stats is not None and len(d) > 1:
        t, p = stats.ttest_rel(b.loc[seeds, "test_acc"], a.loc[seeds, "test_acc"])
        out += f", paired t = {t:.2f}, p = {p:.3f}"
    print(out)


print("=== alpha = 0.1, K = 10: FedAvg vs FedProx (mu = 0.001) ===")
fa, fp = select("fedavg", 0.1), select("fedprox", 0.1, mu=0.001)
for name, d in (("FedAvg ", fa), ("FedProx", fp)):
    print(f"  {name} seeds {list(d.index)}")
    print(f"    test acc  {ms(d.test_acc)}")
    print(f"    macro F1  {ms(d.test_macro_f1)}")
    print(f"    best val  {ms(d.best_val_acc)}, best round {d.best_round.min()}-{d.best_round.max()}")
    print(f"    reached 0.85 val on {d.round_to_target.notna().sum()} of {len(d)} seeds")
paired(fa, fp, "paired, all seeds")
paired(fa.loc[[42, 43, 44]], fp.loc[[42, 43, 44]], "paired, seeds 42-44 only")
print()

print("=== Client count at alpha = 0.1 (paired FedProx - FedAvg) ===")
for k in (5, 10, 50):
    paired(select("fedavg", 0.1, k), select("fedprox", 0.1, k, mu=0.001), f"K = {k}")
print()

print("=== mu sensitivity at alpha = 0.1, K = 10 ===")
for mu in (0.001, 0.01, 0.1, 1.0):
    d = select("fedprox", 0.1, mu=mu)
    common = sorted(set(d.index) & set(fa.index))
    diff = (d.loc[common, "test_acc"] - fa.loc[common, "test_acc"]) * 100
    print(f"  mu = {mu:<5} seeds {list(d.index)}: acc {ms(d.test_acc)}, "
          f"F1 {ms(d.test_macro_f1)}, best rounds {list(d.best_round)}")
    print(f"             vs FedAvg on seeds {common}: {diff.mean():+.2f} +/- "
          f"{diff.std(ddof=1) if len(diff) > 1 else float('nan'):.2f} points")
print()

print("=== Client drift (mean L2 distance from the global model) ===")
print("  tag | first logged round | mean over all rounds | mean over last 10 logged rounds")
for tag, (row, log) in sorted(logs.items()):
    dr = log["drift_mean"]
    print(f"  {tag} | {dr.iloc[0]:.3f} | {dr.mean():.3f} | {dr.iloc[-10:].mean():.3f}")
for s in (45, 46):
    a = [l for t, (r, l) in logs.items() if r["algo"] == "fedavg" and r["seed"] == s]
    b = [l for t, (r, l) in logs.items() if r["algo"] == "fedprox" and r["seed"] == s
         and np.isclose(r["mu"], 0.001)]
    if a and b:
        ratio = b[0]["drift_mean"].mean() / a[0]["drift_mean"].mean()
        print(f"  seed {s}: FedProx (mu=0.001) mean drift / FedAvg mean drift = {ratio:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
colours = {None: "tab:blue", 0.001: "tab:orange", 0.01: "tab:green", 0.1: "tab:red"}
for tag, (row, log) in sorted(logs.items()):
    mu = None if row["algo"] == "fedavg" else row["mu"]
    label = "FedAvg" if mu is None else f"FedProx $\\mu$ = {mu:g}"
    ax.plot(log["round"], log["drift_mean"], color=colours[mu],
            ls="-" if row["seed"] in (43, 45) else "--",
            label=f"{label}, seed {row['seed']}")
ax.set_xlabel("Communication round")
ax.set_ylabel("Mean client drift $\\|w_k - w^t\\|_2$")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("drift_curves.png", dpi=200)
print("\nsaved summary_merged.csv and drift_curves.png")
