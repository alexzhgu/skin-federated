"""Per-class effects of label skew.

Compares per-class test F1 for FedAvg under IID (seeds 42-44) with FedAvg at
alpha = 0.1, K = 10 (seeds 42-46), and FedProx (mu = 0.001) with FedAvg at
alpha = 0.1 on the same five seeds.

Usage:
    python perclass_skew.py fed_runs post_analysis
(list every folder that holds _report.txt files)

Outputs:
    perclass_skew.csv   per-class F1 for each condition and the differences
    printed summary and LaTeX rows for the ten largest drops
"""
import os
import re
import sys
import numpy as np
import pandas as pd

DIRS = sys.argv[1:] or ["fed_runs", "post_analysis"]
IID = ["fedavg_iid_seed42", "fedavg_iid_seed43", "fedavg_iid_seed44"]
AVG = ["fedavg_dirichlet_a0.1_c10_seed42", "fedavg_dirichlet_a0.1_seed43",
       "fedavg_dirichlet_a0.1_seed44", "fedavg_dirichlet_a0.1_c10_seed45",
       "fedavg_dirichlet_a0.1_c10_seed46"]
PROX = ["fedprox_dirichlet_a0.1_mu0.001_seed42", "fedprox_dirichlet_a0.1_mu0.001_seed43",
        "fedprox_dirichlet_a0.1_mu0.001_seed44", "fedprox_dirichlet_a0.1_mu0.001_c10_seed45",
        "fedprox_dirichlet_a0.1_mu0.001_c10_seed46"]
ROW = re.compile(r"^\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
SKIP = {"accuracy", "macro avg", "weighted avg"}


def find(tag):
    for d in DIRS:
        p = os.path.join(d, f"{tag}_report.txt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{tag}_report.txt not found in {DIRS}")


def per_class_f1(tag):
    out = {}
    with open(find(tag)) as f:
        for line in f:
            m = ROW.match(line)
            if m and m.group(1).strip() not in SKIP:
                out[m.group(1).strip()] = float(m.group(4))
    return out


def mean_f1(tags):
    return pd.DataFrame([per_class_f1(t) for t in tags]).mean()


df = pd.DataFrame({"iid": mean_f1(IID), "avg_a01": mean_f1(AVG), "prox_a01": mean_f1(PROX)})
df["drop"] = (df["iid"] - df["avg_a01"]) * 100
df["prox_gain"] = (df["prox_a01"] - df["avg_a01"]) * 100
source = df.index.str.match(r"^(Ba|Fu|Vi|Pa) ") | df.index.isin(["Malignant", "Heathy"])
df["group"] = np.where(source, "source", "dermnet")
df = df.sort_values("drop", ascending=False)
df.to_csv("perclass_skew.csv")

print("Per-class F1 drop from IID to alpha = 0.1 (FedAvg, points)")
print(df[["iid", "avg_a01", "drop", "group"]].round(3).to_string())
print(f"\nmean drop over all classes: {df['drop'].mean():.2f} points")
print(df.groupby("group")["drop"].agg(["mean", "std", "min", "max", "count"]).round(2))
print(f"\ncorrelation between IID F1 and drop: {np.corrcoef(df['iid'], df['drop'])[0, 1]:.2f}")

print("\nFedProx minus FedAvg per-class F1 at alpha = 0.1 (points)")
print(f"classes where FedProx is higher: {(df['prox_gain'] > 0).sum()} of {len(df)}")
print(df.groupby("group")["prox_gain"].agg(["mean", "std"]).round(2))
print(df["prox_gain"].sort_values(ascending=False).round(2).to_string())

print("\nLaTeX rows, ten largest drops:")
for name, r in df.head(10).iterrows():
    print(f"    {name} & {r.iid:.3f} & {r.avg_a01:.3f} & {r['drop']:.1f} \\\\")
