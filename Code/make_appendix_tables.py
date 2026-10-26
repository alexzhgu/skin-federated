"""Turn the result CSVs into LaTeX table rows for the appendix.

Usage:
    python make_appendix_tables.py summary_merged.csv partition_stats.csv perclass_skew.csv

Writes app_runs.tex, app_partitions.tex and app_perclass.tex (rows only;
the table headers are in the thesis). Any CSV that is missing is skipped.
"""
import os
import sys
import numpy as np
import pandas as pd

RUNS, PARTS, PERCLASS = (sys.argv[1:4] + [None] * 3)[:3]

CLEAN = {
    "Acne And Rosacea Photos": "Acne and rosacea",
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions":
        "Actinic keratosis, BCC and other malignant lesions",
    "Atopic Dermatitis Photos": "Atopic dermatitis",
    "Ba  Cellulitis": "Bacterial: cellulitis",
    "Ba Cellulitis": "Bacterial: cellulitis",
    "Ba Impetigo": "Bacterial: impetigo",
    "Bullous Disease Photos": "Bullous disease",
    "Cellulitis Impetigo And Other Bacterial Infections":
        "Cellulitis, impetigo and other bacterial infections",
    "Exanthems And Drug Eruptions": "Exanthems and drug eruptions",
    "Fu Athlete Foot": "Fungal: athlete's foot",
    "Fu Nail Fungus": "Fungal: nail fungus",
    "Fu Ringworm": "Fungal: ringworm",
    "Hair Loss Photos Alopecia And Other Hair Diseases": "Hair loss and alopecia",
    "Heathy": "Healthy",
    "Herpes Hpv And Other Stds Photos": "Herpes, HPV and other STDs",
    "Light Diseases And Disorders Of Pigmentation": "Light diseases and pigmentation disorders",
    "Lupus And Other Connective Tissue Diseases": "Lupus and other connective tissue diseases",
    "Malignant": "Malignant",
    "Melanoma Skin Cancer Nevi And Moles": "Melanoma, nevi and moles",
    "Nail Fungus And Other Nail Disease": "Nail fungus and other nail disease",
    "Pa Cutaneous Larva Migrans": "Parasitic: cutaneous larva migrans",
    "Poison Ivy Photos And Other Contact Dermatitis": "Poison ivy and other contact dermatitis",
    "Rashes": "Rashes",
    "Scabies Lyme Disease And Other Infestations And Bites":
        "Scabies, Lyme disease and other infestations",
    "Systemic Disease": "Systemic disease",
    "Urticaria Hives": "Urticaria",
    "Vascular Tumors": "Vascular tumours",
    "Vasculitis Photos": "Vasculitis",
    "Vi Chickenpox": "Viral: chickenpox",
    "Vi Shingles": "Viral: shingles",
    "Warts Molluscum And Other Viral Infections": "Warts, molluscum and other viral infections",
}


def have(path):
    return path is not None and os.path.exists(path)


if have(RUNS):
    r = pd.read_csv(RUNS)
    r["alpha_sort"] = np.where(r["partition"] == "iid", 99, r["alpha"])
    r["mu_sort"] = r["mu"].fillna(0)
    r = r.sort_values(["alpha_sort", "clients", "algo", "mu_sort", "seed"], ascending=[False, True, True, True, True])
    with open("app_runs.tex", "w") as f:
        for _, x in r.iterrows():
            part = "IID" if x.partition == "iid" else f"{x.alpha:g}"
            method = "FedAvg" if x.algo == "fedavg" else f"FedProx ({x.mu:g})"
            f.write(f"{part} & {int(x.clients)} & {method} & {int(x.seed)} & "
                    f"{x.test_acc:.4f} & {x.test_macro_f1:.4f} & {x.best_val_acc:.4f} & "
                    f"{int(x.best_round)} & {x.total_comms_gb:.1f} \\\\\n")
    print(f"app_runs.tex: {len(r)} rows")

if have(PARTS):
    p = pd.read_csv(PARTS)
    with open("app_partitions.tex", "w") as f:
        for _, x in p.iterrows():
            part = "IID" if x.partition == "IID" else x.partition.replace("alpha=", "")
            f.write(f"{part} & {int(x.K)} & {int(x.seed)} & {int(x.size_min):,}--{int(x.size_max):,} & "
                    f"{int(x.size_median):,} & {int(x.classes_min)} & {x.classes_median:g} & "
                    f"{int(x.empty_cells)}/{int(x.total_cells)} & {int(x.rounds_all_30)} \\\\\n")
    print(f"app_partitions.tex: {len(p)} rows")

if have(PERCLASS):
    c = pd.read_csv(PERCLASS, index_col=0)
    with open("app_perclass.tex", "w") as f:
        for name, x in c.iterrows():
            group = "Source" if x.group == "source" else "DermNet"
            f.write(f"{CLEAN.get(name, name)} & {group} & {x.iid:.3f} & {x.avg_a01:.3f} & "
                    f"{x['drop']:.1f} & {x.prox_a01:.3f} & {x.prox_gain:+.1f} \\\\\n")
    print(f"app_perclass.tex: {len(c)} rows")
