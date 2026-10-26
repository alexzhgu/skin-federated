"""
Baseline summary and per-class LaTeX table from a baseline run directory.

Reads <run>/test_results.json and <run>/config.json, prints test accuracy,
macro F1, best epoch and epochs run, and writes a per-class precision / recall /
F1 / support table sorted by ascending F1. Class names are converted to the
names used in Table 3.1 of the thesis.

Usage:
    python baseline_report.py --run runs/20260718_085832 --n 10   # 10 lowest + 10 highest (Table 4.1)
    python baseline_report.py --run runs/20260718_085832 --n 0 \
        --out results/baseline_perclass_full.tex                  # all 30 classes (appendix)
"""

from pathlib import Path
import argparse
import json
import re

SUMMARY_KEYS = {"accuracy", "macro avg", "weighted avg"}


def has(*words, without=()):
    return lambda s: all(w in s for w in words) and not any(w in s for w in without)


# Display names from Table 3.1. Each rule matches the dataset's folder name after
# normalise(). A folder name must match exactly one rule, otherwise the script stops.
DISPLAY_RULES = [
    ("Acne and rosacea", has("acne")),
    ("Actinic keratosis, BCC and other malignant lesions", has("actinic")),
    ("Atopic dermatitis", has("atopic")),
    ("Bacterial: cellulitis", has("cellulitis", without=("impetigo",))),
    ("Bacterial: impetigo", has("impetigo", without=("cellulitis",))),
    ("Bullous disease", has("bullous")),
    ("Cellulitis, impetigo and other bacterial infections", has("cellulitis", "impetigo")),
    ("Exanthems and drug eruptions", has("exanthem")),
    ("Fungal: athlete's foot", has("athlete")),
    ("Fungal: nail fungus", has("nail", "fungus", without=("disease",))),
    ("Fungal: ringworm", has("ringworm")),
    ("Hair loss and alopecia", has("hair")),
    ("Healthy", has("healthy")),
    ("Herpes, HPV and other STDs", has("herpes")),
    ("Light diseases and pigmentation disorders", has("pigment")),
    ("Lupus and other connective tissue diseases", has("lupus")),
    ("Malignant", lambda s: s == "malignant"),
    ("Melanoma, nevi and moles", has("melanoma")),
    ("Nail fungus and other nail disease", has("nail", "disease")),
    ("Parasitic: cutaneous larva migrans", has("larva")),
    ("Poison ivy and other contact dermatitis", has("poison")),
    ("Rashes", has("rash")),
    ("Scabies, Lyme disease and other infestations", has("scabies")),
    ("Systemic disease", has("systemic")),
    ("Urticaria", has("urticaria")),
    ("Vascular tumours", has("vascular", "tumo")),
    ("Vasculitis", has("vasculitis")),
    ("Viral: chickenpox", has("chicken")),
    ("Viral: shingles", has("shingles")),
    ("Warts, molluscum and other viral infections", has("wart")),
]


def normalise(name):
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def build_name_map(raw_names):
    mapping, problems = {}, []
    for raw in raw_names:
        hits = [d for d, rule in DISPLAY_RULES if rule(normalise(raw))]
        if len(hits) != 1:
            problems.append(f"  {raw!r} matched {hits or 'nothing'}")
        else:
            mapping[raw] = hits[0]
    taken = {}
    for raw, disp in mapping.items():
        if disp in taken:
            problems.append(f"  {raw!r} and {taken[disp]!r} both map to {disp!r}")
        taken[disp] = raw
    if problems:
        raise SystemExit("class names that need a rule in DISPLAY_RULES:\n"
                         + "\n".join(problems))
    return mapping


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def row(name, m):
    return (f"    {tex_escape(name)} & {m['precision']:.3f} & {m['recall']:.3f} & "
            f"{m['f1-score']:.3f} & {int(m['support'])} \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="0 = all classes; N = N lowest and N highest")
    ap.add_argument("--out", default="results/baseline_perclass.tex")
    args = ap.parse_args()

    run = Path(args.run)
    res = json.loads((run / "test_results.json").read_text())
    cfg_path = run / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    log_path = run / "train_log.csv"
    epochs_run = (len(log_path.read_text().splitlines()) - 1) if log_path.exists() else None

    print(f"run:          {run}")
    print(f"classes:      {cfg.get('num_classes')}  seed: {cfg.get('seed')}")
    print(f"test acc:     {res['test_acc']:.4f}")
    print(f"macro F1:     {res['test_macro_f1']:.4f}")
    print(f"best val acc: {res['best_val_acc']:.4f} at epoch {res['best_epoch']}")
    print(f"epochs run:   {epochs_run}")

    raw = {k: v for k, v in res["per_class"].items() if k not in SUMMARY_KEYS}
    names = build_name_map(raw)
    pc = {names[k]: v for k, v in raw.items()}
    ranked = sorted(pc.items(), key=lambda kv: kv[1]["f1-score"])

    print("\nname mapping:")
    for k in sorted(raw):
        print(f"  {k:<70} -> {names[k]}")
    print("\nlowest F1:")
    for name, m in ranked[:5]:
        print(f"  {m['f1-score']:.3f}  {name}")
    print("highest F1:")
    for name, m in ranked[-5:]:
        print(f"  {m['f1-score']:.3f}  {name}")

    lines = [r"  \begin{tabular}{p{7.5cm}cccc}", r"    \hline",
             r"    Class & Precision & Recall & F1 & Support \\ \hline"]
    if args.n and 2 * args.n < len(ranked):
        lines += [row(k, v) for k, v in ranked[:args.n]]
        lines.append(r"    \multicolumn{5}{c}{\dots} \\")
        lines += [row(k, v) for k, v in ranked[-args.n:]]
    else:
        lines += [row(k, v) for k, v in ranked]
    lines += [r"    \hline", r"  \end{tabular}"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()