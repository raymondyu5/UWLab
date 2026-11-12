"""Training curves for the checkerboard step-budget study (docs/claude/experiments/2026-11-03-icl-checker-steps-study.md).

Reads logs/icl/study_{bc,bcextw,bcextb,icl}_s{seed}/metrics.jsonl; x = env steps beyond D_0, success_once on white (ID) and black (OOD) squares, mean over
seeds with a min-max band (only seeds that have reached that iteration are averaged). Second row: training loss.
Usage: python scripts/imitation_learning/icl/plot_checker_study.py [--out logs/icl/study_curves.png]
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default="logs/icl")
parser.add_argument("--out", default="logs/icl/study_curves.png")
args = parser.parse_args()

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (run prefix, eval key, h index, label, colour, linestyle)  colours = categorical slots 1,2,3,5 of the reference palette
SERIES = [
    ("bc", "eval", 0, r"$\pi_{bc}$ (D$_0$ only)", "#2a78d6", "-"),
    ("bcextw", "eval", 0, r"$\pi_{bc}$ + expert, white", "#eb6834", "-"),
    ("bcextb", "eval", 0, r"$\pi_{bc}$ + expert, black (upper bound)", "#1baf7a", "-"),
    ("icl", "eval", 1, r"$\pi_{icl}$, h=1 (noisy cloud)", "#e87ba4", "-"),
    ("icl", "eval", 0, r"$\pi_{icl}$, h=0 (no context)", "#e87ba4", (0, (4, 2))),
    ("icl", "eval_clean", 1, r"$\pi_{icl}$, h=1, scored episode clean cloud", "#e87ba4", (0, (1, 1.5))),
]


def load(prefix):
    runs = []
    for f in sorted(glob.glob(f"{args.logs}/study_{prefix}_s*/metrics.jsonl")):
        runs.append([json.loads(line) for line in open(f)])
    return runs


def sc(e, h):
    return e[h].get("pass_any", e[h]["success_once"]) if isinstance(e, list) else e


def curve(runs, fn):
    n = max(len(r) for r in runs)
    xs, mean, lo, hi = [], [], [], []
    for i in range(n):
        rows = [r[i] for r in runs if len(r) > i]
        v = [fn(r) for r in rows]
        xs.append(rows[0]["env_steps"] / 1e3), mean.append(np.mean(v)), lo.append(min(v)), hi.append(max(v))
    return np.array(xs), np.array(mean), np.array(lo), np.array(hi)


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for col, (region, title) in enumerate((("id", "White squares (in distribution)"), ("ood", "Black squares (out of distribution)"))):
    ax = axes[0, col]
    for prefix, key, h, label, color, ls in SERIES:
        runs = load(prefix)
        if not runs:
            continue
        k = f"{key}_{region}"
        x, m, lo, hi = curve(runs, lambda r: sc(r[k], h))
        ax.plot(x, m, color=color, ls=ls, lw=2, label=label if col == 0 else None)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_ylabel("success (cube lifted)", color=INK2)
    ax.set_ylim(0, 1)
for col in range(2):
    ax = axes[1, col]
    for prefix, label, color in (("bc", "pi_bc", "#2a78d6"), ("bcextw", "bc + expert white", "#eb6834"), ("bcextb", "bc + expert black", "#1baf7a"),
                                 ("icl", "pi_icl", "#e87ba4")):
        runs = load(prefix)
        if runs and col == 0:
            x, m, lo, hi = curve(runs, lambda r: r["loss"])
            ax.plot(x, m, color=color, lw=2)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.set_xlabel("env steps beyond D$_0$ (thousands)", color=INK2)
    if col == 0:
        ax.set_ylabel("training loss (last 100 steps)", color=INK2)
        ax.set_yscale("log")
        ax.set_title("Training loss (not comparable across methods: different label sources)", color=INK2, fontsize=10, loc="left")
    else:
        ax.axis("off")
for ax in axes.flat:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(0.97, 0.12), frameon=False, fontsize=10)
fig.suptitle("Checkerboard 2x2: success vs extra env steps (mean over seeds, band = min-max)", color=INK, x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=130, facecolor="#fcfcfb")
print("wrote", args.out)
