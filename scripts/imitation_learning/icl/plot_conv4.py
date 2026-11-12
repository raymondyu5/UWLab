"""Success curves for the conv4 checkerboard study: one figure per region (white = ID, black = OOD).
Usage: python plot_conv4.py [--out_prefix logs/icl/conv4]   (mean over the seeds found; band = min-max)"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out_prefix", default="logs/icl/conv4")
a = p.parse_args()
# method -> (run prefix, eval index h, label, colour, linestyle)
M = [("icl", 0, "ICL first try (h=0)", "#d6604d", "-"), ("icl", 1, "ICL pass@2", "#d6604d", "--"),
     ("bcextw", 0, "bc_ext white pass@1", "#4393c3", "-"), ("bcextw", 1, "bc_ext white pass@2", "#4393c3", "--"),
     ("bcextb", 0, "bc_ext black pass@1 (upper bound)", "#1b7837", "-"), ("bcextb", 1, "bc_ext black pass@2", "#1b7837", "--")]
REF = {"id": (0.594, 0.82), "ood": (0.340, 0.85)}  # fixed pi_bc (clean-cloud eval) and the expert with the true cube


def load(prefix):
    out = []
    for f in sorted(glob.glob(f"logs/icl/conv4_{prefix}_s*/metrics.jsonl")):
        out.append([json.loads(l) for l in open(f)])
    return out


for key, name in (("eval_id", "white (ID)"), ("eval_ood", "black (OOD)")):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pre, h, lab, col, ls in M:
        runs = load(pre)
        if not runs:
            continue
        n = min(len(r) for r in runs)
        x = np.array([r["env_steps"] for r in runs[0][:n]]) / 1e3
        y = np.array([[r[key][h]["success_once" if h == 0 else "pass_any"] for r in rr[:n]] for rr in runs])
        ax.plot(x, y.mean(0), color=col, ls=ls, label=f"{lab} [{len(runs)} seed{'s' if len(runs) > 1 else ''}]")
        if len(runs) > 1:
            ax.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.12)
    r_bc, r_ex = REF["id" if key == "eval_id" else "ood"]
    ax.axhline(r_bc, color="gray", ls=":", label="fixed pi_bc (clean eval)")
    ax.axhline(r_ex, color="k", ls=":", lw=0.8, label="expert, true cube")
    ax.set_xlabel("env steps beyond D_0 (thousands)"); ax.set_ylabel("success_once"); ax.set_ylim(0, 1)
    ax.set_title(f"{name}, 256 fixed tasks, clean cloud at every try"); ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    fn = f"{a.out_prefix}_success_{'white' if key == 'eval_id' else 'black'}.png"
    fig.savefig(fn, dpi=110, bbox_inches="tight"); print(fn)
