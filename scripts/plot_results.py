"""Plot results/results.json into assets/results.png and print the markdown tables for the README."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"bc": "#9e5cff", "chunk_mlp": "#ff59a6", "act": "#1fc8d6"}
NAMES = {"bc": "BC-MLP", "chunk_mlp": "Chunk-MLP (k=10)", "act": "ACT-lite (k=10)"}


def rows(results, **match):
    return [r for r in results if all(r[k] == v for k, v in match.items())]


def curve(ax, results, study, policy, data, style="-", label=None):
    rs = sorted(rows(results, study=study, policy=policy, data=data, mode="ensemble"), key=lambda r: r["n_demos"])
    if not rs:
        return
    n = np.array([r["n_demos"] for r in rs])
    p = np.array([r["rate"] for r in rs]) * 100
    lo = p - np.array([r["ci95"][0] for r in rs]) * 100
    hi = np.array([r["ci95"][1] for r in rs]) * 100 - p
    ax.errorbar(n, p, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], fmt=style + "o", color=COLORS[policy], capsize=3, lw=2, ms=5,
                label=label or NAMES[policy])


def main():
    results = json.loads((ROOT / "results" / "results.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})

    ax = axes[0]
    for p in NAMES:
        curve(ax, results, "main", p, "dart")
    ax.set(xscale="log", xlabel="expert demos", ylabel="success rate, % (100 unseen layouts)",
           title="More demos, more cube", ylim=(-3, 103))
    ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for p in ("bc", "act"):
        curve(ax, results, "main", p, "dart", "-", f"{NAMES[p]}, DART demos")
        curve(ax, results, "no_dart", p, "clean", "--", f"{NAMES[p]}, clean demos")
    ax.set(xscale="log", xlabel="expert demos", title="Does noise injection help?", ylim=(-3, 103))
    ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    ax = axes[2]
    labels, vals, cols, errs = [], [], [], []
    for p in ("chunk_mlp", "act"):
        for mode in ("ensemble", "latest", "open_loop"):
            study = "main" if mode == "ensemble" else "execution"
            r = rows(results, study=study, policy=p, data="dart", n_demos=50, mode=mode, seed=0)
            if r:
                labels.append(mode.replace("_", "-"))
                vals.append(r[0]["rate"] * 100)
                cols.append(COLORS[p])
                errs.append([vals[-1] - r[0]["ci95"][0] * 100, r[0]["ci95"][1] * 100 - vals[-1]])
    x = np.arange(len(vals)) + np.repeat([0, 0.5], 3)[: len(vals)]
    ax.bar(x, vals, color=cols, alpha=0.85, yerr=np.clip(np.array(errs).T, 0, None), capsize=3,
           error_kw={"ecolor": "#555"})
    for xi, v in zip(x, vals):
        ax.text(xi + 0.08, v + 1.5, f"{v:.0f}", ha="left", fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    for p in ("chunk_mlp", "act"):
        ax.bar([0], [0], color=COLORS[p], label=NAMES[p])
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set(ylim=(0, 108), title="How to execute a chunk (50 demos)")

    fig.tight_layout()
    out = ROOT / "assets" / "results.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    tables(results)


def fmt(r):
    lo, hi = r["ci95"]
    return f"{r['rate'] * 100:.0f}% [{lo * 100:.0f}, {hi * 100:.0f}]"


def tables(results):
    print("\n| policy | " + " | ".join(f"{n} demos" for n in (10, 25, 50, 100, 200)) + " |")
    print("|---|" + "---|" * 5)
    for data, study in (("dart", "main"), ("clean", "no_dart")):
        for p in NAMES:
            rs = {r["n_demos"]: r for r in rows(results, study=study, policy=p, data=data, mode="ensemble")}
            if rs:
                print(f"| {NAMES[p]}, {data} | " + " | ".join(fmt(rs[n]) if n in rs else "-" for n in (10, 25, 50, 100, 200)) + " |")
    print()
    for r in results:
        if r["study"] in ("execution", "chunk_size", "seeds"):
            print(f"{r['study']:10s} {r['key']:28s} {r['mode']:9s} {fmt(r)}")
    print()
    for r in rows(results, study="main"):
        print(f"{r['key']:28s} params {r['params']:7d}  train {r['train_seconds'] or 0:5.0f}s  loss {r['final_loss'] or 0:.4f}"
              f"  steps-to-success {r['mean_steps_success']}")


if __name__ == "__main__":
    sys.exit(main())
