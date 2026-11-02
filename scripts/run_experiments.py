"""Train + evaluate every config of the study; results go to results/results.json (resumable).

    python scripts/run_experiments.py                 # everything
    python scripts/run_experiments.py --only main     # just the demos ablation

Training runs on the GPU one model at a time, then evaluation (MuJoCo, CPU) fans out over processes.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copycat.data import load_demos  # noqa: E402
from copycat.policies import Policy  # noqa: E402

RESULTS = ROOT / "results" / "results.json"
CKPT = ROOT / "checkpoints"
DATASETS = {"dart": ROOT / "data" / "demos_dart.npz", "clean": ROOT / "data" / "demos_clean.npz"}
DEMO_COUNTS = [10, 25, 50, 100, 200]
CHUNK = 10
STEPS = 20_000
EVAL_EPISODES = 100

# name -> (Policy kwargs, training kwargs)
POLICIES = {
    "bc": (dict(kind="bc", k=1), dict(lr=1e-3)),
    "chunk_mlp": (dict(kind="chunk_mlp", k=CHUNK), dict(lr=1e-3)),
    "act": (dict(kind="act", k=CHUNK, d_model=64), dict(lr=3e-4)),
}


def experiments():
    """(study, policy, dataset, n_demos, train seed, execution mode, Policy overrides)"""
    out = []
    for n in DEMO_COUNTS:
        for p in POLICIES:
            out.append(("main", p, "dart", n, 0, "ensemble", {}))
    for n in DEMO_COUNTS:
        for p in ("bc", "act"):
            out.append(("no_dart", p, "clean", n, 0, "ensemble", {}))
    for mode in ("latest", "open_loop"):
        for p in ("chunk_mlp", "act"):
            out.append(("execution", p, "dart", 50, 0, mode, {}))
    for k in (1, 5, 20):
        out.append(("chunk_size", "act", "dart", 50, 0, "ensemble", {"k": k}))
    for seed in (1, 2):
        for p in POLICIES:
            out.append(("seeds", p, "dart", 50, seed, "ensemble", {}))
    return out


def key(p, data, n, seed, over):
    extra = "".join(f"_{k}{v}" for k, v in over.items())
    return f"{p}_{data}_{n}_s{seed}{extra}"


def train_missing(todo, steps):
    from copycat.train import save_policy, train

    for study, p, data, n, seed, mode, over in todo:
        ckpt = CKPT / f"{key(p, data, n, seed, over)}.pt"
        if ckpt.exists():
            continue
        pkw, tkw = POLICIES[p]
        policy = Policy(**{**pkw, **over})
        info = train(policy, load_demos(DATASETS[data], n), steps=steps, seed=seed, **tkw)
        save_policy(policy, ckpt, train_info=info)
        print(f"trained {ckpt.name:30s} {info['seconds']:5.0f}s  loss {info['final_loss']:.4f}", flush=True)


def eval_job(exp):
    import torch

    from copycat.evaluate import evaluate
    from copycat.train import load_policy

    torch.set_num_threads(2)
    study, p, data, n, seed, mode, over = exp
    k = key(p, data, n, seed, over)
    ckpt = torch.load(CKPT / f"{k}.pt", map_location="cpu", weights_only=False)
    policy = load_policy(CKPT / f"{k}.pt")
    t0 = time.time()
    res = evaluate(policy, EVAL_EPISODES, mode=mode)
    info = ckpt["train_info"]
    return {"study": study, "key": k, "policy": p, "data": data, "n_demos": n, "seed": seed, "mode": mode,
            "k": policy.k, **res, "train_seconds": info["seconds"], "final_loss": info["final_loss"],
            "params": sum(x.numel() for x in policy.parameters()), "eval_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run just one study (main, no_dart, execution, chunk_size, seeds)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {(r["study"], r["key"], r["mode"]) for r in results}
    todo = [e for e in experiments()
            if (not args.only or e[0] == args.only) and (e[0], key(*e[1:5], e[6]), e[5]) not in done]
    CKPT.mkdir(exist_ok=True)
    RESULTS.parent.mkdir(exist_ok=True)

    train_missing(todo, args.steps)
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(eval_job, todo):
            results.append(row)
            RESULTS.write_text(json.dumps(results, indent=1))
            print(f"[{row['study']:10s}] {row['key']:28s} {row['mode']:9s} success {row['successes']:3d}/{row['n']}",
                  flush=True)


if __name__ == "__main__":
    main()
