"""Collect expert demonstrations into data/.

    python scripts/collect.py --n 200 --noise 0.02 --out data/demos_dart.npz
    python scripts/collect.py --n 200 --noise 0.0  --out data/demos_clean.npz --seed0 50000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copycat.data import collect, save_demos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.02, help="max DART noise std (rad); 0 = clean demos")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "demos_dart.npz"))
    args = ap.parse_args()

    t0 = time.time()
    demos, attempts = collect(args.n, args.noise, args.seed0, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_demos(args.out, demos)
    lengths = [len(d.obs) for d in demos]
    print(f"{len(demos)} demos from {attempts} attempts ({len(demos) / attempts:.1%} expert success), "
          f"{sum(lengths)} samples, mean length {np.mean(lengths):.1f} ticks, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
