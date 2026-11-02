"""Classify why rollouts fail, so the README can say more than a success rate.

    python scripts/failure_modes.py checkpoints/bc_dart_200_s0.pt checkpoints/act_dart_200_s0.pt
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copycat.env import CUBE_HALF  # noqa: E402
from copycat.evaluate import EVAL_SEED0, rollout  # noqa: E402
from copycat.train import load_policy  # noqa: E402

LIFTED = CUBE_HALF + 0.02  # cube bottom 2 cm off the table


def classify(trace) -> str:
    if trace["max_z"] < LIFTED:
        if trace["gripped_low"] > 10:
            return "froze holding the cube on the table"
        return "never got a grip"
    if trace["end_grip_closed"] and trace["end_z"] > LIFTED:
        return "froze holding the cube in the air"
    if trace["end_pad_dist"] > 0.045:
        return "dropped or placed off the pad"
    return "other"


def main():
    out = {}
    for path in sys.argv[1:]:
        policy = load_policy(path)
        traces = {}

        def on_step(envs, success):
            for i, e in enumerate(envs):
                c = e.cube_pos
                t = traces.setdefault(i, {"max_z": 0.0, "gripped_low": 0})
                t["max_z"] = max(t["max_z"], float(c[2]))
                held = e.grip > 0.01 and e.grip < 0.03 and np.linalg.norm(e.ee_pos - c) < 0.03
                t["gripped_low"] += int(held and c[2] < LIFTED)
                t["end_z"], t["end_grip_closed"] = float(c[2]), bool(held)
                t["end_pad_dist"] = float(np.linalg.norm(c[:2] - e.pad_xy))

        res = rollout(policy, list(range(EVAL_SEED0, EVAL_SEED0 + 100)), on_step=on_step)
        modes = Counter(classify(traces[i]) for i in np.where(~res["success"])[0])
        out[Path(path).stem] = {"successes": int(res["success"].sum()), "failures": dict(modes)}
        print(Path(path).stem, out[Path(path).stem])
    (ROOT / "results" / "failure_modes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
