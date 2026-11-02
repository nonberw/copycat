"""Batched policy rollouts: N envs step in lockstep so the policy runs one forward pass per tick."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from .env import MAX_STEPS, PickPlaceEnv, build_xml
from .policies import ChunkExecutor, Policy

EVAL_SEED0 = 100_000  # demos use seeds < 100k, so evaluation layouts are never seen in training


def rollout(policy: Policy, seeds: list[int], mode: str = "ensemble", m: float = 0.1,
            max_steps: int = MAX_STEPS, on_step: Callable[[list[PickPlaceEnv], np.ndarray], None] | None = None
            ) -> dict:
    model = mujoco.MjModel.from_xml_string(build_xml())
    envs = [PickPlaceEnv(model) for _ in seeds]
    obs = np.stack([e.reset(s) for e, s in zip(envs, seeds)])
    executor = ChunkExecutor(policy, mode, m)
    success = np.zeros(len(envs), bool)
    steps = np.full(len(envs), max_steps)
    for t in range(max_steps):
        actions = executor(obs)
        for i, env in enumerate(envs):
            if success[i]:
                continue  # finished envs freeze; their rows still go through the batch, harmlessly
            obs[i], ok = env.step(actions[i])
            if ok:
                success[i], steps[i] = True, t + 1
        if on_step:
            on_step(envs, success)
        if success.all():
            break
    return {"success": success, "steps": steps, "rate": float(success.mean())}


def evaluate(policy: Policy, n: int = 100, seed0: int = EVAL_SEED0, batch: int = 100, **kw) -> dict:
    seeds = list(range(seed0, seed0 + n))
    parts = [rollout(policy, seeds[i:i + batch], **kw) for i in range(0, n, batch)]
    success = np.concatenate([p["success"] for p in parts])
    steps = np.concatenate([p["steps"] for p in parts])
    return {"n": n, "successes": int(success.sum()), "rate": float(success.mean()),
            "mean_steps_success": float(steps[success].mean()) if success.any() else None,
            "ci95": wilson(int(success.sum()), n)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate; honest error bars for 100 coin flips."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))
