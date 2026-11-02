"""Demonstration collection with DART-style noise injection, plus PyTorch datasets.

DART (Laskey et al., 2017): execute `expert_action + noise`, but store the clean `expert_action`
as the label. The robot drifts into states the clean expert never visits, and the dataset shows
how to get back from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .env import MAX_STEPS, PickPlaceEnv
from .expert import ScriptedExpert

NOISE_RHO = 0.8  # AR(1) correlation per tick: white noise at 20 Hz just gets filtered out by the servos


@dataclass
class Episode:
    obs: np.ndarray      # (T, OBS_DIM)
    act: np.ndarray      # (T, ACT_DIM) clean expert labels
    success: bool
    seed: int


def run_expert(seed: int, noise: float = 0.0, max_steps: int = MAX_STEPS, env: PickPlaceEnv | None = None) -> Episode:
    """One expert episode. `noise` is the std (rad) of the perturbation added to the executed joint targets."""
    env = env or PickPlaceEnv()
    rng = np.random.default_rng(seed + 1_000_003)
    obs = env.reset(seed)
    expert = ScriptedExpert()
    eps = np.zeros(6)
    O, A, ok = [], [], False
    for _ in range(max_steps):
        a = expert.act(env)
        O.append(obs)
        A.append(a)
        eps = NOISE_RHO * eps + np.sqrt(1 - NOISE_RHO**2) * noise * rng.standard_normal(6)
        executed = a.copy()
        executed[:6] += eps
        obs, ok = env.step(executed)
        if ok:
            break
    return Episode(np.array(O, np.float32), np.array(A, np.float32), ok, seed)


def _job(args):
    seed, noise = args
    return run_expert(seed, noise)


def collect(n: int, noise: float, seed0: int = 0, workers: int = 8) -> tuple[list[Episode], int]:
    """Collect `n` successful demos. Each episode draws its noise level uniformly from [0, noise],
    so the data covers both clean and heavily perturbed behaviour. Returns (demos, attempts)."""
    rng = np.random.default_rng(seed0)
    demos, attempts, seed = [], 0, seed0
    with Pool(workers) as pool:
        while len(demos) < n:
            batch = [(seed + i, float(rng.uniform(0, noise))) for i in range(max(workers, n - len(demos)))]
            seed += len(batch)
            for ep in pool.map(_job, batch):
                attempts += 1
                if ep.success and len(demos) < n:
                    demos.append(ep)
    return demos, attempts


def save_demos(path: str | Path, demos: list[Episode]):
    lengths = np.array([len(d.obs) for d in demos])
    np.savez_compressed(path, obs=np.concatenate([d.obs for d in demos]),
                        act=np.concatenate([d.act for d in demos]), lengths=lengths,
                        seeds=np.array([d.seed for d in demos]))


def load_demos(path: str | Path, n: int | None = None) -> list[Episode]:
    z = np.load(path)
    bounds = np.concatenate([[0], np.cumsum(z["lengths"])])
    k = len(z["lengths"]) if n is None else n
    if k > len(z["lengths"]):
        raise ValueError(f"asked for {k} demos, {path} only has {len(z['lengths'])}")
    return [Episode(z["obs"][a:b], z["act"][a:b], True, int(s))
            for a, b, s in zip(bounds[:k], bounds[1:k + 1], z["seeds"][:k])]


def chunk_targets(demos: list[Episode], k: int) -> tuple[np.ndarray, np.ndarray]:
    """(obs_t, [a_t, ..., a_{t+k-1}]) pairs. Past the end of an episode the last action is repeated:
    the expert is done and holding still, which is exactly what it would keep commanding."""
    O, Y = [], []
    for d in demos:
        T = len(d.act)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(k)[None], T - 1)
        O.append(d.obs)
        Y.append(d.act[idx])
    return np.concatenate(O), np.concatenate(Y)
