"""Supervised training on demonstrations: L1 regression of (normalized, joint-relative) action chunks."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .data import Episode, chunk_targets
from .policies import Policy, to_relative


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(policy: Policy, demos: list[Episode], steps: int = 20_000, batch: int = 256, lr: float = 1e-3,
          weight_decay: float = 1e-4, warmup: int = 500, seed: int = 0, log_every: int = 0,
          dev: torch.device | None = None) -> dict:
    """Train in place for a fixed number of gradient steps (same budget for every dataset size)."""
    dev = dev or device()
    torch.backends.cuda.matmul.allow_tf32 = True  # ~2x on Ampere, irrelevant for L1 regression precision
    torch.manual_seed(seed)
    obs, targets = chunk_targets(demos, policy.k)
    targets = to_relative(obs, targets)
    policy.obs_norm.fit(obs)
    policy.act_norm.fit(targets)
    policy.to(dev).train()
    X = policy.obs_norm.norm(torch.as_tensor(obs, device=dev))
    Y = policy.act_norm.norm(torch.as_tensor(targets, device=dev))

    use_graph = dev.type == "cuda"
    # With a CUDA graph the lr must live on the GPU so the captured optimizer step can read it.
    opt = torch.optim.AdamW(policy.parameters(), lr=torch.tensor(lr, device=dev) if use_graph else lr,
                            weight_decay=weight_decay, capturable=use_graph)

    def schedule(s):  # linear warmup, then cosine decay to zero
        return lr * min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))

    def train_step():
        idx = torch.randint(len(X), (batch,), device=dev)
        loss = (policy(X[idx]) - Y[idx]).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, foreach=True)
        opt.step()
        opt.zero_grad(set_to_none=False)
        return loss

    if use_graph:
        # These nets are tiny, so on Windows the step time is all kernel-launch latency (~25 ms for
        # the transformer). Capturing the whole step once and replaying it is ~10x faster.
        opt.param_groups[0]["lr"].fill_(schedule(0))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up steps, required before capture (they count as training)
                train_step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_loss = train_step()

    t0, losses = time.time(), []
    for step in range(steps):
        if use_graph:
            opt.param_groups[0]["lr"].fill_(schedule(step))
            graph.replay()
            loss = static_loss
        else:
            for g in opt.param_groups:
                g["lr"] = schedule(step)
            loss = train_step()
        if step % 100 == 0 or step == steps - 1:
            losses.append(loss.item())
            if log_every and step % log_every == 0:
                print(f"  step {step:6d}  loss {losses[-1]:.4f}")
    policy.eval()
    return {"final_loss": float(np.mean(losses[-10:])), "seconds": time.time() - t0,
            "samples": len(X), "loss_curve": losses}


def save_policy(policy: Policy, path, **extra):
    torch.save({"config": policy.config, "state_dict": policy.state_dict(), **extra}, path)


def load_policy(path, dev: torch.device | str | None = "cpu") -> Policy:
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    policy = Policy(**ckpt["config"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(dev).eval()
