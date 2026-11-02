"""Policies: plain BC MLP, MLP with action chunks, and an ACT-style transformer with action chunks.

Every policy maps a raw observation batch (B, OBS_DIM) to a raw action chunk (B, k, ACT_DIM);
plain BC is just k = 1. Normalization statistics live inside the module as buffers, so a saved
checkpoint is all you need to run it.

Actions are absolute joint targets, but the network predicts them relative to the current joint
angles: a target spans ~0.3 rad across the dataset while one tick of motion is ~0.03 rad, so a
regression error of a few percent on the absolute value would swallow the whole step.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .env import ACT_DIM, OBS_DIM

KINDS = ("bc", "chunk_mlp", "act")
# Observation slices fed to the transformer as separate tokens.
TOKENS = {"proprio": slice(0, 13), "cube": slice(13, 18), "pad": slice(18, 20)}


class Normalizer(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))

    def fit(self, x: np.ndarray):
        self.mean.copy_(torch.as_tensor(x.mean(0)))
        self.std.copy_(torch.as_tensor(x.std(0) + 1e-3))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


def mlp(din: int, dout: int, hidden: int, depth: int, dropout: float) -> nn.Sequential:
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, dout))


class Policy(nn.Module):
    def __init__(self, kind: str, k: int = 1, hidden: int = 256, depth: int = 3, d_model: int = 128,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if kind == "bc" and k != 1:
            raise ValueError("plain BC predicts one action (k=1); use chunk_mlp for chunks")
        self.config = dict(kind=kind, k=k, hidden=hidden, depth=depth, d_model=d_model, heads=heads,
                           layers=layers, dropout=dropout)
        self.kind, self.k = kind, k
        # Per (future step, action dim): a target 10 ticks ahead moves much further than the next one.
        self.obs_norm, self.act_norm = Normalizer(OBS_DIM), Normalizer(k, ACT_DIM)
        if kind in ("bc", "chunk_mlp"):
            self.net = mlp(OBS_DIM, k * ACT_DIM, hidden, depth, dropout)
        else:
            self.embed = nn.ModuleDict({n: nn.Linear(s.stop - s.start, d_model) for n, s in TOKENS.items()})
            self.token_pos = nn.Parameter(torch.randn(len(TOKENS), d_model) * 0.02)
            self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)  # one learned query per future step
            kw = dict(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model, dropout=dropout,
                      activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**kw), layers,
                                                 norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**kw), layers, norm=nn.LayerNorm(d_model))
            self.head = nn.Linear(d_model, ACT_DIM)

    def forward(self, obs_n: torch.Tensor) -> torch.Tensor:
        """Normalized obs (B, OBS_DIM) -> normalized action chunk (B, k, ACT_DIM)."""
        if self.kind != "act":
            return self.net(obs_n).view(-1, self.k, ACT_DIM)
        tokens = torch.stack([self.embed[n](obs_n[:, s]) for n, s in TOKENS.items()], 1) + self.token_pos
        queries = self.queries.expand(len(obs_n), -1, -1)
        # Sequences of 3 and 10 tokens: the fused attention kernels tile for 64+ and waste ~half the
        # step, while plain matmuls are instant.
        with sdpa_kernel(SDPBackend.MATH):
            return self.head(self.decoder(queries, self.encoder(tokens)))

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Raw obs (B, OBS_DIM) -> raw action chunk (B, k, ACT_DIM)."""
        x = torch.as_tensor(np.asarray(obs, np.float32), device=self.obs_norm.mean.device)
        rel = self.act_norm.denorm(self(self.obs_norm.norm(x))).cpu().numpy()
        return from_relative(np.asarray(obs, np.float32), rel)


def to_relative(obs: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Absolute action chunk (B, k, ACT_DIM) -> joint targets minus current joints; gripper untouched."""
    out = chunk.copy()
    out[..., :6] -= obs[:, None, :6]
    return out


def from_relative(obs: np.ndarray, rel: np.ndarray) -> np.ndarray:
    out = rel.copy()
    out[..., :6] += obs[:, None, :6]
    return out


class ChunkExecutor:
    """Turns chunk predictions into one action per tick for a batch of envs running in lockstep.

    mode="ensemble": query every tick and average all predictions for the current step with
                     weights exp(-m * i), i = 0 for the oldest (temporal ensembling, as in ACT).
    mode="latest":   query every tick, execute only the first action of the newest chunk.
    mode="open_loop": query every k ticks and play the whole chunk blind.
    """

    def __init__(self, policy: Policy, mode: str = "ensemble", m: float = 0.1):
        self.policy, self.mode, self.m = policy, mode, m
        self.reset()

    def reset(self):
        self.chunks: list[np.ndarray] = []  # newest last; chunks[-j] was predicted j-1 ticks ago
        self.t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        k = self.policy.k
        if self.mode == "open_loop":
            if self.t % k == 0:
                self.chunks = [self.policy.predict(obs)]
            a = self.chunks[0][:, self.t % k]
        else:
            self.chunks = (self.chunks + [self.policy.predict(obs)])[-k:]
            if self.mode == "latest":
                a = self.chunks[-1][:, 0]
            else:
                n = len(self.chunks)
                preds = np.stack([c[:, n - 1 - i] for i, c in enumerate(self.chunks)])  # oldest first
                w = np.exp(-self.m * np.arange(n))
                a = np.tensordot(w / w.sum(), preds, axes=1)
        self.t += 1
        return a
