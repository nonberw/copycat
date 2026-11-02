import numpy as np
import pytest
import torch

from copycat.data import run_expert
from copycat.env import ACT_DIM, OBS_DIM
from copycat.evaluate import rollout, wilson
from copycat.policies import ChunkExecutor, Policy
from copycat.train import load_policy, save_policy, train


@pytest.mark.parametrize("kind,k", [("bc", 1), ("chunk_mlp", 4), ("act", 4)])
def test_forward_shapes(kind, k):
    policy = Policy(kind, k=k)
    out = policy(torch.zeros(5, OBS_DIM))
    assert out.shape == (5, k, ACT_DIM)
    assert policy.predict(np.zeros((3, OBS_DIM))).shape == (3, k, ACT_DIM)


def test_bc_rejects_chunks():
    with pytest.raises(ValueError):
        Policy("bc", k=5)


class FakePolicy:
    """Chunk predicted at tick t is [t*10 + j for j in range(k)], so every average is checkable by hand."""
    k = 3

    def __init__(self):
        self.calls = 0

    def predict(self, obs):
        base = 10.0 * self.calls
        self.calls += 1
        chunk = base + np.arange(self.k, dtype=float)
        return np.repeat(chunk[None, :, None], len(obs), 0).repeat(ACT_DIM, 2)


def test_temporal_ensembling_weights():
    ex = ChunkExecutor(FakePolicy(), "ensemble", m=0.5)
    obs = np.zeros((2, OBS_DIM))
    ex(obs)
    ex(obs)
    a = ex(obs)  # step 2: chunk0[2] = 2, chunk1[1] = 11, chunk2[0] = 20; oldest weighted most
    w = np.exp(-0.5 * np.arange(3))
    assert np.allclose(a, (w @ [2.0, 11.0, 20.0]) / w.sum())


def test_open_loop_and_latest_modes():
    fake = FakePolicy()
    ex = ChunkExecutor(fake, "open_loop")
    acts = [ex(np.zeros((1, OBS_DIM)))[0, 0] for _ in range(6)]
    assert acts == [0, 1, 2, 10, 11, 12] and fake.calls == 2
    ex = ChunkExecutor(FakePolicy(), "latest")
    assert [ex(np.zeros((1, OBS_DIM)))[0, 0] for _ in range(3)] == [0, 10, 20]


@pytest.mark.parametrize("kind,k", [("bc", 1), ("act", 4)])
def test_training_smoke(kind, k, tmp_path):
    demos = [run_expert(s, noise=0.01) for s in range(3)]
    policy = Policy(kind, k=k, hidden=64, d_model=32)
    info = train(policy, demos, steps=300, batch=64, lr=1e-3, warmup=20, dev=torch.device("cpu"))
    assert info["loss_curve"][-1] < 0.5 * info["loss_curve"][0]
    path = tmp_path / "p.pt"
    save_policy(policy, path)
    back = load_policy(path)
    obs = demos[0].obs[:4]
    np.testing.assert_allclose(back.predict(obs), policy.predict(obs), atol=1e-6)
    res = rollout(back, [100_000, 100_001], max_steps=5)
    assert res["success"].shape == (2,)


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi and 0.8 < lo and hi < 0.96
    assert np.isclose(wilson(0, 100)[0], 0.0) and np.isclose(wilson(100, 100)[1], 1.0)
