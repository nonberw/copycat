import numpy as np

from copycat.data import chunk_targets, load_demos, run_expert, save_demos
from copycat.env import ACT_DIM, OBS_DIM


def test_episode_shapes():
    ep = run_expert(0)
    assert ep.success
    assert ep.obs.shape == (len(ep.obs), OBS_DIM) and ep.act.shape == (len(ep.obs), ACT_DIM)
    assert ep.obs.dtype == np.float32 and ep.act.dtype == np.float32


def test_save_load_roundtrip(tmp_path):
    demos = [run_expert(s, noise=0.01) for s in range(3)]
    path = tmp_path / "d.npz"
    save_demos(path, demos)
    back = load_demos(path)
    assert [d.seed for d in back] == [0, 1, 2]
    for a, b in zip(demos, back):
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.act, b.act)
    assert len(load_demos(path, 2)) == 2


def test_chunk_targets_shift_and_pad():
    demos = [run_expert(s) for s in range(2)]
    k = 5
    obs, Y = chunk_targets(demos, k)
    T0 = len(demos[0].act)
    assert obs.shape == (sum(len(d.act) for d in demos), OBS_DIM)
    assert Y.shape == (len(obs), k, ACT_DIM)
    np.testing.assert_array_equal(Y[3, 2], demos[0].act[5])        # a_{t+j}
    np.testing.assert_array_equal(Y[T0 - 1, 4], demos[0].act[-1])  # padded with the last action
    np.testing.assert_array_equal(Y[T0, 0], demos[1].act[0])       # never leaks across episodes
