import numpy as np

from copycat.data import run_expert
from copycat.env import PickPlaceEnv
from copycat.evaluate import EVAL_SEED0
from copycat.expert import grasp_yaw


def test_expert_success_rate_is_high():
    env = PickPlaceEnv()
    eps = [run_expert(EVAL_SEED0 + s, env=env) for s in range(30)]
    assert np.mean([e.success for e in eps]) >= 0.95


def test_expert_recovers_from_dart_noise():
    env = PickPlaceEnv()
    eps = [run_expert(EVAL_SEED0 + s, noise=0.02, env=env) for s in range(15)]
    assert np.mean([e.success for e in eps]) >= 0.9
    # Labels stay clean: the gripper command is still exactly open or closed.
    grip = np.concatenate([e.act[:, 6] for e in eps])
    assert set(np.unique(grip)) <= {0.0, np.float32(0.04)}


def test_grasp_yaw_picks_the_equivalent_near_radial():
    xy = np.array([0.4, 0.2])
    radial = np.arctan2(xy[1], xy[0])
    for yaw in np.linspace(-3, 3, 13):
        g = grasp_yaw(xy, yaw)
        assert abs(g - radial) <= np.pi / 4 + 1e-9
        assert np.isclose(np.cos(4 * g), np.cos(4 * yaw))
