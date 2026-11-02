import numpy as np

from copycat.env import (ACT_DIM, CUBE_HALF, MIN_CUBE_PAD_DIST, OBS_DIM, OBS_NAMES, START_Q, PickPlaceEnv,
                         cube_yaw)


def test_obs_layout():
    env = PickPlaceEnv()
    obs = env.reset(0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert len(OBS_NAMES) == OBS_DIM
    np.testing.assert_allclose(obs[13:16], env.cube_pos, atol=1e-6)
    np.testing.assert_allclose(obs[18:20], env.pad_xy, atol=1e-6)


def test_reset_is_deterministic_per_seed():
    a, b = PickPlaceEnv(), PickPlaceEnv()
    np.testing.assert_array_equal(a.reset(7), b.reset(7))
    assert not np.allclose(a.reset(7), a.reset(8))


def test_reset_layout_is_valid():
    env = PickPlaceEnv()
    for seed in range(50):
        env.reset(seed)
        assert np.linalg.norm(env.cube_pos[:2] - env.pad_xy) > MIN_CUBE_PAD_DIST
        assert abs(env.cube_pos[2] - CUBE_HALF) < 1e-6
        assert not env.success()


def test_step_is_deterministic():
    rng = np.random.default_rng(0)
    actions = [np.concatenate([START_Q + rng.normal(0, 0.05, 6), [0.04]]) for _ in range(15)]
    runs = []
    for _ in range(2):
        env = PickPlaceEnv()
        env.reset(3)
        runs.append(np.stack([env.step(a)[0] for a in actions]))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_holding_still_keeps_the_arm_still():
    """Gravity compensation + position servos: commanding the current pose shouldn't sag."""
    env = PickPlaceEnv()
    obs = env.reset(1)
    a = np.concatenate([obs[:6], [0.04]])
    for _ in range(20):
        obs, _ = env.step(a)
    assert np.abs(obs[:6] - a[:6]).max() < 0.01
    assert ACT_DIM == 7


def test_cube_yaw_feature_is_symmetric():
    """sin(4 yaw), cos(4 yaw) can't tell a cube from itself rotated by 90 deg, and neither can a camera."""
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    y1, y2 = cube_yaw(rz(0.3)), cube_yaw(rz(0.3 + np.pi / 2))
    assert np.isclose(np.sin(4 * y1), np.sin(4 * y2)) and np.isclose(np.cos(4 * y1), np.cos(4 * y2))
