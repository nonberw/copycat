"""Pick-and-place task: ARMageddon arm with a parallel gripper, one cube, one target pad.

Each episode randomizes the cube pose and the pad position. The policy runs at 20 Hz and
commands absolute joint position targets plus a gripper width; physics runs at 500 Hz.
"""

from __future__ import annotations

import mujoco
import numpy as np
from armageddon import HOME, default_arm, solve
from armageddon.kinematics import rotation
from armageddon.mjcf import ARM_MATERIALS, FINGER_TRAVEL, arm_actuators, arm_xml

CONTROL_HZ = 20
MAX_STEPS = 200  # 10 s per episode
CUBE_HALF = 0.02
PAD_RADIUS = 0.045
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0  # commanding past the cube width is what produces the squeeze
WORKSPACE_LOW = np.array([0.28, -0.22])
WORKSPACE_HIGH = np.array([0.50, 0.22])
MIN_CUBE_PAD_DIST = 0.15
MAX_REL_YAW = np.deg2rad(40)  # cube yaw relative to the radial direction, see README
OBS_DIM = 20
ACT_DIM = 7
OBS_NAMES = (
    [f"q{i}" for i in range(6)] + [f"qd{i}" for i in range(6)] + ["grip"]
    + ["cube_x", "cube_y", "cube_z", "cube_sin4yaw", "cube_cos4yaw", "pad_x", "pad_y"]
)


def gripper_down(yaw: float) -> np.ndarray:
    """Tool pointing straight down, fingers closing along the horizontal direction yaw + 90 deg."""
    return rotation(np.array([0.0, 0.0, 1.0]), yaw) @ rotation(np.array([0.0, 1.0, 0.0]), np.pi)


CHAIN = default_arm()
# Start above the middle of the workspace, tool down: every episode begins from here (plus jitter).
START_Q = solve(CHAIN, np.array([0.36, 0.0, 0.25]), gripper_down(0.0), q0=HOME, restarts=10,
                q_rest=HOME, nullspace_gain=0.05, rng=np.random.default_rng(0)).q


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def build_xml() -> str:
    return f"""<mujoco model="copycat">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.09 0.32" rgb2="0.01 0.01 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.12 0.2" rgb2="0.08 0.08 0.13" width="512" height="512" mark="edge" markrgb="0.45 0.3 0.9"/>
    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.1"/>
{ARM_MATERIALS}
  </asset>

  <default>
    <joint damping="3" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0.6 -0.6 2.2" dir="-0.3 0.3 -1" diffuse="0.9 0.85 1" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor" contype="0" conaffinity="1" condim="3" friction="1 0.02 0.002"/>
    <geom name="mat" type="box" pos="0.39 0 0.0005" size="0.15 0.27 0.0005" rgba="0.32 0.31 0.36 1" contype="0" conaffinity="0"/>
    <camera name="front" pos="1.0 0 0.75" xyaxes="0 1 0 -0.6 0 0.8" fovy="45"/>
{arm_xml(CHAIN, gripper="parallel")}
    <body name="pad" mocap="true" pos="0.4 0.15 0.0012">
      <geom type="cylinder" size="{PAD_RADIUS} 0.0012" rgba="0.15 0.9 0.95 0.55" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="{PAD_RADIUS * 0.45} 0.0014" rgba="0.15 0.9 0.95 0.9" contype="0" conaffinity="0"/>
    </body>
    <body name="cube" pos="0.4 -0.1 {CUBE_HALF}">
      <freejoint/>
      <geom type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" rgba="1 0.62 0.1 1" mass="0.05"
            contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" solref="0.005 1"/>
    </body>
  </worldbody>

  <actuator>
{arm_actuators(CHAIN, gripper="parallel")}
  </actuator>
</mujoco>
"""


def cube_yaw(xmat: np.ndarray) -> float:
    """Yaw of the cube's x axis. A cube looks the same every 90 deg, so only yaw mod pi/2 matters."""
    R = xmat.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


class PickPlaceEnv:
    """Minimal gym-like env: reset(seed) -> obs, step(action) -> (obs, success)."""

    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model or mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.substeps = round(1 / (CONTROL_HZ * self.model.opt.timestep))
        self._cube_adr = self.model.jnt_qposadr[self.model.body("cube").jntadr[0]]
        self._pad_mocap = self.model.body("pad").mocapid[0]
        self.lower, self.upper = CHAIN.lower, CHAIN.upper
        self.t = 0

    # ------------------------------------------------------------------ api --

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        while True:
            cube_xy = rng.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)
            pad_xy = rng.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)
            if np.linalg.norm(cube_xy - pad_xy) > MIN_CUBE_PAD_DIST:
                break
        yaw = np.arctan2(cube_xy[1], cube_xy[0]) + rng.uniform(-MAX_REL_YAW, MAX_REL_YAW)
        q0 = CHAIN.clip(START_Q + rng.normal(0, 0.03, 6))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = q0
        self.data.qpos[6:8] = GRIPPER_OPEN
        a = self._cube_adr
        self.data.qpos[a:a + 3] = (*cube_xy, CUBE_HALF)
        self.data.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        self.data.mocap_pos[self._pad_mocap] = (*pad_xy, 0.0012)
        self.data.ctrl[:6] = q0
        self.data.ctrl[6:8] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self.obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        action = np.asarray(action, dtype=float)
        self.data.ctrl[:6] = np.clip(action[:6], self.lower, self.upper)
        self.data.ctrl[6:8] = np.clip(action[6], 0.0, FINGER_TRAVEL)
        for _ in range(self.substeps):
            # Feed forward gravity/coriolis for the arm joints only (not the fingers or the cube).
            self.data.qfrc_applied[:6] = self.data.qfrc_bias[:6]
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self.obs(), self.success()

    def obs(self) -> np.ndarray:
        d = self.data
        cube = d.body("cube")
        yaw = cube_yaw(cube.xmat)
        return np.concatenate([
            d.qpos[:6], d.qvel[:6], [d.qpos[6:8].mean()],
            cube.xpos, [np.sin(4 * yaw), np.cos(4 * yaw)],
            d.mocap_pos[self._pad_mocap][:2],
        ]).astype(np.float32)

    # -------------------------------------------------------------- queries --

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.body("cube").xpos.copy()

    @property
    def cube_yaw(self) -> float:
        return cube_yaw(self.data.body("cube").xmat)

    @property
    def pad_xy(self) -> np.ndarray:
        return self.data.mocap_pos[self._pad_mocap][:2].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site("ee").xpos.copy()

    @property
    def grip(self) -> float:
        """Half-gap between the finger pads in meters."""
        return float(self.data.qpos[6:8].mean())

    def success(self) -> bool:
        """Cube resting on the pad and the gripper has let go of it."""
        p = self.cube_pos
        on_pad = np.linalg.norm(p[:2] - self.pad_xy) < PAD_RADIUS
        resting = p[2] < CUBE_HALF + 0.004
        return bool(on_pad and resting and self.grip > 0.03)
