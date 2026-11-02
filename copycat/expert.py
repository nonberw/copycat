"""Scripted expert: a state machine on privileged state (true cube pose) plus ARMageddon IK.

It is a feedback controller, not a replayed trajectory: every tick it reads the measured joint
angles, takes a bounded Cartesian step toward the current phase's goal and solves IK from there.
That is what lets it recover from the DART perturbations injected during data collection.
"""

from __future__ import annotations

import numpy as np
from armageddon import forward, solve
from armageddon.kinematics import rotation, rotvec_from_matrix

from .env import CHAIN, CONTROL_HZ, CUBE_HALF, GRIPPER_CLOSED, GRIPPER_OPEN, PickPlaceEnv, gripper_down

APPROACH_HEIGHT = 0.09  # tool point above the cube center before descending
CARRY_HEIGHT = 0.12     # tool point height while carrying
GRASP_DEPTH = 0.004     # go a hair below the cube center so the pads cover more of the faces
PLACE_Z = CUBE_HALF - GRASP_DEPTH + 0.008  # release with the cube 8 mm above the pad
MAX_SPEED = 0.30        # m/s
MAX_ANG_SPEED = 2.0     # rad/s
GAIN = 0.6              # fraction of the remaining error covered per tick (before the speed cap)
LEAD = 1.5              # command a bit past the step so the position servos don't lag behind

PHASES = ("approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat")


def grasp_yaw(cube_xy: np.ndarray, yaw: float) -> float:
    """Cube yaw is only defined mod 90 deg; pick the equivalent closest to the radial direction,
    which keeps the wrist roll small."""
    natural = np.arctan2(cube_xy[1], cube_xy[0])
    options = yaw + np.pi / 2 * np.arange(-3, 4)
    return float(options[np.argmin(np.abs(options - natural))])


class ScriptedExpert:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.timer = 0
        self.yaw = None  # grasp yaw, locked once the fingers close

    def _goto(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, float]:
        """Bounded step from the measured pose toward (pos, rot). Returns joint targets and the distance left."""
        pose = forward(CHAIN, q)
        dt = 1 / CONTROL_HZ
        err = pos - pose.pos
        dist = float(np.linalg.norm(err))
        step = GAIN * err
        n = np.linalg.norm(step)
        if n > MAX_SPEED * dt:
            step *= MAX_SPEED * dt / n
        w = rotvec_from_matrix(rot @ pose.rot.T)
        ang = np.linalg.norm(w)
        w_step = GAIN * w
        if np.linalg.norm(w_step) > MAX_ANG_SPEED * dt:
            w_step = w / ang * MAX_ANG_SPEED * dt
        target_rot = rotation(w_step / max(np.linalg.norm(w_step), 1e-12), np.linalg.norm(w_step)) @ pose.rot
        res = solve(CHAIN, pose.pos + LEAD * step, target_rot, q0=q, max_iters=30)
        return res.q, dist + 0.05 * float(ang)

    def act(self, env: PickPlaceEnv) -> np.ndarray:
        q = env.data.qpos[:6].copy()
        cube, pad = env.cube_pos, env.pad_xy
        ee = forward(CHAIN, q).pos
        grip_w = env.grip
        holding = grip_w > 0.01 and np.linalg.norm(cube - ee) < 0.035 and cube[2] > CUBE_HALF - 0.005
        self.timer += 1

        # Lost the cube while carrying it: open up and start over.
        if self.phase in ("lift", "carry", "lower") and not holding:
            self._set("approach")
        # Cube landed off the pad (or got knocked off): go get it again.
        if self.phase == "retreat" and not env.success() and grip_w > 0.03:
            on_table = cube[2] < CUBE_HALF + 0.004
            if on_table and np.linalg.norm(cube[:2] - pad) > 0.03:
                self._set("approach")

        yaw = self.yaw if self.yaw is not None else grasp_yaw(cube[:2], env.cube_yaw)
        R = gripper_down(yaw)
        grip = GRIPPER_OPEN

        if self.phase == "approach":
            self.yaw = None
            yaw = grasp_yaw(cube[:2], env.cube_yaw)
            R = gripper_down(yaw)
            goal = cube + [0, 0, APPROACH_HEIGHT]
            # Coming in low and off to the side: rise first so the fingers don't plow into the cube.
            if np.linalg.norm(ee[:2] - cube[:2]) > 0.02 and ee[2] < cube[2] + 0.05:
                goal = np.array([ee[0], ee[1], cube[2] + APPROACH_HEIGHT])
            qd, _ = self._goto(q, goal, R)
            xy_err = np.linalg.norm(ee[:2] - cube[:2])
            if xy_err < 0.015 and abs(ee[2] - goal[2]) < 0.025 and grip_w > 0.035:
                self._set("descend")
        elif self.phase == "descend":
            goal = cube - [0, 0, GRASP_DEPTH]
            qd, _ = self._goto(q, goal, R)
            xy_err = np.linalg.norm(ee[:2] - cube[:2])
            if xy_err > 0.03:
                self._set("approach")
            elif xy_err < 0.012 and ee[2] - goal[2] < 0.01:
                self._set("grasp")
        elif self.phase == "grasp":
            if self.yaw is None:
                self.yaw = yaw
            grip = GRIPPER_CLOSED
            qd, _ = self._goto(q, cube - [0, 0, GRASP_DEPTH], R)
            if self.timer >= 8:
                self._set("lift" if holding else "approach")
        elif self.phase == "lift":
            grip = GRIPPER_CLOSED
            qd, _ = self._goto(q, np.array([ee[0], ee[1], CARRY_HEIGHT]), R)
            if ee[2] > CARRY_HEIGHT - 0.03:
                self._set("carry")
        elif self.phase == "carry":
            grip = GRIPPER_CLOSED
            qd, _ = self._goto(q, np.array([pad[0], pad[1], CARRY_HEIGHT]), R)
            if np.linalg.norm(ee[:2] - pad) < 0.015:
                self._set("lower")
        elif self.phase == "lower":
            grip = GRIPPER_CLOSED
            goal = np.array([pad[0], pad[1], PLACE_Z])
            qd, _ = self._goto(q, goal, R)
            xy_err = np.linalg.norm(ee[:2] - pad)
            if xy_err > 0.035:
                self._set("carry")
            elif xy_err < 0.015 and ee[2] - goal[2] < 0.008:
                self._set("release")
        elif self.phase == "release":
            qd, _ = self._goto(q, np.array([pad[0], pad[1], PLACE_Z]), R)
            if self.timer >= 6:
                self._set("retreat")
        else:  # retreat
            qd, _ = self._goto(q, np.array([pad[0], pad[1], CARRY_HEIGHT]), R)
        return np.concatenate([qd, [grip]])

    def _set(self, phase: str):
        self.phase, self.timer = phase, 0
