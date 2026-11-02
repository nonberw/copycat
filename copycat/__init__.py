"""COPYCAT: imitation learning for a robot arm, from scratch.

Torch-free on purpose: the env and the expert import fast (data collection spawns worker
processes). Policies live in `copycat.policies`.
"""

from .env import PickPlaceEnv
from .expert import ScriptedExpert

__all__ = ["PickPlaceEnv", "ScriptedExpert"]
