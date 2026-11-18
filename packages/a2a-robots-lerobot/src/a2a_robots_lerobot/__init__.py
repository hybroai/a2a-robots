"""Hugging Face LeRobot backend for a2a-robots."""

__version__ = "0.1.1"

from a2a_robots_lerobot.backend import LeRobotArmBackend
from a2a_robots_lerobot.config import (
    KOCH_ARM_JOINTS,
    SO_ARM_JOINTS,
    ArmSpec,
    bi_so_arms,
    koch_arms,
    so101_arms,
    so101_hand_spec,
)
from a2a_robots_lerobot.fake import AnalyticArmKinematics, FakeLeRobot
from a2a_robots_lerobot.planner import (
    KinematicsSolver,
    LeRobotKinematicsPlanner,
    PlacoKinematics,
    matrix_to_pose,
    pose_error,
    pose_to_matrix,
)

__all__ = [
    "AnalyticArmKinematics",
    "ArmSpec",
    "FakeLeRobot",
    "KOCH_ARM_JOINTS",
    "KinematicsSolver",
    "LeRobotArmBackend",
    "LeRobotKinematicsPlanner",
    "PlacoKinematics",
    "SO_ARM_JOINTS",
    "__version__",
    "bi_so_arms",
    "koch_arms",
    "matrix_to_pose",
    "pose_error",
    "pose_to_matrix",
    "so101_arms",
    "so101_hand_spec",
]
