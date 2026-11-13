"""Joint-space trajectory planner for Pollen Robotics Reachy 2 arms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner, interpolate

ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_yaw",
    "elbow_pitch",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


def hand_joints(hand: str) -> tuple[str, ...]:
    """Return the canonical ordered 7-DoF joint names for a Reachy 2 arm."""
    return tuple(f"{hand}_{joint}" for joint in ARM_JOINTS)


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert an ``a2a_robots.Pose`` into a 4x4 homogeneous transform."""
    w, x, y, z = (float(v) for v in pose.orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("quaternion cannot be zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[0, 1] = 2.0 * (xy - wz)
    mat[0, 2] = 2.0 * (xz + wy)
    mat[1, 0] = 2.0 * (xy + wz)
    mat[1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[1, 2] = 2.0 * (yz - wx)
    mat[2, 0] = 2.0 * (xz - wy)
    mat[2, 1] = 2.0 * (yz + wx)
    mat[2, 2] = 1.0 - 2.0 * (xx + yy)
    mat[:3, 3] = [float(v) for v in pose.position]
    return mat


def _rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a canonical unit quaternion (w, x, y, z)."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (float(rot[2, 1]) - float(rot[1, 2])) / s
        y = (float(rot[0, 2]) - float(rot[2, 0])) / s
        z = (float(rot[1, 0]) - float(rot[0, 1])) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + float(rot[0, 0]) - float(rot[1, 1]) - float(rot[2, 2])) * 2.0
        w = (float(rot[2, 1]) - float(rot[1, 2])) / s
        x = 0.25 * s
        y = (float(rot[0, 1]) + float(rot[1, 0])) / s
        z = (float(rot[0, 2]) + float(rot[2, 0])) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + float(rot[1, 1]) - float(rot[0, 0]) - float(rot[2, 2])) * 2.0
        w = (float(rot[0, 2]) - float(rot[2, 0])) / s
        x = (float(rot[0, 1]) + float(rot[1, 0])) / s
        y = 0.25 * s
        z = (float(rot[1, 2]) + float(rot[2, 1])) / s
    else:
        s = math.sqrt(1.0 + float(rot[2, 2]) - float(rot[0, 0]) - float(rot[1, 1])) * 2.0
        w = (float(rot[1, 0]) - float(rot[0, 1])) / s
        x = (float(rot[0, 2]) + float(rot[2, 0])) / s
        y = (float(rot[1, 2]) + float(rot[2, 1])) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (round(w, 6), round(x, 6), round(y, 6), round(z, 6))


def matrix_to_pose(matrix: np.ndarray, frame: str) -> Pose:
    """Convert a 4x4 homogeneous transform into an ``a2a_robots.Pose``."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform matrix, got shape {mat.shape}")
    pos = (round(float(mat[0, 3]), 6), round(float(mat[1, 3]), 6), round(float(mat[2, 3]), 6))
    quat = _rotation_matrix_to_quaternion(mat[:3, :3])
    return Pose(frame, pos, quat)


def pose_error(a: Pose, b: Pose) -> tuple[float, float]:
    """Return ``(position_error_m, orientation_error_rad)`` between two poses."""
    dx = float(a.position[0]) - float(b.position[0])
    dy = float(a.position[1]) - float(b.position[1])
    dz = float(a.position[2]) - float(b.position[2])
    pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)

    dot = sum(float(x) * float(y) for x, y in zip(a.orientation, b.orientation, strict=True))
    dot = min(1.0, abs(dot))
    ori_err = 2.0 * math.acos(dot)
    return pos_err, ori_err


class Reachy2Planner(Planner):
    """Joint-space planner backed by a Reachy 2 arm's kinematics model."""

    def __init__(
        self,
        arms: Mapping[str, Any],
        *,
        frame: str = "world",
        max_speed_rad_s: float = 0.25,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = 0.05,
    ) -> None:
        if max_speed_rad_s <= 0.0:
            raise ValueError("max_speed_rad_s must be positive")
        self._arms = dict(arms)
        self._frame = frame
        self._max_speed_rad_s = max_speed_rad_s
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad

    def plan(
        self,
        hand: str,
        start: Mapping[str, float],
        target: Pose,
        dt: float,
    ) -> JointTrajectory:
        arm = self._arms.get(hand)
        if arm is None:
            raise PlanError(FailureCode.UNREACHABLE, f"unknown hand {hand!r}")

        joints = hand_joints(hand)
        q0 = [float(start[name]) for name in joints]
        target_matrix = pose_to_matrix(target)

        try:
            solution = arm.inverse_kinematics(target_matrix, q0=q0, degrees=False)
        except (ValueError, RuntimeError) as exc:
            raise PlanError(
                FailureCode.UNREACHABLE, "no joint solution reaches the target"
            ) from exc

        goal = [float(value) for value in solution]
        if len(goal) != len(joints):
            raise PlanError(
                FailureCode.UNREACHABLE, "inverse kinematics returned wrong joint count"
            )

        # Always pass explicit 7-element joint angles in radians because
        # reachy2-sdk 1.0.15 has a bug when forward_kinematics() reads present state.
        fk_matrix = np.asarray(arm.forward_kinematics(goal, degrees=False), dtype=np.float64)
        reached = matrix_to_pose(fk_matrix, target.frame or self._frame)
        pos_err, ori_err = pose_error(reached, target)
        if pos_err > self._position_tolerance_m or ori_err > self._orientation_tolerance_rad:
            raise PlanError(FailureCode.UNREACHABLE, "no joint solution reaches the target")

        speeds = [self._max_speed_rad_s] * len(joints)
        points = interpolate(q0, goal, speeds, dt)
        return JointTrajectory(joints, tuple(points), dt)
