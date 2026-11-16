"""Kinematics wrapper and joint-space trajectory planner for LeRobot arms."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner, interpolate

from .config import ArmSpec


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


class KinematicsSolver(Protocol):
    """Arm kinematics in radians and the arm's local base frame."""

    def forward_kinematics(self, q_rad: Sequence[float]) -> np.ndarray:
        """Return a 4x4 homogeneous transform of the end-effector in the arm base frame."""
        ...

    def inverse_kinematics(
        self,
        q0_rad: Sequence[float],
        target_matrix: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> Sequence[float]:
        """Return joint angles in radians that reach ``target_matrix`` from ``q0_rad``."""
        ...


class PlacoKinematics:
    """Adapts ``lerobot.model.kinematics.RobotKinematics`` to radian ``KinematicsSolver``."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Sequence[str] | None = None,
        max_iters: int = 64,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        try:
            from lerobot.model.kinematics import RobotKinematics

            names = list(joint_names) if joint_names is not None else None
            self._kinematics = RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=target_frame_name,
                joint_names=names,
            )
        except ImportError as exc:
            raise ModuleNotFoundError(
                "lerobot[kinematics] (including placo) is required for URDF kinematics; "
                "install a2a-robots-lerobot[kinematics] on Python >=3.12 (for example: "
                "`uv sync --directory packages/a2a-robots-lerobot "
                "--python 3.12 --extra kinematics`)"
            ) from exc
        self._lock = threading.Lock()
        self._max_iters = max_iters

    def forward_kinematics(self, q_rad: Sequence[float]) -> np.ndarray:
        q_deg = np.rad2deg(np.asarray(q_rad, dtype=np.float64))
        with self._lock:
            return np.asarray(self._kinematics.forward_kinematics(q_deg), dtype=np.float64)

    def inverse_kinematics(
        self,
        q0_rad: Sequence[float],
        target_matrix: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> list[float]:
        q0_deg = np.rad2deg(np.asarray(q0_rad, dtype=np.float64))
        target = np.asarray(target_matrix, dtype=np.float64)
        with self._lock:
            sol_deg = self._kinematics.inverse_kinematics(
                q0_deg,
                target,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                max_iters=self._max_iters,
            )
        return [float(v) for v in np.deg2rad(np.asarray(sol_deg, dtype=np.float64))]


class LeRobotKinematicsPlanner(Planner):
    """Joint-space planner backed by ``KinematicsSolver`` (URDF/Placo or custom)."""

    def __init__(
        self,
        arms: Mapping[str, ArmSpec],
        *,
        solvers: Mapping[str, KinematicsSolver] | None = None,
        frame: str = "world",
        max_speed_rad_s: float = 0.25,
        max_joint_delta_rad: float = 1.0,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = 0.05,
        ik_max_iters: int = 64,
    ) -> None:
        if not arms:
            raise ValueError("at least one arm spec must be provided")
        if max_speed_rad_s <= 0.0:
            raise ValueError("max_speed_rad_s must be positive")
        if max_joint_delta_rad <= 0.0:
            raise ValueError("max_joint_delta_rad must be positive")
        if position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if orientation_tolerance_rad <= 0.0:
            raise ValueError("orientation_tolerance_rad must be positive")

        self._specs = dict(arms)
        self._frame = frame
        self._max_speed_rad_s = max_speed_rad_s
        self._max_joint_delta_rad = max_joint_delta_rad
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad

        provided = dict(solvers or {})
        self._solvers: dict[str, KinematicsSolver] = {}
        self._base_transforms: dict[str, np.ndarray] = {}
        self._inv_base_transforms: dict[str, np.ndarray] = {}
        for hand, spec in self._specs.items():
            base_tf = spec.resolved_base_transform()
            self._base_transforms[hand] = base_tf
            self._inv_base_transforms[hand] = np.linalg.inv(base_tf)
            if hand in provided:
                self._solvers[hand] = provided[hand]
            elif spec.urdf_path is not None:
                self._solvers[hand] = PlacoKinematics(
                    spec.urdf_path,
                    target_frame_name=spec.target_frame_name,
                    joint_names=spec.arm_joints,
                    max_iters=ik_max_iters,
                )
            else:
                raise ValueError(
                    f"hand {hand!r} needs either ArmSpec.urdf_path or an entry in solvers"
                )

    def forward_pose(self, hand: str, q_rad: Sequence[float], frame: str | None = None) -> Pose:
        """Compute the world-frame ``Pose`` for ``hand`` at radian joint angles ``q_rad``."""
        solver = self._solvers[hand]
        local_matrix = np.asarray(solver.forward_kinematics(q_rad), dtype=np.float64)
        world_matrix = self._base_transforms[hand] @ local_matrix
        return matrix_to_pose(world_matrix, frame or self._frame)

    def plan(
        self,
        hand: str,
        start: Mapping[str, float],
        target: Pose,
        dt: float,
    ) -> JointTrajectory:
        spec = self._specs.get(hand)
        if spec is None:
            raise PlanError(FailureCode.UNREACHABLE, f"unknown hand {hand!r}")

        solver = self._solvers[hand]
        joints = spec.arm_joints
        q0 = [float(start[name]) for name in joints]
        target_world = pose_to_matrix(target)
        target_local = self._inv_base_transforms[hand] @ target_world

        try:
            solution = solver.inverse_kinematics(
                q0,
                target_local,
                position_weight=spec.position_weight,
                orientation_weight=spec.orientation_weight,
            )
            goal = [float(value) for value in solution]
            if len(goal) != len(joints):
                raise PlanError(
                    FailureCode.UNREACHABLE, "inverse kinematics returned wrong joint count"
                )
            reached = self.forward_pose(hand, goal, target.frame or self._frame)
        except PlanError:
            raise
        except (ValueError, RuntimeError) as exc:
            raise PlanError(
                FailureCode.UNREACHABLE, "no joint solution reaches the target"
            ) from exc

        pos_err, ori_err = pose_error(reached, target)
        if pos_err > self._position_tolerance_m or (
            spec.check_orientation and ori_err > self._orientation_tolerance_rad
        ):
            raise PlanError(FailureCode.UNREACHABLE, "no joint solution reaches the target")

        for start_q, goal_q in zip(q0, goal, strict=True):
            if abs(goal_q - start_q) > self._max_joint_delta_rad:
                raise PlanError(
                    FailureCode.UNREACHABLE,
                    "joint solution requires too large a joint step",
                )

        speeds = [self._max_speed_rad_s] * len(joints)
        points = interpolate(q0, goal, speeds, dt)
        return JointTrajectory(joints, tuple(points), dt)
