"""Offline fake LeRobot ``Robot`` and kinematics doubles for tests and examples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .config import SO_ARM_JOINTS, ArmSpec, bi_so_arms, so101_arms
from .planner import KinematicsSolver, matrix_to_pose, pose_error

_DEFAULT_Q_RAD: tuple[float, ...] = (0.0, -0.35, 0.70, 0.20, 0.0)


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _trans(x: float, y: float, z: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = [x, y, z]
    return mat


def _orientation_error_vector(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    r_err = target @ current.T
    skew = 0.5 * np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )
    sin_norm = float(np.linalg.norm(skew))
    cos_val = max(-1.0, min(1.0, 0.5 * (float(np.trace(r_err)) - 1.0)))
    if sin_norm < 1e-9:
        if cos_val >= 0.0:
            return np.zeros(3, dtype=np.float64)
        diag = np.maximum(0.0, np.diag(r_err) + 1.0)
        return math.pi * np.sqrt(diag / 2.0)
    angle = math.atan2(sin_norm, cos_val)
    return skew * (angle / sin_norm)


class AnalyticArmKinematics(KinematicsSolver):
    """5-DoF or 6-DoF analytic arm kinematics in radians for offline tests."""

    def __init__(self, *, dof: int = 5) -> None:
        if dof not in (5, 6):
            raise ValueError("dof must be 5 or 6")
        self.dof = dof

    def forward_kinematics(self, q_rad: Sequence[float]) -> np.ndarray:
        if len(q_rad) != self.dof:
            raise ValueError(f"expected {self.dof} joint angles, got {len(q_rad)}")
        q = [float(v) for v in q_rad]
        mat = _trans(0.0, 0.0, 0.10)
        mat = mat @ _rot_z(q[0])  # shoulder_pan
        mat = mat @ _rot_y(q[1])  # shoulder_lift
        mat = mat @ _trans(0.18, 0.0, 0.0)
        mat = mat @ _rot_y(q[2])  # elbow_flex
        mat = mat @ _trans(0.16, 0.0, 0.0)
        mat = mat @ _rot_y(q[3])  # wrist_flex
        mat = mat @ _rot_x(q[4])  # wrist_roll
        if self.dof == 6:
            mat = mat @ _rot_z(q[5])  # optional 6th yaw joint
        mat = mat @ _trans(0.06, 0.0, 0.0)
        return mat

    def inverse_kinematics(
        self,
        q0_rad: Sequence[float],
        target_matrix: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> list[float]:
        target_mat = np.asarray(target_matrix, dtype=np.float64)
        if target_mat.shape != (4, 4):
            raise ValueError(f"expected 4x4 target matrix, got {target_mat.shape}")
        if len(q0_rad) != self.dof:
            raise ValueError(f"expected {self.dof} joint angles, got {len(q0_rad)}")

        q = np.array([float(v) for v in q0_rad], dtype=np.float64)
        eps = 1e-6
        damping = 1e-3
        weights = np.array(
            [position_weight] * 3 + [orientation_weight] * 3,
            dtype=np.float64,
        )

        for _ in range(100):
            cur = self.forward_kinematics(q)
            pos_err_vec = target_mat[:3, 3] - cur[:3, 3]
            ori_err_vec = _orientation_error_vector(cur[:3, :3], target_mat[:3, :3])
            err = np.concatenate([pos_err_vec, ori_err_vec]) * weights
            if float(np.linalg.norm(pos_err_vec)) < 1e-5 and (
                orientation_weight == 0.0 or float(np.linalg.norm(ori_err_vec)) < 1e-4
            ):
                break

            jac = np.zeros((6, self.dof), dtype=np.float64)
            for idx in range(self.dof):
                q_pert = q.copy()
                q_pert[idx] += eps
                pert = self.forward_kinematics(q_pert)
                jac[:3, idx] = ((pert[:3, 3] - cur[:3, 3]) / eps) * position_weight
                jac[3:, idx] = (
                    _orientation_error_vector(cur[:3, :3], pert[:3, :3]) / eps
                ) * orientation_weight

            lhs = jac @ jac.T + (damping * damping) * np.eye(6)
            dq = jac.T @ np.linalg.solve(lhs, err)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > 0.25:
                dq *= 0.25 / step_norm
            q = np.clip(q + dq, -math.pi, math.pi)

        reached = matrix_to_pose(self.forward_kinematics(q), "world")
        target_pose = matrix_to_pose(target_mat, "world")
        pos_err, ori_err = pose_error(reached, target_pose)
        if pos_err > 1e-3 or (self.dof == 6 and orientation_weight >= 0.5 and ori_err > 1e-2):
            raise ValueError("no joint solution reaches the target")
        return [float(v) for v in q]


@dataclass
class FakeGraspableObject:
    """An object in the fake scene that a LeRobot gripper can stall against."""

    name: str
    position: tuple[float, float, float]
    opening_pct: float = 35.0
    slip: bool = False
    capture_radius_m: float = 0.05
    held_by: str | None = None
    slip_after_steps: int = 7
    stall_steps: int = field(default=0, init=False)


class FakeLeRobot:
    """In-memory stand-in matching the ``lerobot.robots.Robot`` contract."""

    supports_command_hold: bool = False
    supports_position_hold: bool = True

    def __init__(
        self,
        *,
        bimanual: bool = False,
        use_degrees: bool = True,
        connected: bool = True,
        calibrated: bool = True,
        with_camera: bool = False,
        dof: int = 5,
    ) -> None:
        if dof not in (5, 6):
            raise ValueError("dof must be 5 or 6")
        self._bimanual = bimanual
        self._dof = dof
        self._connected = connected
        self._calibrated = calibrated
        self._with_camera = with_camera
        self.calibrate_calls = 0
        self.connect_calls: list[bool] = []
        self.sent_actions: list[dict[str, float]] = []
        self.objects: dict[str, FakeGraspableObject] = {}
        self.kinematics: dict[str, AnalyticArmKinematics] = {}
        self.arms: dict[str, ArmSpec] = {}

        if bimanual:
            base_arms = bi_so_arms()
        else:
            base_arms = so101_arms()

        if dof == 6:
            joints6 = (*SO_ARM_JOINTS, "wrist_yaw")
            new_arms: dict[str, ArmSpec] = {}
            for hand, spec in base_arms.items():
                prefix = "left_" if bimanual and hand == "left" else "right_" if bimanual else ""
                new_arms[hand] = ArmSpec(
                    arm_joints=joints6,
                    action_keys=tuple(f"{prefix}{j}.pos" for j in joints6),
                    observation_keys=tuple(f"{prefix}{j}.pos" for j in joints6),
                    gripper_action_key=spec.gripper_action_key,
                    gripper_observation_key=spec.gripper_observation_key,
                    joint_unit="deg" if use_degrees else "rad",
                    base_transform=spec.base_transform,
                    orientation_weight=1.0,
                    check_orientation=True,
                )
            self.arms = new_arms
        else:
            if use_degrees:
                self.arms = base_arms
            else:
                self.arms = {
                    hand: ArmSpec(
                        arm_joints=spec.arm_joints,
                        action_keys=spec.action_keys,
                        observation_keys=spec.observation_keys,
                        gripper_action_key=spec.gripper_action_key,
                        gripper_observation_key=spec.gripper_observation_key,
                        joint_unit="rad",
                        base_transform=spec.base_transform,
                        check_orientation=spec.check_orientation,
                    )
                    for hand, spec in base_arms.items()
                }

        for hand in self.arms:
            self.kinematics[hand] = AnalyticArmKinematics(dof=dof)

        if bimanual:
            self.config = SimpleNamespace(
                left_arm_config=SimpleNamespace(use_degrees=use_degrees),
                right_arm_config=SimpleNamespace(use_degrees=use_degrees),
            )
        else:
            self.config = SimpleNamespace(use_degrees=use_degrees)

        q0_rad = _DEFAULT_Q_RAD if dof == 5 else (*_DEFAULT_Q_RAD, 0.0)
        self._state: dict[str, float] = {}
        self.goal_positions: dict[str, float] = {}
        for spec in self.arms.values():
            for key, angle_rad in zip(spec.resolved_action_keys(), q0_rad, strict=True):
                val = math.degrees(angle_rad) if use_degrees else angle_rad
                self._state[key] = val
                self.goal_positions[key] = val
            if spec.gripper_action_key is not None:
                self._state[spec.gripper_action_key] = 50.0
                self.goal_positions[spec.gripper_action_key] = 50.0

    @property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = dict.fromkeys(self._state, float)
        if self._with_camera:
            features["top"] = (8, 8, 3)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._state, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.connect_calls.append(calibrate)
        self._connected = True
        if calibrate and not self._calibrated:
            self.calibrate()

    def calibrate(self) -> None:
        self.calibrate_calls += 1
        self._calibrated = True

    def configure(self) -> None:
        pass

    def _ee_world_pos(self, hand: str) -> np.ndarray:
        spec = self.arms[hand]
        raw = [self._state[key] for key in spec.resolved_observation_keys()]
        q_rad = [math.radians(v) for v in raw] if spec.joint_unit == "deg" else raw
        local = self.kinematics[hand].forward_kinematics(q_rad)
        world = spec.resolved_base_transform() @ local
        return world[:3, 3]

    def place_object(
        self,
        name: str,
        position: tuple[float, float, float],
        *,
        opening_pct: float = 35.0,
        slip: bool = False,
        capture_radius_m: float = 0.05,
    ) -> FakeGraspableObject:
        obj = FakeGraspableObject(
            name=name,
            position=position,
            opening_pct=opening_pct,
            slip=slip,
            capture_radius_m=capture_radius_m,
        )
        self.objects[name] = obj
        return obj

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("FakeLeRobot is not connected")
        obs: dict[str, Any] = dict(self._state)
        if self._with_camera:
            obs["top"] = np.zeros((8, 8, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise ConnectionError("FakeLeRobot is not connected")
        missing = set(self.action_features) - set(action)
        if missing:
            raise KeyError(f"send_action missing keys: {sorted(missing)}")

        sent = {key: float(action[key]) for key in self.action_features}
        self.sent_actions.append(dict(sent))
        self.goal_positions.update(sent)

        for hand, spec in self.arms.items():
            for key in spec.resolved_action_keys():
                self._state[key] = sent[key]

            ee_pos = self._ee_world_pos(hand)
            for obj in self.objects.values():
                if obj.held_by == hand:
                    obj.position = (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]))

            grip_key = spec.gripper_action_key
            if grip_key is None:
                continue
            target = min(100.0, max(0.0, sent[grip_key]))
            current = self._state[grip_key]
            closing = target < current
            if not closing:
                for obj in self.objects.values():
                    if obj.held_by == hand and target > obj.opening_pct:
                        obj.held_by = None
                self._state[grip_key] = target
                continue

            grasped = False
            for obj in self.objects.values():
                dist = float(np.linalg.norm(ee_pos - np.asarray(obj.position, dtype=np.float64)))
                if dist <= obj.capture_radius_m and target <= obj.opening_pct:
                    if obj.slip:
                        obj.held_by = None
                        if current <= obj.opening_pct + 0.5 and target < obj.opening_pct - 1.0:
                            obj.stall_steps += 1
                        if obj.stall_steps < obj.slip_after_steps:
                            self._state[grip_key] = obj.opening_pct
                        else:
                            self._state[grip_key] = target
                    else:
                        obj.held_by = hand
                        self._state[grip_key] = obj.opening_pct
                    grasped = True
                    break
            if not grasped:
                self._state[grip_key] = target

        return dict(self._state)

    def disconnect(self) -> None:
        self._connected = False
