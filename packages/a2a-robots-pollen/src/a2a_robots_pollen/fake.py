"""Offline fake Reachy 2 SDK double for tests and local examples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from a2a_robots_pollen.planner import matrix_to_pose, pose_error

_UPPER_ARM_M = 0.28
_FOREARM_M = 0.28
_TOOL_M = 0.08
_DEFAULT_Q_RAD: tuple[float, ...] = (-0.4, 0.0, 0.0, -1.3, 0.0, 0.1, 0.0)


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
    """Return a 3D axis-angle vector rotating ``current`` to ``target``."""
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
    cos_val = 0.5 * (float(np.trace(r_err)) - 1.0)
    cos_val = max(-1.0, min(1.0, cos_val))
    if sin_norm < 1e-9:
        if cos_val >= 0.0:
            return np.zeros(3, dtype=np.float64)
        # 180-degree rotation fallback along largest diagonal axis
        diag = np.maximum(0.0, np.diag(r_err) + 1.0)
        axis = np.sqrt(diag / 2.0)
        return math.pi * axis
    angle = math.atan2(sin_norm, cos_val)
    return skew * (angle / sin_norm)


@dataclass
class FakeGraspableObject:
    """An object in the fake Reachy 2 scene that a gripper can stall against."""

    name: str
    position: tuple[float, float, float]
    opening_pct: float = 35.0
    slip: bool = False
    capture_radius_m: float = 0.05
    held_by: str | None = None
    # A slipping object holds the fingers for this many control steps, long
    # enough to count as contact but too short to count as a grasp.
    slip_after_steps: int = 7


class FakeOrbitaJoint:
    """In-memory stand-in for ``reachy2_sdk.orbita.orbita_joint.OrbitaJoint``."""

    def __init__(self, initial_deg: float = 0.0) -> None:
        self.present_position: float = float(initial_deg)
        self.goal_position: float = float(initial_deg)


class FakeShoulder:
    def __init__(self, pitch: FakeOrbitaJoint, roll: FakeOrbitaJoint) -> None:
        self.pitch = pitch
        self.roll = roll


class FakeElbow:
    def __init__(self, yaw: FakeOrbitaJoint, pitch: FakeOrbitaJoint) -> None:
        self.yaw = yaw
        self.pitch = pitch


class FakeWrist:
    def __init__(self, roll: FakeOrbitaJoint, pitch: FakeOrbitaJoint, yaw: FakeOrbitaJoint) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class FakeGripper:
    """In-memory stand-in for ``reachy2_sdk.grippers.parallel_gripper.ParallelGripper``."""

    def __init__(self, arm: FakeArm, *, opening: float = 50.0) -> None:
        self._arm = arm
        self.opening: float = float(opening)
        self.max_opening: float = 100.0
        self.goal_position: float = float(opening)
        self.cancel_count: int = 0
        self._is_on: bool = True

    def is_on(self) -> bool:
        return self._is_on

    def turn_on(self) -> bool:
        self._is_on = True
        return True

    def turn_off(self) -> bool:
        self._is_on = False
        return True

    def get_current_opening(self) -> float:
        return self.opening

    def set_opening(self, percentage: float) -> None:
        if not 0.0 <= float(percentage) <= 100.0:
            raise ValueError("opening percentage must be between 0 and 100")
        target = float(percentage)
        if not self._is_on:
            # Match reachy2-sdk ParallelGripper.set_opening when the gripper is off.
            raise RuntimeError("Gripper is off. Opening value not sent.")
        self.goal_position = target

        closing = target < self.opening
        if not closing:
            # Release any held object when opening beyond its width.
            for obj in self._arm._scene_objects():
                if obj.held_by == self._arm._hand and target > obj.opening_pct:
                    obj.held_by = None
            self.opening = target
            return

        ee_pos = self._arm.forward_kinematics(
            self._arm.get_current_positions(degrees=False), degrees=False
        )[:3, 3]
        for obj in self._arm._scene_objects():
            dist = float(np.linalg.norm(ee_pos - np.asarray(obj.position, dtype=np.float64)))
            if dist <= obj.capture_radius_m and target <= obj.opening_pct:
                if obj.slip:
                    # Hold the fingers at obj.opening_pct for slip_after_steps
                    # squeezes, then slip out to the commanded target.
                    obj.held_by = None
                    stalls = getattr(obj, "_stall_steps", 0)
                    if self.opening <= obj.opening_pct + 0.5 and target < obj.opening_pct - 1.0:
                        stalls += 1
                        obj._stall_steps = stalls  # type: ignore[attr-defined]
                    if stalls < obj.slip_after_steps:
                        self.opening = obj.opening_pct
                    else:
                        self.opening = target
                    return
                obj.held_by = self._arm._hand
                self.opening = obj.opening_pct
                return

        self.opening = target

    def open(self) -> None:
        self.set_opening(100.0)

    def close(self) -> None:
        self.set_opening(0.0)

    def cancel_all_goto(self) -> None:
        self.cancel_count += 1
        self.goal_position = self.opening


class FakeArm:
    """In-memory 7-DoF stand-in for ``reachy2_sdk.parts.arm.Arm``."""

    def __init__(
        self,
        hand: str,
        *,
        scene: dict[str, FakeGraspableObject] | None = None,
        initial_positions_rad: Sequence[float] = _DEFAULT_Q_RAD,
    ) -> None:
        self._hand = hand
        self._scene = scene if scene is not None else {}
        self._shoulder_y = 0.20 if hand == "left" else -0.20
        self._shoulder_z = 0.40
        self._is_on = True
        self.cancel_count = 0

        q_deg = [math.degrees(float(a)) for a in initial_positions_rad]
        self._joints = [FakeOrbitaJoint(deg) for deg in q_deg]
        self.shoulder = FakeShoulder(self._joints[0], self._joints[1])
        self.elbow = FakeElbow(self._joints[2], self._joints[3])
        self.wrist = FakeWrist(self._joints[4], self._joints[5], self._joints[6])
        self.gripper = FakeGripper(self)

    def _scene_objects(self) -> list[FakeGraspableObject]:
        return list(self._scene.values())

    def is_on(self) -> bool:
        return self._is_on

    def turn_on(self) -> bool:
        self._is_on = True
        self.gripper.turn_on()
        return True

    def turn_off(self) -> bool:
        self._is_on = False
        self.gripper.turn_off()
        return True

    def get_current_positions(self, degrees: bool = True) -> list[float]:
        if degrees:
            return [joint.present_position for joint in self._joints]
        return [math.radians(joint.present_position) for joint in self._joints]

    def send_goal_positions(self, check_positions: bool = True) -> None:
        del check_positions
        if not self._is_on:
            return
        for joint in self._joints:
            joint.present_position = float(joint.goal_position)
        ee_pos = self.forward_kinematics(self.get_current_positions(degrees=False), degrees=False)[
            :3, 3
        ]
        for obj in self._scene.values():
            if obj.held_by == self._hand:
                obj.position = (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]))

    def cancel_all_goto(self) -> None:
        self.cancel_count += 1
        for joint in self._joints:
            joint.goal_position = joint.present_position

    def forward_kinematics(
        self, joints_positions: Sequence[float] | None = None, degrees: bool = True
    ) -> np.ndarray:
        if joints_positions is None:
            q_rad = self.get_current_positions(degrees=False)
        else:
            if len(joints_positions) != 7:
                raise ValueError(f"expected 7 joint angles, got {len(joints_positions)}")
            q_rad = [math.radians(float(v)) if degrees else float(v) for v in joints_positions]

        mat = _trans(0.0, self._shoulder_y, self._shoulder_z)
        mat = mat @ _rot_y(q_rad[0])
        mat = mat @ _rot_x(q_rad[1])
        mat = mat @ _rot_z(q_rad[2])
        mat = mat @ _trans(0.0, 0.0, -_UPPER_ARM_M)
        mat = mat @ _rot_y(q_rad[3])
        mat = mat @ _trans(0.0, 0.0, -_FOREARM_M)
        mat = mat @ _rot_x(q_rad[4])
        mat = mat @ _rot_y(q_rad[5])
        mat = mat @ _rot_z(q_rad[6])
        mat = mat @ _trans(0.0, 0.0, -_TOOL_M)
        return mat

    def inverse_kinematics(
        self,
        target: np.ndarray,
        q0: Sequence[float] | None = None,
        degrees: bool = True,
    ) -> list[float]:
        target_mat = np.asarray(target, dtype=np.float64)
        if target_mat.shape != (4, 4):
            raise ValueError(f"expected 4x4 target matrix, got {target_mat.shape}")

        if q0 is None:
            q = np.array(self.get_current_positions(degrees=False), dtype=np.float64)
        else:
            if len(q0) != 7:
                raise ValueError(f"expected 7 initial joint angles, got {len(q0)}")
            q = np.array(
                [math.radians(float(v)) if degrees else float(v) for v in q0],
                dtype=np.float64,
            )

        eps = 1e-6
        damping = 1e-3
        target_pose = matrix_to_pose(target_mat, "world")

        for _ in range(80):
            cur = self.forward_kinematics(q, degrees=False)
            pos_err_vec = target_mat[:3, 3] - cur[:3, 3]
            ori_err_vec = _orientation_error_vector(cur[:3, :3], target_mat[:3, :3])
            err = np.concatenate([pos_err_vec, ori_err_vec])
            if (
                float(np.linalg.norm(pos_err_vec)) < 1e-5
                and float(np.linalg.norm(ori_err_vec)) < 1e-4
            ):
                break

            jac = np.zeros((6, 7), dtype=np.float64)
            for idx in range(7):
                q_pert = q.copy()
                q_pert[idx] += eps
                pert = self.forward_kinematics(q_pert, degrees=False)
                dp = (pert[:3, 3] - cur[:3, 3]) / eps
                dr = _orientation_error_vector(cur[:3, :3], pert[:3, :3]) / eps
                jac[:3, idx] = dp
                jac[3:, idx] = dr

            lhs = jac @ jac.T + (damping * damping) * np.eye(6)
            dq = jac.T @ np.linalg.solve(lhs, err)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > 0.25:
                dq *= 0.25 / step_norm
            q = q + dq
            q = np.clip(q, -math.pi, math.pi)

        reached = matrix_to_pose(self.forward_kinematics(q, degrees=False), "world")
        pos_err, ori_err = pose_error(reached, target_pose)
        if pos_err > 1e-3 or ori_err > 1e-2:
            raise ValueError("no joint solution reaches the target")

        if degrees:
            return [math.degrees(float(v)) for v in q]
        return [float(v) for v in q]


class FakeReachySDK:
    """Offline test double matching the subset of ``ReachySDK`` used by ``Reachy2Backend``."""

    def __init__(self, *, has_left: bool = True, has_right: bool = True) -> None:
        self.objects: dict[str, FakeGraspableObject] = {}
        self.l_arm: FakeArm | None = FakeArm("left", scene=self.objects) if has_left else None
        self.r_arm: FakeArm | None = FakeArm("right", scene=self.objects) if has_right else None
        self._connected = True

    def place_object(
        self,
        name: str,
        position: tuple[float, float, float],
        *,
        opening_pct: float = 35.0,
        slip: bool = False,
        capture_radius_m: float = 0.05,
    ) -> FakeGraspableObject:
        """Register an object in the fake scene for grasp and slip tests."""
        obj = FakeGraspableObject(
            name=name,
            position=position,
            opening_pct=opening_pct,
            slip=slip,
            capture_radius_m=capture_radius_m,
        )
        self.objects[name] = obj
        return obj

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    def turn_on(self) -> bool:
        ok = True
        for arm in (self.l_arm, self.r_arm):
            if arm is not None:
                ok = arm.turn_on() and ok
        return ok

    def turn_off(self) -> bool:
        ok = True
        for arm in (self.l_arm, self.r_arm):
            if arm is not None:
                ok = arm.turn_off() and ok
        return ok
