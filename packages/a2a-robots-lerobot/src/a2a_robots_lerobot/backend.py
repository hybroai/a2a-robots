"""LeRobot backend for ``a2a_robots``."""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from threading import Event
from typing import Any

import numpy as np

from a2a_robots import FailureCode, HandState, MotionOutcome, Pose, RobotBackend
from a2a_robots.planning import PlanError, Planner

from .config import ArmSpec
from .planner import (
    KinematicsSolver,
    LeRobotKinematicsPlanner,
    PlacoKinematics,
    matrix_to_pose,
    pose_error,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(value)))


def _check_use_degrees(robot: Any, arms: Mapping[str, ArmSpec]) -> None:
    if not any(spec.joint_unit == "deg" for spec in arms.values()):
        return
    config = getattr(robot, "config", None)
    if config is None:
        return
    for candidate in (
        config,
        getattr(config, "left_arm_config", None),
        getattr(config, "right_arm_config", None),
    ):
        if candidate is not None and getattr(candidate, "use_degrees", None) is False:
            raise ValueError(
                "LeRobot robot must be configured with use_degrees=True for degree-based kinematics"
            )


def _to_opening_pct(spec: ArmSpec, raw_value: float) -> float:
    span = spec.gripper_open_value - spec.gripper_closed_value
    return 100.0 * (float(raw_value) - spec.gripper_closed_value) / span


def _from_opening_pct(spec: ArmSpec, pct: float) -> float:
    clamped = _clamp(pct, 0.0, 100.0)
    span = spec.gripper_open_value - spec.gripper_closed_value
    return spec.gripper_closed_value + (clamped / 100.0) * span


class LeRobotArmBackend(RobotBackend):
    """Drives one or more arms managed by a ``lerobot.robots.Robot`` instance."""

    def __init__(
        self,
        robot: Any,
        arms: Mapping[str, ArmSpec],
        *,
        frame: str = "world",
        realtime: bool = True,
        control_dt: float = 0.02,
        settle_time_s: float = 1.0,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = 0.05,
        max_speed_rad_s: float = 0.25,
        max_joint_delta_rad: float = 1.0,
        min_gripper_steps: int = 5,
        contact_stall_s: float = 0.1,
        grasp_settle_s: float = 0.2,
        startup_grace_s: float = 0.2,
        grasp_opening_tol_pct: float = 5.0,
        connect: bool = False,
        require_use_degrees: bool = True,
        disconnect_on_shutdown: bool = True,
        planner: Planner | None = None,
        solvers: Mapping[str, KinematicsSolver] | None = None,
    ) -> None:
        if not arms:
            raise ValueError("at least one hand must be configured")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if settle_time_s < 0.0:
            raise ValueError("settle_time_s must be non-negative")
        if position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if orientation_tolerance_rad <= 0.0:
            raise ValueError("orientation_tolerance_rad must be positive")
        if max_speed_rad_s <= 0.0:
            raise ValueError("max_speed_rad_s must be positive")
        if max_joint_delta_rad <= 0.0:
            raise ValueError("max_joint_delta_rad must be positive")
        if contact_stall_s <= 0.0:
            raise ValueError("contact_stall_s must be positive")
        if grasp_settle_s <= contact_stall_s:
            raise ValueError("grasp_settle_s must be greater than contact_stall_s")
        if startup_grace_s < 0.0:
            raise ValueError("startup_grace_s must be non-negative")
        if grasp_opening_tol_pct < 0.0:
            raise ValueError("grasp_opening_tol_pct must be non-negative")

        self._specs = dict(arms)
        if require_use_degrees:
            _check_use_degrees(robot, self._specs)

        action_features = dict(robot.action_features)
        obs_features = dict(robot.observation_features)
        for hand, spec in self._specs.items():
            for key in spec.resolved_action_keys():
                if key not in action_features:
                    raise ValueError(f"hand {hand!r} action key {key!r} missing from robot")
            for key in spec.resolved_observation_keys():
                if key not in obs_features:
                    raise ValueError(f"hand {hand!r} observation key {key!r} missing from robot")
            if (
                spec.gripper_action_key is not None
                and spec.gripper_action_key not in action_features
            ):
                raise ValueError(
                    f"hand {hand!r} gripper action key "
                    f"{spec.gripper_action_key!r} missing from robot"
                )
            grip_obs_key = spec.resolved_gripper_observation_key()
            if grip_obs_key is not None and grip_obs_key not in obs_features:
                raise ValueError(
                    f"hand {hand!r} gripper observation key {grip_obs_key!r} missing from robot"
                )

        if connect and not robot.is_connected:
            # Never trigger interactive terminal calibration inside an A2A server.
            robot.connect(calibrate=False)
            if hasattr(robot, "is_calibrated") and not robot.is_calibrated:
                with contextlib.suppress(Exception):
                    robot.disconnect()
                raise RuntimeError(
                    "LeRobot robot is not calibrated; run calibration before starting the adapter"
                )
        elif not robot.is_connected:
            raise ConnectionError("LeRobot robot instance is not connected")
        elif hasattr(robot, "is_calibrated") and not robot.is_calibrated:
            raise RuntimeError("LeRobot robot instance is not calibrated")

        self._robot = robot
        self._hands = frozenset(self._specs)
        self._frame = frame
        self._realtime = realtime
        self._control_dt = control_dt
        self._settle_steps = max(1, round(settle_time_s / control_dt))
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad
        self._min_gripper_steps = max(1, min_gripper_steps)
        self._contact_stall_steps = max(2, round(contact_stall_s / control_dt))
        self._grasp_settle_steps = max(
            self._contact_stall_steps + 1, round(grasp_settle_s / control_dt)
        )
        self._startup_grace_steps = max(0, round(startup_grace_s / control_dt))
        self._grasp_opening_tol_pct = grasp_opening_tol_pct
        self._disconnect_on_shutdown = disconnect_on_shutdown
        self._lock = threading.Lock()
        self._closed_hands: set[str] = set()
        self._grasp_opening_pct: dict[str, float] = {}

        if planner is not None:
            self._planner: Planner = planner
            self._fk_solvers = self._build_fk_solvers(self._specs, solvers)
        else:
            kin_planner = LeRobotKinematicsPlanner(
                self._specs,
                solvers=solvers,
                frame=frame,
                max_speed_rad_s=max_speed_rad_s,
                max_joint_delta_rad=max_joint_delta_rad,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            self._planner = kin_planner
            self._fk_solvers = None

        self._base_transforms = {
            hand: spec.resolved_base_transform() for hand, spec in self._specs.items()
        }
        self._action_keys = tuple(action_features)
        initial_obs = self._robot.get_observation()
        self._target_action: dict[str, float] = {
            key: float(initial_obs[key]) if key in initial_obs else 0.0 for key in self._action_keys
        }

    @staticmethod
    def _build_fk_solvers(
        specs: Mapping[str, ArmSpec],
        solvers: Mapping[str, KinematicsSolver] | None,
    ) -> dict[str, KinematicsSolver] | None:
        if solvers is None and all(spec.urdf_path is None for spec in specs.values()):
            return None
        provided = dict(solvers or {})
        out: dict[str, KinematicsSolver] = {}
        for hand, spec in specs.items():
            if hand in provided:
                out[hand] = provided[hand]
            elif spec.urdf_path is not None:
                out[hand] = PlacoKinematics(
                    spec.urdf_path,
                    target_frame_name=spec.target_frame_name,
                    joint_names=spec.arm_joints,
                )
            else:
                raise ValueError(
                    f"hand {hand!r} needs either ArmSpec.urdf_path or an entry in solvers for FK"
                )
        return out

    @property
    def hands(self) -> frozenset[str]:
        return self._hands

    def _obs_to_rad(self, hand: str, obs: Mapping[str, Any]) -> list[float]:
        spec = self._specs[hand]
        raw = [float(obs[key]) for key in spec.resolved_observation_keys()]
        if spec.joint_unit == "deg":
            return [math.radians(v) for v in raw]
        return raw

    def _rad_to_action(self, hand: str, q_rad: Sequence[float]) -> dict[str, float]:
        spec = self._specs[hand]
        keys = spec.resolved_action_keys()
        if spec.joint_unit == "deg":
            return {key: math.degrees(float(val)) for key, val in zip(keys, q_rad, strict=True)}
        return {key: float(val) for key, val in zip(keys, q_rad, strict=True)}

    def _hand_pose_from_obs(self, hand: str, obs: Mapping[str, Any]) -> Pose:
        q_rad = self._obs_to_rad(hand, obs)
        if isinstance(self._planner, LeRobotKinematicsPlanner):
            return self._planner.forward_pose(hand, q_rad, self._frame)
        if self._fk_solvers is not None:
            local = np.asarray(self._fk_solvers[hand].forward_kinematics(q_rad), dtype=np.float64)
            return matrix_to_pose(self._base_transforms[hand] @ local, self._frame)
        if hasattr(self._planner, "forward_pose"):
            return self._planner.forward_pose(hand, q_rad, self._frame)  # type: ignore[no-any-return]
        raise RuntimeError(
            "custom planner has no forward_pose() and no solvers/urdf_path was provided"
        )

    def _gripper_opening_pct(self, hand: str, obs: Mapping[str, Any]) -> float:
        spec = self._specs[hand]
        obs_key = spec.resolved_gripper_observation_key()
        assert obs_key is not None
        return _to_opening_pct(spec, float(obs[obs_key]))

    def _still_grasping(self, hand: str, opening_pct: float) -> bool:
        spec = self._specs[hand]
        if spec.gripper_action_key is None:
            return False
        grasp_at = self._grasp_opening_pct.get(hand)
        closed_pct = _to_opening_pct(spec, spec.closed_threshold)
        open_pct = _to_opening_pct(spec, spec.open_threshold)
        return (
            hand in self._closed_hands
            and grasp_at is not None
            and abs(opening_pct - grasp_at) <= self._grasp_opening_tol_pct
            and closed_pct < opening_pct < open_pct
        )

    def _grasp_state(self, hand: str, obs: Mapping[str, Any]) -> tuple[bool, dict[str, float]]:
        spec = self._specs[hand]
        if spec.gripper_action_key is None:
            return False, {}
        opening_pct = self._gripper_opening_pct(hand, obs)
        if not self._still_grasping(hand, opening_pct):
            return False, {}
        estimated_force = round((100.0 - _clamp(opening_pct, 0.0, 100.0)) / 10.0, 4)
        return True, {"finger": estimated_force}

    def _send_updates_locked(self, updates: Mapping[str, float]) -> None:
        self._target_action.update(updates)
        self._robot.send_action(dict(self._target_action))

    def _sync_action_from_obs_locked(self, obs: Mapping[str, Any]) -> None:
        for hand, spec in self._specs.items():
            for obs_key, act_key in zip(
                spec.resolved_observation_keys(), spec.resolved_action_keys(), strict=True
            ):
                if act_key in self._target_action:
                    self._target_action[act_key] = float(obs[obs_key])
            grip_act = spec.gripper_action_key
            grip_obs = spec.resolved_gripper_observation_key()
            if grip_act is not None and grip_obs is not None and grip_act in self._target_action:
                opening_pct = _to_opening_pct(spec, float(obs[grip_obs]))
                if self._still_grasping(hand, opening_pct):
                    self._target_action[grip_act] = spec.gripper_closed_value
                else:
                    self._target_action[grip_act] = _from_opening_pct(spec, opening_pct)

    def observe(self) -> dict[str, HandState]:
        with self._lock:
            obs = self._robot.get_observation()
            states: dict[str, HandState] = {}
            for hand in self._hands:
                pose = self._hand_pose_from_obs(hand, obs)
                held, contact = self._grasp_state(hand, obs)
                states[hand] = HandState(pose=pose, contact=contact, held=held)
            return states

    def move_to(self, hand: str, target: Pose, stop: Event) -> MotionOutcome:
        spec = self._specs[hand]
        with self._lock:
            obs = self._robot.get_observation()
            self._sync_action_from_obs_locked(obs)
            q0 = self._obs_to_rad(hand, obs)
            start = dict(zip(spec.arm_joints, q0, strict=True))
            try:
                trajectory = self._planner.plan(hand, start, target, self._control_dt)
            except PlanError as error:
                return MotionOutcome(error.failure_code, error.detail)

        if tuple(trajectory.joints) != spec.arm_joints:
            raise ValueError(
                f"planner returned joints {trajectory.joints!r}, expected {spec.arm_joints!r}"
            )

        points = trajectory.points

        def track(step: int) -> bool:
            point = points[min(step, len(points) - 1)]
            self._send_updates_locked(self._rad_to_action(hand, point))
            if step < len(points) - 1:
                return False
            current_obs = self._robot.get_observation()
            pos_err, ori_err = pose_error(self._hand_pose_from_obs(hand, current_obs), target)
            if pos_err > self._position_tolerance_m:
                return False
            if spec.check_orientation and ori_err > self._orientation_tolerance_rad:
                return False
            return True

        settled = self._run(stop, track, extra_steps=len(points))
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.UNREACHABLE, "the hand did not settle at the target")
        return MotionOutcome()

    def open(self, hand: str, stop: Event) -> MotionOutcome:
        spec = self._specs[hand]
        if spec.gripper_action_key is None:
            if stop.is_set():
                return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
            return MotionOutcome(FailureCode.UNREACHABLE, f"hand {hand!r} has no gripper")

        grip_key = spec.gripper_action_key
        open_pct_threshold = _to_opening_pct(spec, spec.open_threshold)
        step_pct = (
            100.0 * spec.gripper_step / abs(spec.gripper_open_value - spec.gripper_closed_value)
        )

        with self._lock:
            obs = self._robot.get_observation()
            self._sync_action_from_obs_locked(obs)
            start_opening = _clamp(self._gripper_opening_pct(hand, obs), 0.0, 100.0)

        steps = max(self._min_gripper_steps, math.ceil(abs(100.0 - start_opening) / step_pct))

        def track(step: int) -> bool:
            fraction = min(1.0, (step + 1) / steps)
            target_pct = _clamp(start_opening + (100.0 - start_opening) * fraction, 0.0, 100.0)
            self._send_updates_locked({grip_key: _from_opening_pct(spec, target_pct)})
            if step < steps - 1:
                return False
            current_obs = self._robot.get_observation()
            return self._gripper_opening_pct(hand, current_obs) >= open_pct_threshold

        settled = self._run(stop, track, extra_steps=steps)
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.TIMEOUT, "the gripper did not settle")
        with self._lock:
            self._closed_hands.discard(hand)
            self._grasp_opening_pct.pop(hand, None)
        return MotionOutcome()

    def close(self, hand: str, stop: Event) -> MotionOutcome:
        spec = self._specs[hand]
        if spec.gripper_action_key is None:
            if stop.is_set():
                return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
            return MotionOutcome(FailureCode.UNREACHABLE, f"hand {hand!r} has no gripper")

        grip_key = spec.gripper_action_key
        closed_pct_threshold = _to_opening_pct(spec, spec.closed_threshold)
        open_pct_threshold = _to_opening_pct(spec, spec.open_threshold)
        step_pct = (
            100.0 * spec.gripper_step / abs(spec.gripper_open_value - spec.gripper_closed_value)
        )

        with self._lock:
            obs = self._robot.get_observation()
            self._sync_action_from_obs_locked(obs)
            start_opening = _clamp(self._gripper_opening_pct(hand, obs), 0.0, 100.0)

        steps = max(self._min_gripper_steps, math.ceil(abs(start_opening) / step_pct))
        touched = False
        moved_from_start = False
        stalled_steps = 0
        prev_actual_pct: float | None = None

        def track(step: int) -> bool:
            nonlocal touched, moved_from_start, stalled_steps, prev_actual_pct
            fraction = min(1.0, (step + 1) / steps)
            target_pct = _clamp(start_opening * (1.0 - fraction), 0.0, 100.0)
            self._send_updates_locked({grip_key: _from_opening_pct(spec, target_pct)})
            actual_pct = self._gripper_opening_pct(hand, self._robot.get_observation())
            if abs(actual_pct - start_opening) >= 0.5:
                moved_from_start = True
            if (
                (step >= self._startup_grace_steps or moved_from_start)
                and prev_actual_pct is not None
                and abs(actual_pct - prev_actual_pct) < 0.5
                and actual_pct - target_pct > 1.0
                and closed_pct_threshold < actual_pct < open_pct_threshold
            ):
                stalled_steps += 1
            else:
                stalled_steps = 0
            prev_actual_pct = actual_pct
            if stalled_steps >= self._contact_stall_steps:
                touched = True
            return step >= steps - 1 and (
                actual_pct <= closed_pct_threshold or stalled_steps >= self._grasp_settle_steps
            )

        settled = self._run(stop, track, extra_steps=steps)
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.TIMEOUT, "the gripper did not settle")

        with self._lock:
            obs = self._robot.get_observation()
            opening_pct = self._gripper_opening_pct(hand, obs)
            self._closed_hands.add(hand)
            self._grasp_opening_pct[hand] = opening_pct
            held, _ = self._grasp_state(hand, obs)
        if touched and not held:
            return MotionOutcome(FailureCode.SLIPPED, "object slipped during the grasp")
        return MotionOutcome()

    def hold(self) -> None:
        with self._lock:
            if getattr(self._robot, "supports_command_hold", False) and hasattr(
                self._robot, "hold"
            ):
                self._robot.hold()
            obs = self._robot.get_observation()
            self._sync_action_from_obs_locked(obs)
            if not getattr(self._robot, "supports_command_hold", False):
                self._robot.send_action(dict(self._target_action))

    def shutdown(self) -> None:
        with self._lock:
            if self._disconnect_on_shutdown and self._robot.is_connected:
                self._robot.disconnect()

    def _run(self, stop: Event, step_fn: Callable[[int], bool], extra_steps: int = 0) -> bool:
        start = time.monotonic()
        budget = self._settle_steps + extra_steps
        for step in range(budget):
            if stop.is_set():
                return False
            with self._lock:
                done = step_fn(step)
            if done:
                return True
            if self._realtime:
                remaining = start + (step + 1) * self._control_dt - time.monotonic()
                if remaining > 0 and stop.wait(remaining):
                    return False
        return False
