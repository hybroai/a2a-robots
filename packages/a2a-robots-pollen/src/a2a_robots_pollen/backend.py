"""Pollen Robotics Reachy 2 backend for a2a-robots."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from threading import Event
from typing import Any

import numpy as np

from a2a_robots import FailureCode, HandState, MotionOutcome, Pose, RobotBackend
from a2a_robots.planning import PlanError, Planner
from a2a_robots_pollen.planner import (
    Reachy2Planner,
    hand_joints,
    matrix_to_pose,
    pose_error,
)

HAND_ATTRS: dict[str, str] = {"left": "l_arm", "right": "r_arm"}


def _clamp_opening(pct: float) -> float:
    return min(100.0, max(0.0, float(pct)))


def connect(host: str = "localhost", **kwargs: Any) -> Any:
    """Connect to a live or simulated Reachy 2 gRPC server."""
    try:
        from reachy2_sdk import ReachySDK
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reachy2-sdk is required to connect to a Reachy 2 robot; "
            "install it with `pip install reachy2-sdk`"
        ) from exc

    reachy = ReachySDK(host=host, **kwargs)
    if hasattr(reachy, "is_connected") and not reachy.is_connected():
        raise ConnectionError(f"failed to connect to Reachy 2 gRPC server at {host!r}")
    return reachy


class Reachy2Backend(RobotBackend):
    """Drives Pollen Robotics Reachy 2 arms and grippers through ``reachy2-sdk``."""

    def __init__(
        self,
        reachy: Any,
        *,
        hands: Iterable[str] = ("left", "right"),
        frame: str = "world",
        realtime: bool = True,
        control_dt: float = 0.02,
        settle_time_s: float = 1.0,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = 0.05,
        max_speed_rad_s: float = 0.25,
        closed_opening_pct: float = 2.0,
        open_opening_pct: float = 95.0,
        gripper_step_pct: float = 10.0,
        min_gripper_steps: int = 5,
        contact_stall_s: float = 0.1,
        grasp_settle_s: float = 0.2,
        turn_on: bool = True,
        planner: Planner | None = None,
    ) -> None:
        """``contact_stall_s`` is how long the gripper must stop closing short of its
        target before ``close`` counts it as touching something; shorter stalls (a
        late state update, brief friction) are ignored. ``grasp_settle_s`` is how
        long it must stay stalled to count as a grasp. A stall that passes the first
        threshold but then closes past the object before the second is ``slipped``.
        The gripper only reports its opening, so a lag lasting between the two
        durations that then catches up in empty air also reads as ``slipped``."""
        if hasattr(reachy, "is_connected") and not reachy.is_connected():
            raise ConnectionError("Reachy SDK instance is not connected")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if settle_time_s < 0.0:
            raise ValueError("settle_time_s must be non-negative")
        if position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if orientation_tolerance_rad <= 0.0:
            raise ValueError("orientation_tolerance_rad must be positive")
        if gripper_step_pct <= 0.0:
            raise ValueError("gripper_step_pct must be positive")
        if not (0.0 <= closed_opening_pct < open_opening_pct <= 100.0):
            raise ValueError("expected 0.0 <= closed_opening_pct < open_opening_pct <= 100.0")
        if contact_stall_s <= 0.0:
            raise ValueError("contact_stall_s must be positive")
        if grasp_settle_s <= contact_stall_s:
            raise ValueError("grasp_settle_s must be greater than contact_stall_s")

        configured_hands = tuple(hands)
        if not configured_hands:
            raise ValueError("at least one hand must be configured")

        arms: dict[str, Any] = {}
        for hand in configured_hands:
            attr = HAND_ATTRS.get(hand)
            if attr is None:
                raise ValueError(f"unsupported hand {hand!r}; expected 'left' or 'right'")
            arm = getattr(reachy, attr, None)
            if arm is None:
                raise ValueError(f"Reachy instance has no active arm for hand {hand!r}")
            if getattr(arm, "gripper", None) is None:
                raise ValueError(f"Reachy arm {hand!r} has no gripper")
            if turn_on and hasattr(arm, "is_on") and not arm.is_on():
                if not arm.turn_on():
                    raise RuntimeError(f"failed to turn on Reachy arm {hand!r}")
            arms[hand] = arm

        self._reachy = reachy
        self._hands = frozenset(configured_hands)
        self._arms = arms
        self._frame = frame
        self._realtime = realtime
        self._control_dt = control_dt
        self._settle_steps = max(1, round(settle_time_s / control_dt))
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad
        self._closed_opening_pct = closed_opening_pct
        self._open_opening_pct = open_opening_pct
        self._gripper_step_pct = gripper_step_pct
        self._min_gripper_steps = max(1, min_gripper_steps)
        self._contact_stall_steps = max(2, round(contact_stall_s / control_dt))
        self._grasp_settle_steps = max(
            self._contact_stall_steps + 1, round(grasp_settle_s / control_dt)
        )
        self._lock = threading.Lock()
        self._closed_hands: set[str] = set()
        self._planner = planner or Reachy2Planner(
            arms,
            frame=frame,
            max_speed_rad_s=max_speed_rad_s,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

    @property
    def hands(self) -> frozenset[str]:
        return self._hands

    @staticmethod
    def _arm_joint_objects(arm: Any) -> tuple[Any, ...]:
        return (
            arm.shoulder.pitch,
            arm.shoulder.roll,
            arm.elbow.yaw,
            arm.elbow.pitch,
            arm.wrist.roll,
            arm.wrist.pitch,
            arm.wrist.yaw,
        )

    def _read_arm_radians(self, hand: str) -> list[float]:
        return [float(v) for v in self._arms[hand].get_current_positions(degrees=False)]

    def _write_arm_radians(self, hand: str, q_rad: Sequence[float]) -> None:
        arm = self._arms[hand]
        joints = self._arm_joint_objects(arm)
        for joint, angle in zip(joints, q_rad, strict=True):
            joint.goal_position = math.degrees(float(angle))
        arm.send_goal_positions(check_positions=False)

    def _hand_pose(self, hand: str) -> Pose:
        arm = self._arms[hand]
        # Pass explicit 7-element joint angles in radians to work around
        # reachy2-sdk 1.0.15 reading the gripper inside forward_kinematics(None).
        q_rad = self._read_arm_radians(hand)
        matrix = np.asarray(arm.forward_kinematics(q_rad, degrees=False), dtype=np.float64)
        return matrix_to_pose(matrix, self._frame)

    def _gripper_opening(self, hand: str) -> float:
        gripper = self._arms[hand].gripper
        opening = gripper.opening
        if opening is None and hasattr(gripper, "get_current_opening"):
            opening = gripper.get_current_opening()
        return float(opening)

    def _grasp_state(self, hand: str) -> tuple[bool, dict[str, float]]:
        opening = self._gripper_opening(hand)
        held = (
            hand in self._closed_hands
            and self._closed_opening_pct < opening < self._open_opening_pct
        )
        if not held:
            return False, {}
        force = round((100.0 - opening) / 10.0, 4)
        return True, {"finger": force}

    def observe(self) -> dict[str, HandState]:
        with self._lock:
            states: dict[str, HandState] = {}
            for hand in self._hands:
                pose = self._hand_pose(hand)
                held, contact = self._grasp_state(hand)
                states[hand] = HandState(pose=pose, contact=contact, held=held)
            return states

    def move_to(self, hand: str, target: Pose, stop: Event) -> MotionOutcome:
        expected_joints = hand_joints(hand)
        with self._lock:
            q0 = self._read_arm_radians(hand)
            start = dict(zip(expected_joints, q0, strict=True))
            try:
                trajectory = self._planner.plan(hand, start, target, self._control_dt)
            except PlanError as error:
                return MotionOutcome(error.failure_code, error.detail)

        if tuple(trajectory.joints) != expected_joints:
            return MotionOutcome(
                FailureCode.UNREACHABLE,
                f"planner returned joints {trajectory.joints!r}, expected {expected_joints!r}",
            )

        points = trajectory.points

        def track(step: int) -> bool:
            point = points[min(step, len(points) - 1)]
            self._write_arm_radians(hand, point)
            if step < len(points) - 1:
                return False
            pos_err, ori_err = pose_error(self._hand_pose(hand), target)
            return (
                pos_err <= self._position_tolerance_m and ori_err <= self._orientation_tolerance_rad
            )

        settled = self._run(stop, track, extra_steps=len(points))
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.UNREACHABLE, "the hand did not settle at the target")
        return MotionOutcome()

    def open(self, hand: str, stop: Event) -> MotionOutcome:
        with self._lock:
            start_opening = _clamp_opening(self._gripper_opening(hand))

        steps = max(
            self._min_gripper_steps,
            math.ceil(abs(100.0 - start_opening) / self._gripper_step_pct),
        )
        gripper = self._arms[hand].gripper

        def track(step: int) -> bool:
            fraction = min(1.0, (step + 1) / steps)
            target_pct = _clamp_opening(start_opening + (100.0 - start_opening) * fraction)
            gripper.set_opening(target_pct)
            return step >= steps - 1 and self._gripper_opening(hand) >= self._open_opening_pct

        settled = self._run(stop, track, extra_steps=steps)
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.TIMEOUT, "the gripper did not settle")
        with self._lock:
            # Clear grasp state only after a successful open, so a stopped or
            # timed-out open does not claim the hand released an object.
            self._closed_hands.discard(hand)
        return MotionOutcome()

    def close(self, hand: str, stop: Event) -> MotionOutcome:
        with self._lock:
            start_opening = _clamp_opening(self._gripper_opening(hand))

        steps = max(
            self._min_gripper_steps,
            math.ceil(abs(start_opening) / self._gripper_step_pct),
        )
        gripper = self._arms[hand].gripper
        touched = False
        stalled_steps = 0
        prev_actual_pct: float | None = None

        def track(step: int) -> bool:
            nonlocal touched, stalled_steps, prev_actual_pct
            fraction = min(1.0, (step + 1) / steps)
            target_pct = _clamp_opening(start_opening * (1.0 - fraction))
            gripper.set_opening(target_pct)
            actual_pct = self._gripper_opening(hand)
            # A stall is the opening not shrinking while still short of the target.
            if (
                prev_actual_pct is not None
                and abs(actual_pct - prev_actual_pct) < 0.5
                and actual_pct - target_pct > 1.0
                and self._closed_opening_pct < actual_pct < self._open_opening_pct
            ):
                stalled_steps += 1
            else:
                stalled_steps = 0
            prev_actual_pct = actual_pct
            # Only a sustained stall counts as contact, so a single stale reading
            # or a brief lag that then catches up is not reported as a slip.
            if stalled_steps >= self._contact_stall_steps:
                touched = True
            return step >= steps - 1 and (
                actual_pct <= self._closed_opening_pct or stalled_steps >= self._grasp_settle_steps
            )

        settled = self._run(stop, track, extra_steps=steps)
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "motion stopped before completion")
        if not settled:
            return MotionOutcome(FailureCode.TIMEOUT, "the gripper did not settle")

        with self._lock:
            # Mark the hand closed only after settle, so a stopped mid-close
            # never reports held=True from a partial opening.
            self._closed_hands.add(hand)
            held, _ = self._grasp_state(hand)
        if touched and not held:
            return MotionOutcome(FailureCode.SLIPPED, "object slipped during the grasp")
        return MotionOutcome()

    def hold(self) -> None:
        with self._lock:
            for hand in self._hands:
                arm = self._arms[hand]
                if hasattr(arm, "cancel_all_goto"):
                    arm.cancel_all_goto()
                if hasattr(arm.gripper, "cancel_all_goto"):
                    arm.gripper.cancel_all_goto()
                self._write_arm_radians(hand, self._read_arm_radians(hand))
                arm.gripper.set_opening(_clamp_opening(self._gripper_opening(hand)))

    def shutdown(self) -> None:
        with self._lock:
            if (
                hasattr(self._reachy, "is_connected")
                and self._reachy.is_connected()
                and hasattr(self._reachy, "disconnect")
            ):
                self._reachy.disconnect()

    def _run(self, stop: Event, step_fn: Callable[[int], bool], extra_steps: int = 0) -> bool:
        start = time.monotonic()
        budget = self._settle_steps + extra_steps
        for step in range(budget):
            # Poll stop outside self._lock so concurrent observe() never deadlocks.
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


PollenBackend = Reachy2Backend
