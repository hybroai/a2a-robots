"""Arm specifications and robot presets for Hugging Face LeRobot robots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

JointUnit = Literal["deg", "rad"]

SO_ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

KOCH_ARM_JOINTS: tuple[str, ...] = SO_ARM_JOINTS


def _translation_transform(x: float, y: float, z: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = [float(x), float(y), float(z)]
    return mat


def _normalize_transform(raw: np.ndarray | Sequence[Sequence[float]] | None) -> np.ndarray:
    if raw is None:
        return np.eye(4, dtype=np.float64)
    mat = np.asarray(raw, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"base_transform must have shape (4, 4), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("base_transform must contain finite values")
    return mat.copy()


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Maps one ``a2a_robots`` hand to a LeRobot arm's joints, gripper, and kinematics."""

    arm_joints: tuple[str, ...]
    action_keys: tuple[str, ...] | None = None
    observation_keys: tuple[str, ...] | None = None
    gripper_action_key: str | None = "gripper.pos"
    gripper_observation_key: str | None = None
    joint_unit: JointUnit = "deg"
    urdf_path: str | Path | None = None
    target_frame_name: str = "gripper_frame_link"
    base_transform: np.ndarray | Sequence[Sequence[float]] | None = None
    position_weight: float = 1.0
    orientation_weight: float = 0.01
    check_orientation: bool = True
    gripper_open_value: float = 100.0
    gripper_closed_value: float = 0.0
    open_threshold: float = 95.0
    closed_threshold: float = 2.0
    gripper_step: float = 10.0

    def __post_init__(self) -> None:
        if not self.arm_joints:
            raise ValueError("arm_joints must not be empty")
        if len(set(self.arm_joints)) != len(self.arm_joints):
            raise ValueError("arm_joints must be unique")
        if self.joint_unit not in ("deg", "rad"):
            raise ValueError(f"unsupported joint_unit {self.joint_unit!r}; expected 'deg' or 'rad'")
        if self.position_weight <= 0.0:
            raise ValueError("position_weight must be positive")
        if self.orientation_weight < 0.0:
            raise ValueError("orientation_weight must be non-negative")
        if self.gripper_step <= 0.0:
            raise ValueError("gripper_step must be positive")

        act_keys = self.resolved_action_keys()
        obs_keys = self.resolved_observation_keys()
        if len(act_keys) != len(self.arm_joints):
            raise ValueError("action_keys must have the same length as arm_joints")
        if len(obs_keys) != len(self.arm_joints):
            raise ValueError("observation_keys must have the same length as arm_joints")

        if self.gripper_action_key is not None:
            if self.gripper_open_value == self.gripper_closed_value:
                raise ValueError("gripper_open_value and gripper_closed_value must differ")
            lo = min(self.gripper_closed_value, self.gripper_open_value)
            hi = max(self.gripper_closed_value, self.gripper_open_value)
            if not (lo <= self.closed_threshold <= hi and lo <= self.open_threshold <= hi):
                raise ValueError(
                    "open_threshold and closed_threshold must lie within gripper range"
                )
            if self.gripper_open_value > self.gripper_closed_value:
                if not (self.closed_threshold < self.open_threshold):
                    raise ValueError("closed_threshold must be less than open_threshold")
            elif not (self.open_threshold < self.closed_threshold):
                raise ValueError("open_threshold must be less than closed_threshold")

        # Validate transform shape eagerly.
        _normalize_transform(self.base_transform)

    def resolved_action_keys(self) -> tuple[str, ...]:
        if self.action_keys is not None:
            return tuple(self.action_keys)
        return tuple(f"{joint}.pos" for joint in self.arm_joints)

    def resolved_observation_keys(self) -> tuple[str, ...]:
        if self.observation_keys is not None:
            return tuple(self.observation_keys)
        return self.resolved_action_keys()

    def resolved_gripper_observation_key(self) -> str | None:
        if self.gripper_observation_key is not None:
            return self.gripper_observation_key
        return self.gripper_action_key

    def resolved_base_transform(self) -> np.ndarray:
        return _normalize_transform(self.base_transform)


def so101_hand_spec(
    urdf_path: str | Path | None = None,
    *,
    prefix: str = "",
    target_frame_name: str = "gripper_frame_link",
    base_transform: np.ndarray | Sequence[Sequence[float]] | None = None,
    orientation_weight: float = 0.01,
    check_orientation: bool = False,
) -> ArmSpec:
    """Return an ``ArmSpec`` for a 5-DoF SO-100 / SO-101 arm with ``use_degrees=True``.

    Because SO-100/SO-101 arms are 5-DoF, ``check_orientation`` defaults to
    ``False`` and ``orientation_weight`` defaults to ``0.01`` so IK tracks
    position accurately while softly aligning orientation.
    """
    action_keys = tuple(f"{prefix}{joint}.pos" for joint in SO_ARM_JOINTS)
    return ArmSpec(
        arm_joints=SO_ARM_JOINTS,
        action_keys=action_keys,
        observation_keys=action_keys,
        gripper_action_key=f"{prefix}gripper.pos",
        gripper_observation_key=f"{prefix}gripper.pos",
        joint_unit="deg",
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        base_transform=base_transform,
        orientation_weight=orientation_weight,
        check_orientation=check_orientation,
    )


def so101_arms(
    urdf_path: str | Path | None = None,
    *,
    hand: str = "right",
    target_frame_name: str = "gripper_frame_link",
    base_transform: np.ndarray | Sequence[Sequence[float]] | None = None,
    check_orientation: bool = False,
) -> dict[str, ArmSpec]:
    """Single-arm ``{hand: ArmSpec}`` mapping for ``SO100Follower`` / ``SO101Follower``."""
    return {
        hand: so101_hand_spec(
            urdf_path,
            target_frame_name=target_frame_name,
            base_transform=base_transform,
            check_orientation=check_orientation,
        )
    }


def bi_so_arms(
    urdf_path: str | Path | None = None,
    *,
    target_frame_name: str = "gripper_frame_link",
    left_base_transform: np.ndarray | Sequence[Sequence[float]] | None = None,
    right_base_transform: np.ndarray | Sequence[Sequence[float]] | None = None,
    check_orientation: bool = False,
) -> dict[str, ArmSpec]:
    """Bimanual ``{"left": ..., "right": ...}`` mapping for ``BiSOFollower``."""
    left_tf = (
        left_base_transform
        if left_base_transform is not None
        else _translation_transform(0.0, 0.20, 0.0)
    )
    right_tf = (
        right_base_transform
        if right_base_transform is not None
        else _translation_transform(0.0, -0.20, 0.0)
    )
    return {
        "left": so101_hand_spec(
            urdf_path,
            prefix="left_",
            target_frame_name=target_frame_name,
            base_transform=left_tf,
            check_orientation=check_orientation,
        ),
        "right": so101_hand_spec(
            urdf_path,
            prefix="right_",
            target_frame_name=target_frame_name,
            base_transform=right_tf,
            check_orientation=check_orientation,
        ),
    }


def koch_arms(
    urdf_path: str | Path | None = None,
    *,
    hand: str = "right",
    target_frame_name: str = "gripper_frame_link",
    base_transform: np.ndarray | Sequence[Sequence[float]] | None = None,
    check_orientation: bool = False,
) -> dict[str, ArmSpec]:
    """Single-arm ``{hand: ArmSpec}`` mapping for ``KochFollower`` with ``use_degrees=True``."""
    action_keys = tuple(f"{joint}.pos" for joint in KOCH_ARM_JOINTS)
    return {
        hand: ArmSpec(
            arm_joints=KOCH_ARM_JOINTS,
            action_keys=action_keys,
            observation_keys=action_keys,
            gripper_action_key="gripper.pos",
            gripper_observation_key="gripper.pos",
            joint_unit="deg",
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            base_transform=base_transform,
            orientation_weight=0.01,
            check_orientation=check_orientation,
        )
    }
