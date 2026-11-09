"""Safety limits checked before anything moves.

Only the target pose is checked, not the path to it. That is acceptable in
simulation; real hardware also needs the robot's own path and collision checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .skills import FailureCode, Pose


@dataclass(frozen=True, slots=True)
class Workspace:
    """Axis-aligned box, in metres, in ``frame``."""

    frame: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def contains(self, position: tuple[float, float, float]) -> bool:
        return all(
            lo <= v <= hi for v, lo, hi in zip(position, self.min_xyz, self.max_xyz, strict=True)
        )


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    workspace: Workspace
    max_step_m: float = 0.01
    max_step_rad: float = 0.05
    skill_timeout_s: float = 10.0
    # How long a stop may take before the robot is reported as possibly still moving.
    stop_timeout_s: float = 2.0

    def as_dict(self) -> dict[str, float]:
        return {"max_step_m": self.max_step_m, "max_step_rad": self.max_step_rad}

    def check_move(self, current: Pose, target: Pose) -> tuple[FailureCode, str] | None:
        """Return why ``target`` is refused, or None when it may run unchanged.

        Both poses must already be in ``workspace.frame``.
        """
        if not self.workspace.contains(target.position):
            return FailureCode.OUT_OF_WORKSPACE, "target position is outside the workspace box"
        distance = math.dist(current.position, target.position)
        if distance > self.max_step_m:
            return (
                FailureCode.STEP_TOO_LARGE,
                f"move of {distance:.4f} m exceeds max_step_m {self.max_step_m}",
            )
        angle = rotation_angle(current.orientation, target.orientation)
        if angle > self.max_step_rad:
            return (
                FailureCode.STEP_TOO_LARGE,
                f"rotation of {angle:.4f} rad exceeds max_step_rad {self.max_step_rad}",
            )
        return None


def rotation_angle(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Smallest angle in radians between two unit quaternions."""
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))
