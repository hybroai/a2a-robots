"""Skill requests and results: the wire contract every robot shares.

A request is ``{"skill": <name>, "args": {...}}``. Parsing validates shape,
types and units only; whether a move is safe is decided by ``SafetyLimits``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from a2a_adapter import UserFacingError


class Skill(StrEnum):
    OBSERVE = "observe"
    MOVE_TO = "move_to"
    OPEN = "open"
    CLOSE = "close"
    STOP = "stop"


MOTION_SKILLS = frozenset({Skill.MOVE_TO, Skill.OPEN, Skill.CLOSE})


class FailureCode(StrEnum):
    """Why a skill that ran (or was refused before moving) did not succeed."""

    STEP_TOO_LARGE = "step_too_large"
    OUT_OF_WORKSPACE = "out_of_workspace"
    BUSY = "busy"
    STOPPED = "stopped"
    TIMEOUT = "timeout"
    SLIPPED = "slipped"
    UNREACHABLE = "unreachable"


class RequestError(UserFacingError, ValueError):
    """The request is malformed. The A2A task fails with this message; nothing moves."""


@dataclass(frozen=True, slots=True)
class Pose:
    """Position in metres and a unit quaternion (w, x, y, z), in a named frame."""

    frame: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "units": "m",
            "position": list(self.position),
            "orientation": {"quaternion": list(self.orientation)},
        }


@dataclass(frozen=True, slots=True)
class SkillRequest:
    skill: Skill
    hand: str | None = None
    pose: Pose | None = None


@dataclass(slots=True)
class SkillResult:
    """The one result shape every skill returns."""

    skill: Skill
    success: bool
    failure_code: FailureCode | None = None
    detail: str | None = None
    final_pose: dict[str, Any] = field(default_factory=dict)
    contact: dict[str, Any] = field(default_factory=dict)
    held: dict[str, bool] = field(default_factory=dict)
    limits: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["limits"] is None:
            del data["limits"]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def parse_request(payload: Any, hands: frozenset[str]) -> SkillRequest:
    """Validate a decoded request against the robot's hands."""
    if not isinstance(payload, dict):
        raise RequestError('request must be a JSON object: {"skill": ..., "args": {...}}')
    unknown = set(payload) - {"skill", "args"}
    if unknown:
        raise RequestError(f"unknown request fields: {sorted(unknown)}")
    try:
        skill = Skill(payload.get("skill"))
    except ValueError:
        raise RequestError(f"skill must be one of {[s.value for s in Skill]}") from None
    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise RequestError("args must be a JSON object")

    if skill in (Skill.OBSERVE, Skill.STOP):
        _reject_extra(args, set())
        return SkillRequest(skill)

    hand = args.get("hand")
    if hand not in hands:
        raise RequestError(f"hand must be one of {sorted(hands)}")
    if skill is Skill.MOVE_TO:
        _reject_extra(args, {"hand", "pose"})
        return SkillRequest(skill, hand, _parse_pose(args.get("pose")))
    _reject_extra(args, {"hand"})
    return SkillRequest(skill, hand)


def _reject_extra(args: dict[str, Any], allowed: set[str]) -> None:
    extra = set(args) - allowed
    if extra:
        raise RequestError(f"unexpected args: {sorted(extra)}")


def _parse_pose(raw: Any) -> Pose:
    if not isinstance(raw, dict):
        raise RequestError("pose must be an object with frame, units, position and orientation")
    missing = {"frame", "units", "position", "orientation"} - set(raw)
    if missing:
        raise RequestError(f"pose is missing {sorted(missing)}")
    frame = raw["frame"]
    if not isinstance(frame, str) or not frame:
        raise RequestError("pose.frame must be a non-empty string")
    if raw["units"] != "m":
        raise RequestError('pose.units must be "m"')
    position = _numbers(raw["position"], 3, "pose.position")
    orientation = raw["orientation"]
    if not isinstance(orientation, dict) or set(orientation) != {"quaternion"}:
        raise RequestError('pose.orientation must be {"quaternion": [w, x, y, z]}')
    quaternion = _numbers(orientation["quaternion"], 4, "pose.orientation.quaternion")
    if abs(math.sqrt(sum(q * q for q in quaternion)) - 1.0) > 1e-3:
        raise RequestError("pose.orientation.quaternion must be a unit quaternion")
    return Pose(frame, position, quaternion)  # type: ignore[arg-type]


def _numbers(raw: Any, count: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(raw, list)
        or len(raw) != count
        or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw)
        or not all(math.isfinite(v) for v in raw)
    ):
        raise RequestError(f"{name} must be a list of {count} finite numbers")
    return tuple(float(v) for v in raw)
