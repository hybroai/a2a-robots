"""Expose robots as A2A agents with task-level skills and adapter-enforced safety limits."""

from .adapter import RobotAdapter, serve_robot
from .backend import HandState, MotionOutcome, RobotBackend
from .controller import RobotController, SkillCancelled, StopError
from .fake import FakeBackend
from .limits import SafetyLimits, Workspace
from .skills import FailureCode, Pose, RequestError, Skill, SkillRequest, SkillResult

__all__ = [
    "FailureCode",
    "FakeBackend",
    "HandState",
    "MotionOutcome",
    "Pose",
    "RequestError",
    "RobotAdapter",
    "RobotBackend",
    "RobotController",
    "SafetyLimits",
    "Skill",
    "SkillRequest",
    "SkillCancelled",
    "SkillResult",
    "StopError",
    "Workspace",
    "serve_robot",
]
