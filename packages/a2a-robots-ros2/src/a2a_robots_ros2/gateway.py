"""The ROS side of the bridge, behind a small interface.

``Ros2BridgeAdapter`` talks only to a ``RosGateway``. The rclpy gateway owns
the node, its executor thread and message conversion; tests use a fake.
Messages cross this boundary as plain JSON-style dicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from a2a_adapter import UserFacingError

from .config import SkillSpec

GoalStatus = Literal["succeeded", "aborted", "canceled"]


class RequestError(UserFacingError, ValueError):
    """The request is malformed. The A2A task fails with this message and nothing is sent to ROS."""


class RosError(UserFacingError):
    """The ROS call could not run: no server, goal rejected, or timed out. The task fails
    with this message, so it must name only ROS interfaces and states."""


@dataclass(frozen=True, slots=True)
class GoalResult:
    status: GoalStatus
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TopicSample:
    message: dict[str, Any]
    age_s: float  # time since the message arrived


class GoalHandle(ABC):
    """An accepted action goal."""

    @abstractmethod
    async def result(self) -> GoalResult:
        """Wait for the goal to finish."""

    @abstractmethod
    async def cancel(self) -> None:
        """Ask the action server to cancel the goal and wait for its answer."""


class RosGateway(ABC):
    """Every method except ``prepare`` is called on the asyncio event loop."""

    @abstractmethod
    def prepare(self, skills: Sequence[SkillSpec]) -> None:
        """Load the interface types and set up clients and subscriptions.

        Called once, from the adapter's constructor, so a missing message
        package fails at startup. Raise ``ValueError`` naming the skill.
        """

    @abstractmethod
    async def send_goal(
        self,
        skill: SkillSpec,
        goal: dict[str, Any],
        on_feedback: Callable[[dict[str, Any]], None],
    ) -> GoalHandle:
        """Wait for the action server, send the goal and wait until it is accepted.

        ``on_feedback`` must be called on the event loop. Raise
        ``RequestError`` for fields that do not fit the goal message, and
        ``RosError`` if the server rejects the goal. If the call is cancelled
        after the goal was sent, cancel the goal once it is accepted, so no
        goal keeps running unseen.
        """

    @abstractmethod
    async def call_service(self, skill: SkillSpec, request: dict[str, Any]) -> dict[str, Any]:
        """Wait for the service, call it and return the response.

        Raise ``RequestError`` for fields that do not fit the request message.
        ROS 2 services cannot be cancelled: if this call is cancelled, stop
        waiting; the service may still finish.
        """

    @abstractmethod
    async def read_topic(self, skill: SkillSpec) -> TopicSample:
        """Return the latest message, waiting for the first one if none arrived yet."""

    @abstractmethod
    async def close(self) -> None:
        """Destroy the node and stop its executor."""
