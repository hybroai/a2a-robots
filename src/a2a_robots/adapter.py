"""A2A adapter that exposes one robot through a2a-adapter."""

from __future__ import annotations

import json
from typing import Any

from a2a.types import Part
from a2a_adapter import AdapterMetadata, BaseA2AAdapter, CancelledByAdapterError, serve_agent
from google.protobuf import json_format, struct_pb2

from .backend import RobotBackend
from .controller import RobotController, SkillCancelled
from .limits import SafetyLimits
from .skills import RequestError, parse_request

JSON = "application/json"
_EXAMPLE_POSE = {
    "frame": "world",
    "units": "m",
    "position": [0.30, -0.20, 0.15],
    "orientation": {"quaternion": [1, 0, 0, 0]},
}
_EXAMPLE_RESULT = (
    '{"skill": "%s", "success": true, "failure_code": null, "detail": null, '
    '"final_pose": {"right": {...}}, "contact": {"right": {...}}, "held": {"right": false}}'
)


SAFETY_LIMITS_EXTENSION = "https://github.com/hybroai/a2a-robots/extensions/safety-limits/v1"


def _skill(skill_id: str, description: str, args: dict[str, Any]) -> dict[str, Any]:
    request = json.dumps({"skill": skill_id, "args": args})
    return {
        "id": skill_id,
        "name": skill_id,
        "description": f"{description} Request: {request} Response: {_EXAMPLE_RESULT % skill_id}",
        "tags": ["robot"],
        "examples": [request],
        "input_modes": [JSON, "text/plain"],
        "output_modes": [JSON, "text/plain"],
    }


SKILLS = [
    _skill("observe", "Read the pose, contact and held state of every hand. Always allowed.", {}),
    _skill(
        "move_to",
        "Move a hand to a pose. frame, units and orientation are required; "
        "a move larger than the step limit, or a target outside the workspace box, "
        "is refused before anything moves. The limits are in the agent card's "
        "safety-limits extension.",
        {"hand": "right", "pose": _EXAMPLE_POSE},
    ),
    _skill("open", "Open a hand.", {"hand": "right"}),
    _skill(
        "close", "Close a hand; held reports whether it is holding something.", {"hand": "right"}
    ),
    _skill("stop", "Stop all motion now and freeze every hand. Always allowed.", {}),
]


def _limits_extension(limits: SafetyLimits) -> dict[str, Any]:
    """A data-only A2A extension that states the limits the adapter enforces."""
    workspace = limits.workspace
    return {
        "uri": SAFETY_LIMITS_EXTENSION,
        "description": "Safety limits this robot enforces before anything moves.",
        "required": False,
        "params": {
            "frame": workspace.frame,
            "workspace": {"min_xyz": list(workspace.min_xyz), "max_xyz": list(workspace.max_xyz)},
            "max_step_m": limits.max_step_m,
            "max_step_rad": limits.max_step_rad,
            "skill_timeout_s": limits.skill_timeout_s,
            "stop_timeout_s": limits.stop_timeout_s,
        },
    }


class RobotAdapter(BaseA2AAdapter):
    """Accepts ``{"skill": ..., "args": {...}}`` as a JSON text part or a data part.

    Every reply carries the same result twice: as JSON text, for clients that
    only read text, and as an ``application/json`` data part.
    """

    def __init__(
        self,
        backend: RobotBackend,
        limits: SafetyLimits,
        *,
        name: str = "Robot",
        description: str = "Task-level robot skills over A2A.",
    ) -> None:
        self.controller = RobotController(backend, limits)
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any):
        context = kwargs.get("context")
        payload = _request_payload(user_input, context)
        request = parse_request(payload, self.controller.backend.hands)
        try:
            result = await self.controller.run(request, getattr(context, "task_id", None))
        except SkillCancelled as e:
            # cancel() already stopped the robot; the task is reported canceled.
            raise CancelledByAdapterError(str(e)) from None
        data = struct_pb2.Value()
        json_format.ParseDict(result.to_dict(), data)
        return [Part(text=result.to_json()), Part(data=data, media_type=JSON)]

    async def cancel(self, context_id: str | None = None, **kwargs: Any) -> None:
        """Return once the robot has stopped and the hands are frozen. Raises
        ``StopError`` if it did not stop in time, so the task fails instead of
        being reported canceled."""
        context = kwargs.get("context")
        await self.controller.cancel(getattr(context, "task_id", None))

    async def close(self) -> None:
        await self.controller.aclose()

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name,
            description=self._description,
            skills=SKILLS,
            extensions=[_limits_extension(self.controller.limits)],
            input_modes=[JSON, "text/plain"],
            output_modes=[JSON, "text/plain"],
        )


def _request_payload(user_input: str, context: Any) -> Any:
    message = getattr(context, "message", None)
    data_parts = [p for p in getattr(message, "parts", []) if p.WhichOneof("content") == "data"]
    if len(data_parts) > 1:
        raise RequestError("send one skill request per message")
    if data_parts:
        return json_format.MessageToDict(data_parts[0].data)
    try:
        return json.loads(user_input)
    except json.JSONDecodeError:
        raise RequestError('send the request as JSON: {"skill": ..., "args": {...}}') from None


def serve_robot(
    adapter: RobotAdapter, *, host: str = "127.0.0.1", port: int = 9100, **kwargs: Any
) -> None:
    """Serve on loopback by default: anyone who can reach the server can move the robot."""
    serve_agent(adapter, host=host, port=port, **kwargs)
