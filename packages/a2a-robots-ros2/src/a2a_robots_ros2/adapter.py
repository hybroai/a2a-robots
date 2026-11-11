"""An A2A agent that passes allowlisted ROS 2 calls through.

This is a bring-up and debugging bridge, not the typed robot skill contract of
``a2a_robots.RobotAdapter``: there are no safety limits here beyond what the
robot's own ROS stack enforces.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from a2a.types import Part
from a2a_adapter import AdapterMetadata, BaseA2AAdapter, CancelledByAdapterError, serve_agent
from google.protobuf import json_format, struct_pb2

from .config import Kind, SkillSpec, parse_skills
from .gateway import GoalHandle, RequestError, RosError, RosGateway

logger = logging.getLogger(__name__)

JSON = "application/json"
_KIND_NOTES = {
    Kind.ACTION: "Sends an action goal; args are the goal fields. "
    "Streams feedback; tasks/cancel cancels the goal.",
    Kind.SERVICE: "Calls a service; args are the request fields. "
    "ROS 2 services cannot be cancelled: tasks/cancel stops waiting, "
    "but the call may still complete on the robot.",
    Kind.TOPIC: "Returns the latest message on the topic. Takes no args.",
}


@dataclass(slots=True)
class _Call:
    skill: SkillSpec
    goal: GoalHandle | None = None
    cancel_requested: bool = False
    goal_cancelled: bool = False  # set only once the server answered the cancel
    cancelling: asyncio.Future[None] | None = None


class Ros2BridgeAdapter(BaseA2AAdapter):
    """Accepts ``{"skill": ..., "args": {...}}`` as a JSON text part or a data part.

    ``invoke`` replies with the result as JSON text and as an
    ``application/json`` data part. With ``stream_feedback`` (the default) the
    reply is newline-delimited JSON in the response artifact instead: an
    ``accepted`` line and feedback lines for actions, then the result as the
    last line. Feedback is sent at most ``max_feedback_hz`` times per second;
    ``None`` sends every message.
    """

    def __init__(
        self,
        skills: Sequence[Mapping[str, Any]],
        gateway: RosGateway,
        *,
        name: str = "ROS 2 bridge",
        description: str = "Allowlisted ROS 2 actions, services and topics over A2A.",
        stream_feedback: bool = True,
        max_feedback_hz: float | None = 2.0,
        cancel_timeout_s: float = 5.0,
    ) -> None:
        self.skills = {spec.id: spec for spec in parse_skills(skills)}
        self._gateway = gateway
        gateway.prepare(tuple(self.skills.values()))
        self._name = name
        self._description = description
        self._stream_feedback = stream_feedback
        self._feedback_period_s = 1 / max_feedback_hz if max_feedback_hz else 0.0
        self._cancel_timeout_s = cancel_timeout_s
        self._calls: dict[str, _Call] = {}
        self._busy: set[str] = set()

    # ──── A2A ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any):
        result = await self._run(user_input, kwargs.get("context"), lambda event: None)
        data = struct_pb2.Value()
        json_format.ParseDict(result, data)
        return [Part(text=json.dumps(result)), Part(data=data, media_type=JSON)]

    def supports_streaming(self) -> bool:
        return self._stream_feedback

    async def stream(self, user_input: str, context_id: str | None = None, **kwargs: Any):
        events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        run = asyncio.ensure_future(self._run(user_input, kwargs.get("context"), events.put_nowait))
        try:
            while True:
                get = asyncio.ensure_future(events.get())
                await asyncio.wait({run, get}, return_when=asyncio.FIRST_COMPLETED)
                if not get.done():
                    get.cancel()
                    break
                yield _line(get.result())
            while not events.empty():
                yield _line(events.get_nowait())
            yield _line(run.result())
        finally:
            # Stop the call if the stream was abandoned, and always collect
            # its outcome so a failure is never left unretrieved.
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    async def cancel(self, context_id: str | None = None, **kwargs: Any) -> None:
        """Cancel the running goal and wait for the server's answer, so the
        task is reported cancelled only after the robot was told to stop.
        Raises ``RosError`` if the goal may still be running."""
        call = self._calls.get(getattr(kwargs.get("context"), "task_id", None) or "")
        if call is None:
            return
        call.cancel_requested = True
        await self._cancel_goal(call)

    async def close(self) -> None:
        await asyncio.gather(*(self._stop_goal(call) for call in list(self._calls.values())))
        await self._gateway.close()

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name,
            description=self._description,
            skills=[_card_skill(spec) for spec in self.skills.values()],
            input_modes=[JSON, "text/plain"],
            output_modes=[JSON, "text/plain"],
            streaming=self._stream_feedback,
        )

    # ──── Running one call ────

    async def _run(
        self, user_input: str, context: Any, emit: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        spec, args = self._parse(_request_payload(user_input, context))
        if spec.exclusive and spec.id in self._busy:
            raise RosError(f"{spec.id} is already running")
        task_id = getattr(context, "task_id", None) or uuid.uuid4().hex
        call = _Call(spec)
        self._calls[task_id] = call
        self._busy.add(spec.id)
        try:
            async with asyncio.timeout(spec.timeout_s):
                result = await self._call(call, args, emit)
        except TimeoutError:
            if call.cancel_requested:
                await self._goal_ended(call)
                raise CancelledByAdapterError(f"{spec.id} was cancelled") from None
            raise RosError(_timeout_detail(call)) from None
        finally:
            del self._calls[task_id]
            self._busy.discard(spec.id)
        if call.cancel_requested:
            # tasks/cancel has already reported the task cancelled; do not also complete it.
            raise CancelledByAdapterError(f"{spec.id} was cancelled")
        return result

    async def _call(
        self, call: _Call, args: dict[str, Any], emit: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        spec = call.skill
        if spec.kind is Kind.ACTION:
            return await self._run_action(call, args, emit)
        if spec.kind is Kind.SERVICE:
            response = await self._gateway.call_service(spec, args)
            return {"skill": spec.id, "response": response}
        sample = await self._gateway.read_topic(spec)
        return {"skill": spec.id, "message": sample.message, "age_s": sample.age_s}

    async def _run_action(
        self, call: _Call, args: dict[str, Any], emit: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        skill = call.skill.id
        last_sent = -float("inf")
        early: list[dict[str, Any]] | None = []  # feedback held until the accepted line

        def send_feedback(feedback: dict[str, Any]) -> None:
            nonlocal last_sent
            now = time.monotonic()
            if now - last_sent >= self._feedback_period_s:
                last_sent = now
                emit({"skill": skill, "feedback": feedback})

        def on_feedback(feedback: dict[str, Any]) -> None:
            if early is None:
                send_feedback(feedback)
            else:
                early.append(feedback)

        try:
            call.goal = await self._gateway.send_goal(call.skill, args, on_feedback)
            if call.cancel_requested:  # tasks/cancel arrived while the goal was being sent
                await self._stop_goal(call)
            emit({"skill": skill, "status": "accepted"})
            held, early = early, None
            for feedback in held:
                send_feedback(feedback)
            outcome = await call.goal.result()
        except asyncio.CancelledError:
            # tasks/cancel, a timeout or shutdown: the goal must not keep running.
            await self._stop_goal(call)
            raise
        return {"skill": skill, "status": outcome.status, "result": outcome.result}

    async def _goal_ended(self, call: _Call) -> None:
        """Wait, without a bound, until a goal whose cancel was not confirmed ends.

        When tasks/cancel fails, a2a-adapter reports the task canceled if the
        call has already ended, so a cancelled call must not end while its goal
        may still be running. The executor waits a bounded time and then fails
        the task, and cancelling this wait ends the call.
        """
        if call.goal is None or call.goal_cancelled:
            return
        with contextlib.suppress(Exception):
            await call.goal.result()

    async def _cancel_goal(self, call: _Call) -> None:
        """Cancel the goal, sharing one request between callers that ask at
        the same time. A failed cancel is retried by the next caller."""
        goal = call.goal
        if goal is None or call.goal_cancelled:
            return
        if call.cancelling is None:
            call.cancelling = asyncio.ensure_future(
                asyncio.wait_for(goal.cancel(), self._cancel_timeout_s)
            )
        cancelling = call.cancelling
        try:
            await asyncio.shield(cancelling)
        except Exception as e:
            if call.cancelling is cancelling:
                call.cancelling = None
            raise RosError(
                f"{call.skill.id}: cancelling the goal failed ({e!r}); it may still be running"
            ) from e
        call.goal_cancelled = True

    async def _stop_goal(self, call: _Call) -> None:
        try:
            await self._cancel_goal(call)
        except RosError as e:
            logger.warning("%s", e)

    def _parse(self, payload: Any) -> tuple[SkillSpec, dict[str, Any]]:
        known = ", ".join(self.skills)
        if not isinstance(payload, dict) or not isinstance(payload.get("skill"), str):
            raise RequestError(f'send {{"skill": ..., "args": {{...}}}}; skills: {known}')
        unknown = sorted(set(payload) - {"skill", "args"})
        if unknown:
            raise RequestError(f"unknown request field(s) {', '.join(unknown)}")
        spec = self.skills.get(payload["skill"])
        if spec is None:
            raise RequestError(f"unknown skill {payload['skill']!r}; skills: {known}")
        args = payload.get("args", {})
        if not isinstance(args, dict):
            raise RequestError("args must be an object")
        if spec.kind is Kind.TOPIC and args:
            raise RequestError(f"{spec.id} reads a topic and takes no args")
        return spec, args


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


def _timeout_detail(call: _Call) -> str:
    spec = call.skill
    after = f"{spec.timeout_s:g} s"
    if spec.kind is Kind.ACTION and call.goal is None:
        return f"{spec.id}: no goal was accepted within {after}; is {spec.ros_name} running?"
    if spec.kind is Kind.ACTION and call.goal_cancelled:
        return f"{spec.id}: timed out after {after}; the goal was cancelled"
    if spec.kind is Kind.ACTION:
        return f"{spec.id}: timed out after {after}; cancelling the goal failed, it may still run"
    if spec.kind is Kind.SERVICE:
        return f"{spec.id}: no response within {after}; the call may still complete on the robot"
    return f"{spec.id}: no message on {spec.ros_name} within {after}"


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event) + "\n"


def _card_skill(spec: SkillSpec) -> dict[str, Any]:
    request = json.dumps({"skill": spec.id, "args": {}})
    description = " ".join(
        part
        for part in (spec.description, f"ROS {spec.kind} {spec.ros_name} ({spec.type}).")
        if part
    )
    return {
        "id": spec.id,
        "name": spec.id,
        "description": f"{description} {_KIND_NOTES[spec.kind]} Request: {request}",
        "tags": ["ros2", str(spec.kind)],
        "examples": [request],
        "input_modes": [JSON, "text/plain"],
        "output_modes": [JSON, "text/plain"],
    }


def serve_bridge(
    adapter: Ros2BridgeAdapter, *, host: str = "127.0.0.1", port: int = 9200, **kwargs: Any
) -> None:
    """Serve on loopback by default: anyone who can reach the server can move the robot."""
    serve_agent(adapter, host=host, port=port, **kwargs)
