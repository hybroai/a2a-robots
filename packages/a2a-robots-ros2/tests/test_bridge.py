import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from a2a_adapter import CancelledByAdapterError, to_a2a

from a2a_robots_ros2 import (
    GoalHandle,
    GoalResult,
    Ros2BridgeAdapter,
    RosError,
    RosGateway,
    TopicSample,
)

SKILLS = [
    {
        "id": "rotate",
        "kind": "action",
        "ros_name": "/turtle1/rotate_absolute",
        "type": "turtlesim/action/RotateAbsolute",
        "description": "Turn the turtle to an absolute heading in radians.",
    },
    {"id": "spawn", "kind": "service", "ros_name": "/spawn", "type": "turtlesim/srv/Spawn"},
    {"id": "pose", "kind": "topic", "ros_name": "/turtle1/pose", "type": "turtlesim/msg/Pose"},
]


class FakeGoal(GoalHandle):
    def __init__(self, goal, cancel_finishes, cancel_fails):
        self.goal = goal
        self.cancel_finishes = cancel_finishes
        self.cancel_fails = cancel_fails
        self.cancel_calls = 0
        self._done = asyncio.get_running_loop().create_future()

    def finish(self, status="succeeded", **result):
        if not self._done.done():
            self._done.set_result(GoalResult(status, result))

    async def result(self):
        return await asyncio.shield(self._done)

    async def cancel(self):
        self.cancel_calls += 1
        await asyncio.sleep(0)
        if self.cancel_fails:
            raise RosError("no answer from the action server")
        if self.cancel_finishes:
            self.finish("canceled")


class FakeGateway(RosGateway):
    """Records calls. A goal runs ``script`` (feedback messages, then the
    result) unless ``script`` is None, in which case the test finishes it.
    ``early_feedback`` is delivered before ``send_goal`` returns."""

    def __init__(self):
        self.prepared = None
        self.goals: list[FakeGoal] = []
        self.script: list[dict] | None = [{"remaining": 1.0}, {"remaining": 0.5}]
        self.accept = asyncio.Event()
        self.accept.set()
        self.reject = False
        self.cancel_finishes = True
        self.cancel_fails = False
        self.early_feedback: list[dict] = []
        self.service_calls: list[dict] = []
        self.service_hangs = False
        self.topic: TopicSample | None = TopicSample({"x": 5.5, "theta": 0.0}, 0.04)
        self.closed = False

    def prepare(self, skills):
        self.prepared = [s.id for s in skills]

    async def send_goal(self, skill, goal, on_feedback):
        await self.accept.wait()
        if self.reject:
            raise RosError("goal rejected by the action server")
        handle = FakeGoal(goal, self.cancel_finishes, self.cancel_fails)
        self.goals.append(handle)
        for feedback in self.early_feedback:
            on_feedback(feedback)
        if self.script is not None:
            asyncio.get_running_loop().create_task(self._run(handle, on_feedback, self.script))
        return handle

    @staticmethod
    async def _run(handle, on_feedback, script):
        await asyncio.sleep(0.01)  # after the adapter reports the goal accepted
        for feedback in script:
            on_feedback(feedback)
        handle.finish(delta=0.5)

    async def call_service(self, skill, request):
        self.service_calls.append(request)
        if self.service_hangs:
            await asyncio.Event().wait()
        return {"name": request.get("name", "turtle2")}

    async def read_topic(self, skill):
        if self.topic is None:
            await asyncio.Event().wait()
        return self.topic

    async def close(self):
        self.closed = True


@pytest.fixture
def gateway():
    return FakeGateway()


def bridge(gateway, skills=SKILLS, **kwargs):
    return Ros2BridgeAdapter(skills, gateway, **kwargs)


def ctx(task_id="t1"):
    return SimpleNamespace(task_id=task_id, message=None)


def request(skill, **args):
    return json.dumps({"skill": skill, "args": args})


async def until(condition):
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


# ──── Over JSON-RPC ────


async def rpc(client, method, params):
    response = await client.post(
        "/", json={"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
    )
    return response.json()["result"]


def message(*parts):
    return {"message": {"role": "user", "messageId": uuid.uuid4().hex, "parts": list(parts)}}


async def send(adapter, *parts):
    app = to_a2a(adapter)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ros"
    ) as client:
        return await rpc(client, "message/send", message(*parts))


def text_part(skill, **args):
    return {"kind": "text", "text": request(skill, **args)}


def artifact_lines(task):
    text = "".join(p["text"] for p in task["artifacts"][0]["parts"] if p["kind"] == "text")
    return [json.loads(line) for line in text.splitlines()]


async def test_action_streams_accepted_feedback_and_result_lines(gateway):
    task = await send(bridge(gateway, max_feedback_hz=None), text_part("rotate", theta=1.57))
    assert task["status"]["state"] == "completed"
    assert artifact_lines(task) == [
        {"skill": "rotate", "status": "accepted"},
        {"skill": "rotate", "feedback": {"remaining": 1.0}},
        {"skill": "rotate", "feedback": {"remaining": 0.5}},
        {"skill": "rotate", "status": "succeeded", "result": {"delta": 0.5}},
    ]
    assert gateway.goals[0].goal == {"theta": 1.57}


async def test_feedback_sent_before_acceptance_follows_the_accepted_line(gateway):
    gateway.early_feedback = [{"remaining": 2.0}]
    task = await send(bridge(gateway, max_feedback_hz=None), text_part("rotate", theta=1.57))
    assert [
        line.get("status") or line["feedback"]["remaining"] for line in artifact_lines(task)
    ] == [
        "accepted",
        2.0,
        1.0,
        0.5,
        "succeeded",
    ]


async def test_feedback_is_throttled(gateway):
    gateway.script = [{"remaining": r} for r in (3.0, 2.0, 1.0)]
    task = await send(bridge(gateway), text_part("rotate", theta=1.0))
    feedback = [line for line in artifact_lines(task) if "feedback" in line]
    assert feedback == [{"skill": "rotate", "feedback": {"remaining": 3.0}}]


async def test_without_streaming_the_result_comes_as_text_and_data(gateway):
    task = await send(
        bridge(gateway, stream_feedback=False),
        {"kind": "data", "data": {"skill": "spawn", "args": {"name": "t2", "x": 1.0}}},
    )
    parts = task["artifacts"][0]["parts"]
    data = next(p["data"] for p in parts if p["kind"] == "data")
    text = next(p["text"] for p in parts if p["kind"] == "text")
    assert data == json.loads(text) == {"skill": "spawn", "response": {"name": "t2"}}
    assert gateway.service_calls == [{"name": "t2", "x": 1.0}]


async def test_topic_read_returns_the_latest_message_and_its_age(gateway):
    task = await send(bridge(gateway), text_part("pose"))
    assert artifact_lines(task) == [
        {"skill": "pose", "message": {"x": 5.5, "theta": 0.0}, "age_s": 0.04}
    ]


@pytest.mark.parametrize(
    ("part", "detail"),
    [
        ({"kind": "text", "text": "turn left"}, "send the request as JSON"),
        (text_part("fly"), "unknown skill 'fly'; skills: rotate, spawn, pose"),
        (text_part("pose", x=1), "takes no args"),
        ({"kind": "text", "text": '{"skill": "spawn", "args": [1]}'}, "args must be an object"),
        ({"kind": "text", "text": '{"skill": "spawn", "extra": 1}'}, "unknown request field"),
    ],
)
async def test_malformed_requests_fail_the_task_without_calling_ros(gateway, part, detail):
    task = await send(bridge(gateway), part)
    assert task["status"]["state"] == "failed"
    assert detail in task["status"]["message"]["parts"][0]["text"]
    assert gateway.goals == [] and gateway.service_calls == []


async def test_a_rejected_goal_fails_the_task(gateway):
    gateway.reject = True
    task = await send(bridge(gateway), text_part("rotate", theta=1.0))
    assert task["status"]["state"] == "failed"
    assert "goal rejected" in task["status"]["message"]["parts"][0]["text"]


async def test_an_aborted_goal_completes_with_its_status(gateway):
    gateway.script = None
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await until(lambda: gateway.goals)
    gateway.goals[0].finish("aborted", delta=0.1)
    result = json.loads((await run)[0].text)
    assert result == {"skill": "rotate", "status": "aborted", "result": {"delta": 0.1}}


async def test_agent_card_lists_each_allowlisted_interface(gateway):
    card = bridge(gateway).get_metadata()
    assert [s["id"] for s in card.skills] == ["rotate", "spawn", "pose"]
    rotate, spawn, pose = card.skills
    assert "/turtle1/rotate_absolute" in rotate["description"]
    assert "cannot be cancelled" in spawn["description"]
    assert "takes no args" in pose["description"].lower()
    assert card.streaming is True


# ──── Cancel, timeouts and shutdown ────


@pytest.mark.parametrize("stream_feedback", [True, False])
async def test_tasks_cancel_over_json_rpc_cancels_the_goal(gateway, stream_feedback):
    gateway.script = None
    app = to_a2a(bridge(gateway, stream_feedback=stream_feedback))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ros"
    ) as client:
        params = {**message(text_part("rotate", theta=1.0)), "configuration": {"blocking": False}}
        task_id = (await rpc(client, "message/send", params))["id"]
        await until(lambda: gateway.goals)

        cancelled = await rpc(client, "tasks/cancel", {"id": task_id})
        assert cancelled["status"]["state"] == "canceled"
        assert gateway.goals[0].cancel_calls == 1
        await asyncio.sleep(0.05)  # let the running task wind down
        task = await rpc(client, "tasks/get", {"id": task_id})
        assert task["status"]["state"] == "canceled"


async def test_cancel_cancels_the_goal_and_the_task_does_not_also_complete(gateway):
    gateway.script = None
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await until(lambda: gateway.goals)

    await adapter.cancel(context=ctx())
    assert gateway.goals[0].cancel_calls == 1
    # The goal ended before the A2A server cancelled the task: the executor
    # must not mark the task completed after tasks/cancel reported it cancelled.
    with pytest.raises(CancelledByAdapterError):
        await run


async def test_the_goal_is_not_cancelled_twice(gateway):
    gateway.script = None
    gateway.cancel_finishes = False
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await until(lambda: gateway.goals)

    await adapter.cancel(context=ctx())
    run.cancel()  # what the A2A server does next
    with pytest.raises(asyncio.CancelledError):
        await run
    assert gateway.goals[0].cancel_calls == 1


async def test_a_failed_cancel_is_reported_and_retried(gateway):
    gateway.script = None
    gateway.cancel_fails = True
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await until(lambda: gateway.goals)

    with pytest.raises(RosError, match="cancelling the goal failed.*may still be running"):
        await adapter.cancel(context=ctx())
    goal = gateway.goals[0]
    goal.cancel_fails = False
    await adapter.close()  # the goal was not stopped, so close tries again
    assert goal.cancel_calls == 2
    with pytest.raises(CancelledByAdapterError):
        await run


async def test_a_failed_cancel_over_json_rpc_fails_the_task(gateway):
    gateway.script = None
    gateway.cancel_fails = True
    app = to_a2a(bridge(gateway, stream_feedback=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ros"
    ) as client:
        params = {**message(text_part("rotate", theta=1.0)), "configuration": {"blocking": False}}
        task_id = (await rpc(client, "message/send", params))["id"]
        await until(lambda: gateway.goals)

        body = {"jsonrpc": "2.0", "id": "1", "method": "tasks/cancel", "params": {"id": task_id}}
        cancelled = (await client.post("/", json=body)).json()
        assert "may still be running" in cancelled["error"]["message"]
        for _ in range(100):  # the failed state is stored just after the reply
            task = await rpc(client, "tasks/get", {"id": task_id})
            if task["status"]["state"] != "working":
                break
            await asyncio.sleep(0.01)
        assert task["status"]["state"] == "failed"


async def test_a_cancel_that_times_out_is_not_treated_as_stopped(gateway):
    gateway.script = None
    gateway.accept.clear()
    adapter = bridge(gateway, stream_feedback=False, cancel_timeout_s=0.01)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await asyncio.sleep(0.01)
    gateway.accept.set()
    await until(lambda: gateway.goals)
    goal = gateway.goals[0]

    async def hang():
        goal.cancel_calls += 1
        await asyncio.Event().wait()

    goal.cancel = hang
    with pytest.raises(RosError, match="may still be running"):
        await adapter.cancel(context=ctx())
    run.cancel()  # what the A2A server does next; it retries the cancel
    with pytest.raises(asyncio.CancelledError):
        await run
    assert goal.cancel_calls == 2


async def test_a_goal_whose_timeout_cancel_fails_says_it_may_still_run(gateway):
    gateway.script = None
    gateway.cancel_fails = True
    skills = [{**SKILLS[0], "timeout_s": 0.05}]
    with pytest.raises(RosError, match="timed out after 0.05 s; cancelling the goal failed"):
        await bridge(gateway, skills).invoke(request("rotate", theta=1.0), context=ctx())


async def test_cancel_while_the_goal_is_being_sent_cancels_it_once_accepted(gateway):
    gateway.script = None
    gateway.accept.clear()
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await asyncio.sleep(0.01)
    await adapter.cancel(context=ctx())

    gateway.accept.set()
    with pytest.raises(CancelledByAdapterError):
        await asyncio.wait_for(run, 1)
    assert gateway.goals[0].cancel_calls == 1


async def test_cancelling_the_task_alone_still_cancels_the_goal(gateway):
    gateway.script = None
    adapter = bridge(gateway)
    stream = adapter.stream(request("rotate", theta=1.0), context=ctx())
    consumer = asyncio.create_task(anext(stream))
    await until(lambda: gateway.goals)
    assert json.loads(await consumer) == {"skill": "rotate", "status": "accepted"}

    consumer = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await until(lambda: gateway.goals[0].cancel_calls == 1)


async def test_cancel_for_an_unknown_task_is_a_no_op(gateway):
    await bridge(gateway).cancel(context=ctx("nothing-running"))
    await bridge(gateway).cancel()


async def test_a_goal_that_runs_too_long_is_cancelled(gateway):
    gateway.script = None
    skills = [{**SKILLS[0], "timeout_s": 0.05}]
    with pytest.raises(RosError, match="timed out after 0.05 s; the goal was cancelled"):
        await bridge(gateway, skills).invoke(request("rotate", theta=1.0), context=ctx())
    assert gateway.goals[0].cancel_calls == 1


async def test_a_goal_never_accepted_times_out_naming_the_server(gateway):
    gateway.accept.clear()
    skills = [{**SKILLS[0], "timeout_s": 0.05}]
    with pytest.raises(RosError, match="no goal was accepted.*/turtle1/rotate_absolute"):
        await bridge(gateway, skills).invoke(request("rotate", theta=1.0), context=ctx())


async def test_service_and_topic_timeouts_explain_what_happened(gateway):
    gateway.service_hangs = True
    gateway.topic = None
    skills = [{**SKILLS[1], "timeout_s": 0.05}, {**SKILLS[2], "timeout_s": 0.05}]
    adapter = bridge(gateway, skills)
    with pytest.raises(RosError, match="may still complete on the robot"):
        await adapter.invoke(request("spawn"), context=ctx())
    with pytest.raises(RosError, match="no message on /turtle1/pose"):
        await adapter.invoke(request("pose"), context=ctx())


async def test_an_exclusive_skill_refuses_a_second_call_while_running(gateway):
    gateway.script = None
    adapter = bridge(gateway, [{**SKILLS[0], "exclusive": True}], stream_feedback=False)
    first = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx("a")))
    await until(lambda: gateway.goals)
    with pytest.raises(RosError, match="rotate is already running"):
        await adapter.invoke(request("rotate", theta=2.0), context=ctx("b"))

    gateway.goals[0].finish()
    await first
    gateway.script = []
    await adapter.invoke(request("rotate", theta=2.0), context=ctx("c"))
    assert len(gateway.goals) == 2


async def test_close_cancels_running_goals_and_closes_the_gateway(gateway):
    gateway.script = None
    adapter = bridge(gateway, stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("rotate", theta=1.0), context=ctx()))
    await until(lambda: gateway.goals)
    await adapter.close()
    assert gateway.goals[0].cancel_calls == 1 and gateway.closed
    assert json.loads((await run)[0].text)["status"] == "canceled"


# ──── Construction ────


def test_the_gateway_is_prepared_with_the_allowlist(gateway):
    bridge(gateway)
    assert gateway.prepared == ["rotate", "spawn", "pose"]


def test_a_bad_allowlist_fails_before_the_gateway_is_touched(gateway):
    with pytest.raises(ValueError, match="kind"):
        bridge(gateway, [{**SKILLS[0], "kind": "publisher"}])
    assert gateway.prepared is None
