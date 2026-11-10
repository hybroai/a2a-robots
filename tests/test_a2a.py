"""End to end over JSON-RPC: a client drives the fake robot through the A2A server."""

import asyncio
import json
import uuid

import httpx
import pytest
from a2a_adapter import to_a2a
from helpers import make_limits, pose

from a2a_robots import SAFETY_LIMITS_EXTENSION, FakeBackend, RobotAdapter


@pytest.fixture
def backend():
    return FakeBackend(steps=200, step_time_s=0.01)


@pytest.fixture
async def client(backend):
    app = to_a2a(RobotAdapter(backend, make_limits(), name="Fake robot"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://robot"
    ) as http:
        yield http


async def rpc(client, method, params):
    response = await client.post(
        "/", json={"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
    )
    assert response.status_code == 200
    return response.json()


def message(*parts):
    return {"message": {"role": "user", "messageId": uuid.uuid4().hex, "parts": list(parts)}}


def data_part(skill, **args):
    return {"kind": "data", "data": {"skill": skill, "args": args}}


def text_part(skill, **args):
    return {"kind": "text", "text": json.dumps({"skill": skill, "args": args})}


def result_of(task):
    """The skill result from the artifact's data part, checked against its text twin."""
    parts = task["artifacts"][0]["parts"]
    data = next(p["data"] for p in parts if p["kind"] == "data")
    text = next(p["text"] for p in parts if p["kind"] == "text")
    assert json.loads(text) == data
    assert json.loads(task["status"]["message"]["parts"][0]["text"]) == data
    return data


@pytest.mark.parametrize("part", [data_part("observe"), text_part("observe")])
async def test_observe_over_json_rpc(client, part):
    task = (await rpc(client, "message/send", message(part)))["result"]
    assert task["status"]["state"] == "completed"
    result = result_of(task)
    assert result["success"] is True
    assert set(result["final_pose"]) == {"left", "right"}


async def test_a2a_1_0_send_message_with_data_part(client):
    params = {
        "message": {
            "role": "ROLE_USER",
            "messageId": uuid.uuid4().hex,
            "parts": [{"data": {"skill": "observe"}}],
        }
    }
    reply = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": params},
        headers={"A2A-Version": "1.0"},
    )
    task = reply.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    data = next(p["data"] for p in task["artifacts"][0]["parts"] if "data" in p)
    assert data["success"] is True


async def test_malformed_request_fails_the_task_without_moving(client, backend):
    before = backend.observe()
    task = (await rpc(client, "message/send", message({"kind": "text", "text": "wave hello"})))[
        "result"
    ]
    assert task["status"]["state"] == "failed"
    assert "send the request as JSON" in task["status"]["message"]["parts"][0]["text"]
    assert backend.observe() == before


async def test_oversized_move_completes_with_step_too_large(client):
    part = data_part("move_to", hand="right", pose=pose(0.3, 0.1, 0.4))
    result = result_of((await rpc(client, "message/send", message(part)))["result"])
    assert result["success"] is False
    assert result["failure_code"] == "step_too_large"
    assert result["limits"]["max_step_m"] == 0.05


async def test_stop_preempts_a_move_sent_by_another_client(client, backend):
    move = asyncio.create_task(
        rpc(
            client,
            "message/send",
            message(data_part("move_to", hand="right", pose=pose(0.3, 0.1, 0.14))),
        )
    )
    await asyncio.sleep(0.2)
    stopped = result_of((await rpc(client, "message/send", message(data_part("stop"))))["result"])
    moved = result_of((await asyncio.wait_for(move, 1))["result"])

    assert stopped["success"] is True
    assert moved["failure_code"] == "stopped"
    assert backend.hold_calls == 1


async def start_move(client):
    params = {
        **message(data_part("open", hand="right")),
        "configuration": {"blocking": False},
    }
    task_id = (await rpc(client, "message/send", params))["result"]["id"]
    await asyncio.sleep(0.1)
    return task_id


async def wait_for_state(client, task_id):
    for _ in range(100):  # the final state is stored just after the cancel reply
        task = (await rpc(client, "tasks/get", {"id": task_id}))["result"]
        if task["status"]["state"] not in ("submitted", "working"):
            return task
        await asyncio.sleep(0.01)
    return task


async def test_tasks_cancel_replies_after_the_hands_are_frozen(client, backend):
    task_id = await start_move(client)
    cancelled = (await rpc(client, "tasks/cancel", {"id": task_id}))["result"]

    assert backend.hold_calls == 1
    assert cancelled["status"]["state"] == "canceled"
    assert (await wait_for_state(client, task_id))["status"]["state"] == "canceled"


async def test_tasks_cancel_fails_the_task_when_the_robot_does_not_stop(monkeypatch):
    from test_controller import IgnoresStopBackend

    # a2a-adapter waits this long for the run to end before it fails the task.
    monkeypatch.setattr("a2a_adapter.executor._CANCEL_SETTLE_TIMEOUT", 0.2)
    backend = IgnoresStopBackend()
    app = to_a2a(RobotAdapter(backend, make_limits(stop_timeout_s=0.05)))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://robot"
    ) as client:
        task_id = await start_move(client)
        cancelled = (await rpc(client, "tasks/cancel", {"id": task_id}))["result"]
        backend.release.set()

        assert cancelled["status"]["state"] == "failed"
        assert "may still be moving" in cancelled["status"]["message"]["parts"][0]["text"]
        assert (await wait_for_state(client, task_id))["status"]["state"] == "failed"


async def test_tasks_cancel_is_canceled_when_a_slow_stop_ends_in_time(monkeypatch):
    from test_controller import IgnoresStopBackend

    monkeypatch.setattr("a2a_adapter.executor._CANCEL_SETTLE_TIMEOUT", 2.0)
    backend = IgnoresStopBackend()
    app = to_a2a(RobotAdapter(backend, make_limits(stop_timeout_s=0.05)))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://robot"
    ) as client:
        task_id = await start_move(client)
        reply = asyncio.create_task(rpc(client, "tasks/cancel", {"id": task_id}))
        await asyncio.sleep(0.2)  # the stop has timed out; the motion is still going
        assert not reply.done()
        backend.release.set()

        assert (await reply)["result"]["status"]["state"] == "canceled"


async def test_agent_card_lists_skills_with_examples_and_json_modes(client):
    card = (await client.get("/.well-known/agent-card.json")).json()
    assert card["name"] == "Fake robot"
    assert "application/json" in card["defaultInputModes"]
    assert "application/json" in card["defaultOutputModes"]
    skills = {s["id"]: s for s in card["skills"]}
    assert set(skills) == {"observe", "move_to", "open", "close", "stop"}
    for skill in skills.values():
        assert json.loads(skill["examples"][0])["skill"] == skill["id"]
        assert "Request:" in skill["description"] and "Response:" in skill["description"]


async def test_agent_card_states_the_limits_in_an_extension(client):
    card = (await client.get("/.well-known/agent-card.json")).json()
    (extension,) = card["capabilities"]["extensions"]

    assert extension["uri"] == SAFETY_LIMITS_EXTENSION
    assert not extension.get("required", False)  # data-only: clients need not activate it
    assert extension["params"] == {
        "frame": "world",
        "workspace": {"min_xyz": [0.0, -0.5, 0.0], "max_xyz": [0.6, 0.5, 0.5]},
        "max_step_m": 0.05,
        "max_step_rad": 0.1,
        "skill_timeout_s": 5.0,
        "stop_timeout_s": 2.0,
    }
