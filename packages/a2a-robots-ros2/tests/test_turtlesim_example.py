"""The turtlesim example against a headless turtlesim_node, over JSON-RPC.
Skipped without ROS 2 and turtlesim unless A2A_ROBOTS_REQUIRE_ROS is set."""

import asyncio
import json
import os
import runpy
import subprocess
import uuid
from pathlib import Path

import httpx
import pytest

if os.environ.get("A2A_ROBOTS_REQUIRE_ROS"):
    import turtlesim
else:
    turtlesim = pytest.importorskip("turtlesim")

from a2a_adapter import to_a2a  # noqa: E402
from ament_index_python.packages import get_package_prefix  # noqa: E402

EXAMPLE = Path(__file__).parents[1] / "examples" / "turtlesim_bridge.py"


@pytest.fixture
def turtlesim_node():
    node = Path(get_package_prefix("turtlesim")) / "lib" / "turtlesim" / "turtlesim_node"
    process = subprocess.Popen(
        [str(node)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield
    process.terminate()
    process.wait(10)


@pytest.fixture
async def client(turtlesim_node):
    adapter = runpy.run_path(str(EXAMPLE))["make_adapter"]()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=to_a2a(adapter)), base_url="http://turtlesim"
        ) as client:
            yield client
    finally:
        await adapter.close()


async def call(client, skill, **args):
    params = {
        "message": {
            "role": "user",
            "messageId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": json.dumps({"skill": skill, "args": args})}],
        }
    }
    body = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": params}
    task = (await client.post("/", json=body)).json()["result"]
    assert task["status"]["state"] == "completed", task["status"]
    text = "".join(p["text"] for p in task["artifacts"][0]["parts"] if p["kind"] == "text")
    return [json.loads(line) for line in text.splitlines()]


async def test_the_example_drives_turtlesim(client):
    (pose,) = await call(client, "pose")  # waits for turtlesim to publish
    assert pose["message"]["x"] == pytest.approx(5.544, abs=0.01)

    lines = await call(client, "rotate", theta=1.57)
    assert lines[0] == {"skill": "rotate", "status": "accepted"}
    assert lines[-1]["status"] == "succeeded"

    await call(client, "teleport", x=2.0, y=3.0, theta=0.0)
    for _ in range(50):  # the next pose message shows the move
        (pose,) = await call(client, "pose")
        if pose["message"]["x"] == pytest.approx(2.0, abs=0.01):
            break
        await asyncio.sleep(0.05)
    assert pose["message"]["y"] == pytest.approx(3.0, abs=0.01)

    (spawned,) = await call(client, "spawn", x=8.0, y=8.0, theta=0.0, name="t2")
    assert spawned == {"skill": "spawn", "response": {"name": "t2"}}
