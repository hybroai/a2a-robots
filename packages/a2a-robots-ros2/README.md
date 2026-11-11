# a2a-robots-ros2

An allowlisted ROS 2 bridge for [a2a-robots](../../README.md): it exposes the ROS 2 actions, services and topics you list as A2A skills, for robot bring-up and debugging.

This is not the typed robot skill agent. `a2a_robots.RobotAdapter` gives planners one contract (`move_to`, `stop`, the busy rule, safety limits, one result shape) on any robot; this bridge passes configured ROS calls through and adds no safety limits of its own. Anything physical, such as joint limits or an emergency stop, stays in the robot's ROS stack.

The bridge talks to ROS only through the `RosGateway` interface. `RclpyGateway` implements it for a live ROS 2 graph.

## Allowlist

Each entry maps one skill to one ROS interface. Nothing else can be reached through the bridge.

```python
from a2a_robots_ros2 import Ros2BridgeAdapter, serve_bridge
from a2a_robots_ros2.rclpy_gateway import RclpyGateway

skills = [
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
serve_bridge(Ros2BridgeAdapter(skills, RclpyGateway()))  # http://127.0.0.1:9200/
```

| Field | Meaning |
| --- | --- |
| `id` | Skill name callers use. |
| `kind` | `action`, `service`, or `topic` (returns the latest message). |
| `ros_name` | Absolute ROS name. |
| `type` | Interface type, such as `turtlesim/action/RotateAbsolute`. |
| `description` | Optional. Shown in the Agent Card. |
| `timeout_s` | Optional. Bounds the whole call, including waiting for the server. Defaults: 60 s for actions, 10 s otherwise. |
| `exclusive` | Optional. Refuse a new call to this skill while one is still running. |

The allowlist is checked when the adapter is built, and the gateway loads every interface type then, so a typo or a missing message package fails at startup.

## Requests and results

Requests are JSON, sent as a text part or a data part. `args` holds the goal or request fields, with nested messages as nested objects; a topic skill takes none.

```json
{"skill": "rotate", "args": {"theta": 1.57}}
```

| Kind | Result |
| --- | --- |
| action | `{"skill": "rotate", "status": "succeeded", "result": {...}}`, where `status` is `succeeded`, `aborted` or `canceled` |
| service | `{"skill": "spawn", "response": {...}}` |
| topic | `{"skill": "pose", "message": {...}, "age_s": 0.04}`, where `age_s` is the time since the message arrived |

A goal that ran but did not succeed is a completed A2A task with its status, like a skill result with `success: false`. A malformed request, a rejected goal, a missing server or a timeout fails the task with the reason.

Feedback is streamed by default. The reply is then newline-delimited JSON in the response artifact: for an action an `accepted` line, then feedback lines (at most 2 per second; set `max_feedback_hz`; feedback that arrives before acceptance is held until the `accepted` line), then the result as the last line. Feedback appears in the response artifact, not as task status messages. With `stream_feedback=False` the reply is just the result, as JSON text and as an `application/json` data part.

## Cancel and timeouts

- `tasks/cancel` cancels the goal and waits for the action server's answer before the task is reported cancelled. A cancel that arrives while the goal is still being sent cancels it as soon as it is accepted. A timeout or shutdown cancels the goal too.
- If the server does not confirm the cancel within `cancel_timeout_s` (5 s by default), or the cancel fails, the bridge logs that the goal may still be running and tries again when the call stops and on shutdown. The call then waits for the goal to end, even past the skill's `timeout_s` and when the stream is closed. If the goal has not ended within a2a-adapter's cancel wait (10 s), `tasks/cancel` returns the task failed with that message instead of cancelled.
- ROS 2 services cannot be cancelled. Cancelling a service task, or its timeout, stops the wait, but the call may still complete on the robot.
- `serve_bridge()` binds `127.0.0.1` by default, because anyone who can reach the server can move the robot.

## ROS 2

rclpy is not on PyPI. It comes with a ROS 2 install and is built for that distribution's Python, so it is not a dependency of this package. ROS 2 Jazzy (Python 3.12) and newer are supported; Humble (Python 3.10) is below the Python 3.11 minimum.

To use `RclpyGateway`, source ROS 2 and use a virtual environment on the ROS Python that can see its packages:

```bash
source /opt/ros/jazzy/setup.bash
uv venv --system-site-packages --python /usr/bin/python3
uv pip install ./packages/a2a-robots-ros2  # from a checkout of this repository
```

`RclpyGateway(node_name="a2a_ros2_bridge")` creates one node in its own rclpy context and spins it on a background thread, so ROS callbacks never block the A2A server. The interface packages named in the allowlist (for example `turtlesim`) must be installed and sourced.

- Topics are read with the sensor-data QoS (best effort, keep last 5), which matches both reliable and best-effort publishers. A read returns the latest message, or waits for the first one.
- Args are set with `rosidl_runtime_py.set_message_fields`; a field that does not fit the message fails the task without calling ROS.
- The bridge waits for an action or service server within the skill's `timeout_s`.

## Try it with turtlesim

[`examples/turtlesim_bridge.py`](examples/turtlesim_bridge.py) allowlists turtlesim's `rotate_absolute` action, the `teleport_absolute` and `spawn` services and the pose topic. With ROS 2 Jazzy, `ros-jazzy-turtlesim` installed and the virtual environment from [ROS 2](#ros-2), run each in its own shell with ROS 2 sourced:

```bash
ros2 run turtlesim turtlesim_node
python packages/a2a-robots-ros2/examples/turtlesim_bridge.py  # http://127.0.0.1:9200/
```

The Agent Card at `http://127.0.0.1:9200/.well-known/agent-card.json` lists the four skills. Turn the turtle:

```bash
curl -s http://127.0.0.1:9200/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "m1", "parts": [
    {"kind": "text", "text": "{\"skill\": \"rotate\", \"args\": {\"theta\": 1.57}}"}]}}}' \
  | jq -j '.result.artifacts[0].parts[].text'
```

```text
{"skill": "rotate", "status": "accepted"}
{"skill": "rotate", "feedback": {"remaining": 1.5700000524520874}}
{"skill": "rotate", "feedback": {"remaining": 1.058000087738037}}
{"skill": "rotate", "feedback": {"remaining": 0.5460000038146973}}
{"skill": "rotate", "feedback": {"remaining": 0.03400003910064697}}
{"skill": "rotate", "status": "succeeded", "result": {"delta": -1.5520000457763672}}
```

Use a new `messageId` for each request. Other requests: `{"skill": "pose"}`, `{"skill": "teleport", "args": {"x": 2.0, "y": 3.0, "theta": 0.0}}` and `{"skill": "spawn", "args": {"x": 8.0, "y": 8.0, "theta": 0.0}}`. Stopping the server shuts the ROS node down.

## Tests

The bridge tests use a fake `RosGateway`, so they need no ROS install:

```bash
uv sync --all-packages --extra dev
uv run pytest packages/a2a-robots-ros2
```

The `RclpyGateway` tests run against a live graph and are skipped without rclpy. The turtlesim example test starts a headless `turtlesim_node` (`QT_QPA_PLATFORM=offscreen`) and drives it over JSON-RPC. CI runs both in the `ros:jazzy-ros-base` image with `ros-jazzy-example-interfaces` and `ros-jazzy-turtlesim` installed:

```bash
source /opt/ros/jazzy/setup.bash
export UV_PROJECT_ENVIRONMENT=/opt/a2a-venv
uv venv --system-site-packages --python /usr/bin/python3 "$UV_PROJECT_ENVIRONMENT"
uv sync --frozen --all-packages --extra dev
A2A_ROBOTS_REQUIRE_ROS=1 uv run pytest packages/a2a-robots-ros2
```
