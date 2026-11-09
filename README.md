# a2a-robots

Expose a robot as an [A2A](https://a2a-protocol.org) agent. An agent plans the work and sends one task-level skill at a time; the adapter validates each request, enforces safety limits, runs the skill on the robot backend, and returns a fixed result format.

Built on [a2a-adapter](https://github.com/hybroai/a2a-adapter). Early development.

| Package | What it is |
| --- | --- |
| `a2a-robots` (this directory) | The core: skills, safety limits, the controller, the A2A adapter, and `FakeBackend`. |
| [`a2a-robots-sharpa`](packages/a2a-robots-sharpa) | Sharpa Wave hands, simulated in MuJoCo. |
| [`a2a-robots-ros2`](packages/a2a-robots-ros2) | An allowlisted ROS 2 bridge for bring-up and debugging. Not the typed skill contract. |

## Skills

Requests are JSON, sent as a text part or a data part:

```json
{"skill": "move_to", "args": {"hand": "right", "pose": {
  "frame": "world", "units": "m",
  "position": [0.30, -0.20, 0.15],
  "orientation": {"quaternion": [1, 0, 0, 0]}}}}
```

| Skill | Args | Notes |
| --- | --- | --- |
| `observe` | none | Read only. Always allowed, even during motion. |
| `move_to` | `hand`, `pose` | `frame`, `units` (`"m"`) and a unit quaternion are required. |
| `open` | `hand` | |
| `close` | `hand` | `held` reports whether something is now held. |
| `stop` | none | Stops motion now and freezes every hand. Always allowed. |

Every skill returns the same result, as JSON text and as an `application/json` data part:

```json
{"skill": "close", "success": false, "failure_code": "slipped",
 "detail": "object slipped during the grasp",
 "final_pose": {"right": {...}}, "contact": {"right": {}}, "held": {"right": false}}
```

`failure_code` is one of `step_too_large`, `out_of_workspace`, `busy`, `stopped`, `timeout`, `slipped`, `unreachable`. A skill that ran but did not succeed is a completed A2A task with `success: false`, so the agent can observe and retry. Only a malformed request fails the A2A task, and nothing moves.

## Safety

- A move larger than `max_step_m` or `max_step_rad`, or a target outside the workspace box, is refused before anything moves. The result includes the limits. Moves are never clamped.
- A step is measured from the hand's last target when its last skill was a successful `move_to`, because a robot settles within a small tolerance of the target, not exactly on it. After anything else (a failed move, a stop, a timeout, a cancel, or `open`/`close` on that hand) it is measured from the observed pose. So the hand moves at most `max_step_m` plus the backend's settle tolerance.
- One skill runs at a time per robot. A second motion request fails at once with `busy` instead of queueing.
- `stop`, `tasks/cancel`, timeouts and shutdown all stop the backend. Backends run in a worker thread and poll a stop flag every control step, so stopping does not depend on cancelling a Python task.
- `tasks/cancel` replies only after the motion has stopped and the hands are frozen, and the task is then canceled.
- If the motion does not stop within `stop_timeout_s` (2 s by default), the hands are still told to hold, but the stop, cancel or timed-out skill fails with "the robot did not stop ... it may still be moving", and a cancelled task ends failed rather than canceled. No new motion starts until that motion ends.
- If the backend raises during a motion, the hands are frozen and the task fails with the error.
- The adapter checks only the target pose. The path to it is the planner's job (see below); `MujocoPlanner` rejects paths that collide with the simulated scene. Real hardware also needs the robot's own safety checks.
- `serve_robot()` binds `127.0.0.1` by default, because anyone who can reach the server can move the robot.

## Quick start

```bash
uv sync --extra dev
uv run python examples/fake_robot.py   # serves on http://127.0.0.1:9100/
```

```bash
curl -s http://127.0.0.1:9100/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "1",
    "parts": [{"kind": "data", "data": {"skill": "observe"}}]}}}'
```

## Adding a robot

This repository is a monorepo. The shared package stays `a2a-robots`; each real robot is a sibling package named `a2a-robots-<name>` (for example `a2a-robots-sharpa`) that depends on it. `FakeBackend` stays in the core package only as a dependency-free test double.

### Planning

Arms such as Sharpa D01 take joint position targets, with planning and collision checking on the host. `a2a_robots.planning` defines that layer: a `Planner` turns a pose target into a `JointTrajectory` (one target per control step for the joints that carry the hand), and the backend streams it, checking the stop flag between targets. A planner reports "no safe path" by raising `PlanError`, which becomes a completed skill with `success: false`. `interpolate()` gives a speed-limited, smooth joint-space path for planners to build on. Any planner can be plugged in: `MujocoPlanner` today, a ROS 2 or Isaac planner later.

A robot package subclasses `RobotBackend` and implements `hands`, `observe`, `move_to`, `open`, `close`, `hold` and `shutdown`. Motion methods are synchronous, run in a worker thread, and must check the `stop` event at every control step and return `MotionOutcome(FailureCode.STOPPED)` once it is set. See `src/a2a_robots/fake.py` and `packages/a2a-robots-sharpa`.

Every backend must pass the shared conformance checks in `a2a_robots.testing`. Subclass `BackendContract` in the package's tests and return the backend from `make_backend()`:

```python
from a2a_robots.testing import BackendContract


class TestMyRobotContract(BackendContract):
    def make_backend(self):
        return MyRobotBackend(...)
```

They check that `observe` reports every hand and stays safe during motion, that `move_to`, `open` and `close` return `stopped` when `stop` is set before or during the motion, that `hold` keeps the hands still, and that the robot moves again after a stop. Override `move_target()` or the tolerances if the default small move does not suit the robot.

## Development

```bash
uv sync --all-packages --extra dev
uv run ruff format --check . && uv run ruff check .
uv run pytest
uv run pytest packages/a2a-robots-ros2
uv run pytest packages/a2a-robots-sharpa
```

## License

Apache-2.0
