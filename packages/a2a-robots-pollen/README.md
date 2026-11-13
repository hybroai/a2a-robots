# a2a-robots-pollen

[Pollen Robotics Reachy 2](https://github.com/pollen-robotics/reachy2-sdk) backend for [a2a-robots](../../README.md).

## Reachy 2 backend

`Reachy2Backend` (aliased as `PollenBackend`) exposes Reachy 2's dual 7-DoF arms (`l_arm`, `r_arm`) and parallel grippers through `a2a_robots.RobotBackend` and `a2a_robots.RobotAdapter`.

- Arm poses are reported in the configured `frame` (default `"world"`) via `arm.forward_kinematics(q_rad, degrees=False)` using explicit joint positions from `arm.get_current_positions(degrees=False)` to avoid the `forward_kinematics(None)` actuator-count bug in `reachy2-sdk 1.0.15`.
- The default planner is `Reachy2Planner`, which calls `arm.inverse_kinematics(..., degrees=False)`, verifies forward-kinematics reachability within tolerance, and generates speed-limited minimum-jerk joint trajectories streamed via `arm.send_goal_positions(check_positions=False)`. Pass `planner=` to plug in another `a2a_robots.planning.Planner`.
- `open` and `close` stream `arm.gripper.set_opening(...)` in small increments and check the `stop` flag every control step (20 ms by default). If the gripper stays stalled short of its target (`closed_opening_pct < opening < open_opening_pct`) for `grasp_settle_s` (0.2 s), `observe()` reports `held=True` and non-empty `contact`. A stall that lasts at least `contact_stall_s` (0.1 s) but then closes past the object returns `FailureCode.SLIPPED`; shorter stalls, such as a late state update, are ignored. The gripper only reports its opening, so a lag lasting between the two durations that then catches up in empty air also reads as `slipped`.
- `hold()` cancels active SDK gotos (`cancel_all_goto()`) on every arm and gripper and locks the current joint angles and gripper openings.

## Usage

Offline tests and `FakeReachySDK` work out of the box. To connect to a live or simulated Reachy 2 gRPC server, also install `reachy2-sdk` (kept out of the workspace lockfile because its upstream metadata pins older `numpy`, `protobuf`, and `grpcio` versions):

```bash
uv sync --all-packages --extra dev

# Offline with the built-in test double:
REACHY2_FAKE=1 uv run python packages/a2a-robots-pollen/examples/serve_reachy2.py

# Live / simulated Reachy 2 gRPC server (with reachy2-sdk installed):
uv pip install "reachy2-sdk>=1.0.15,<2"
REACHY2_HOST=localhost uv run python packages/a2a-robots-pollen/examples/serve_reachy2.py
```

Or in Python:

```python
from a2a_robots import RobotAdapter, SafetyLimits, Workspace, serve_robot
from a2a_robots_pollen import Reachy2Backend, connect

backend = Reachy2Backend(connect("localhost"))
limits = SafetyLimits(workspace=Workspace("world", (-0.5, -0.8, -0.5), (0.8, 0.8, 0.8)))
adapter = RobotAdapter(backend, limits, name="Reachy 2")
serve_robot(adapter, port=9100)
```

## Tests

```bash
uv run pytest packages/a2a-robots-pollen
```

