# a2a-robots-pollen

[Pollen Robotics Reachy 2](https://github.com/pollen-robotics/reachy2-sdk) backend for [a2a-robots](../../README.md).

## Reachy 2 backend

`Reachy2Backend` (aliased as `PollenBackend`) exposes Reachy 2's dual 7-DoF arms (`l_arm`, `r_arm`) and parallel grippers through `a2a_robots.RobotBackend` and `a2a_robots.RobotAdapter`.

- Arm poses come from Reachy 2 forward kinematics in the robot's own coordinate system. The backend labels them with the configured `frame` (default `"world"`); on a mobile base that is still the robot frame, not a fixed room frame, so size the workspace box accordingly. Explicit joint positions from `arm.get_current_positions(degrees=False)` are passed into `arm.forward_kinematics(q_rad, degrees=False)` to avoid the `forward_kinematics(None, degrees=False)` degree-vs-radian and 8-actuator bug in `reachy2-sdk 1.0.15`.
- The default planner is `Reachy2Planner`, which calls `arm.inverse_kinematics(..., degrees=False)`, verifies forward-kinematics reachability within tolerance, rejects solutions whose per-joint travel exceeds `max_joint_delta_rad` (default 1.0 rad), and generates speed-limited minimum-jerk joint trajectories streamed via `arm.send_goal_positions(check_positions=False)`. Pass `planner=` to plug in another `a2a_robots.planning.Planner`.
- `open` and `close` stream `arm.gripper.set_opening(...)` in small increments and check the `stop` flag every control step (20 ms by default). Stalls during the first `startup_grace_s` (0.2 s) of `close` are ignored only while the opening is still at its start value (mid-range start plus brief latency). After that grace, or once the opening has moved, stalls count — including a re-close that is already on an object. If the gripper then stays stalled short of its target (`closed_opening_pct < opening < open_opening_pct`) for `grasp_settle_s` (0.2 s), `observe()` reports `held=True` and non-empty `contact` (`contact["finger"]` is a proxy estimate `(100 - opening) / 10`, since Reachy 2's `ParallelGripper` does not measure contact force). A stall that lasts at least `contact_stall_s` (0.1 s) but then closes past the object returns `FailureCode.SLIPPED`; shorter stalls, such as a late state update, are ignored. The gripper only reports its opening, so a lag lasting between the two durations that then catches up in empty air also reads as `slipped`.
- `held` stays true only while the measured opening remains within `grasp_opening_tol_pct` (5%) of the opening recorded when `close` settled. `hold()` cancels active SDK gotos (`cancel_all_goto()`) on every arm and gripper and locks the current joint angles. Hands that are still near that grasp opening are re-commanded to a closed gripper target (0%) so a stop while carrying does not open the squeeze; otherwise `hold()` keeps the measured opening (so a stopped mid-`open` does not slam shut).

## Usage

`a2a-robots-pollen` is an independent `uv` project with its own `uv.lock` and an optional `sdk` extra for `reachy2-sdk`. Offline tests and `FakeReachySDK` work out of the box without installing `reachy2-sdk`:

```bash
# Offline with the built-in test double (no reachy2-sdk):
REACHY2_FAKE=1 uv run --directory packages/a2a-robots-pollen python examples/serve_reachy2.py

# Live / simulated Reachy 2 (installs the `sdk` extra: reachy2-sdk):
uv run --directory packages/a2a-robots-pollen --extra sdk python examples/serve_reachy2.py
```

The offline fake path is what unit tests exercise by default. Integration coverage against the real `reachy2-sdk` classes (in-process gRPC over `FakeReachySDK`) runs with `--extra sdk`. A smoke test against the `pollenrobotics/reachy2` Docker simulator is available when `REACHY2_HOST` is set (see Tests).

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
# Offline unit & adapter tests against FakeReachySDK
uv run --directory packages/a2a-robots-pollen --extra dev pytest

# Integration tests against the real reachy2-sdk classes over an in-process gRPC server
A2A_ROBOTS_REQUIRE_REACHY2_SDK=1 uv run --directory packages/a2a-robots-pollen \
  --extra sdk --extra dev pytest tests/test_reachy2_sdk_grpc.py

# Smoke test against a running Pollen Reachy 2 Docker simulator (pollenrobotics/reachy2)
REACHY2_HOST=localhost A2A_ROBOTS_REQUIRE_REACHY2_SDK=1 uv run --directory packages/a2a-robots-pollen \
  --extra sdk --extra dev pytest tests/test_reachy2_sdk_grpc.py -k docker_sim
```

