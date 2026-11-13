import asyncio
import threading
from collections.abc import Callable

import numpy as np
import pytest

from a2a_robots import (
    FailureCode,
    Pose,
    RobotAdapter,
    RobotController,
    SafetyLimits,
    Workspace,
)
from a2a_robots.skills import parse_request
from a2a_robots_pollen import FakeReachySDK, PollenBackend, Reachy2Backend, connect


def _offset_pose(pose: Pose, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
    x, y, z = pose.position
    return Pose(pose.frame, (x + dx, y + dy, z + dz), pose.orientation)


def test_alias_and_observe_defaults():
    assert PollenBackend is Reachy2Backend
    backend = Reachy2Backend(FakeReachySDK(), realtime=False)
    obs = backend.observe()
    assert set(obs) == {"left", "right"}
    for hand in ("left", "right"):
        assert obs[hand].pose.frame == "world"
        assert obs[hand].contact == {}
        assert obs[hand].held is False


def test_constructor_validation():
    with pytest.raises(ValueError, match="at least one hand"):
        Reachy2Backend(FakeReachySDK(), hands=())

    with pytest.raises(ValueError, match="unsupported hand"):
        Reachy2Backend(FakeReachySDK(), hands=("center",))

    with pytest.raises(ValueError, match="no active arm"):
        Reachy2Backend(FakeReachySDK(has_left=False), hands=("left", "right"))

    no_gripper = FakeReachySDK()
    assert no_gripper.l_arm is not None
    no_gripper.l_arm.gripper = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no gripper"):
        Reachy2Backend(no_gripper)

    turn_on_fails = FakeReachySDK()
    assert turn_on_fails.r_arm is not None
    turn_on_fails.r_arm.turn_off()
    turn_on_fails.r_arm.turn_on = lambda: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="failed to turn on"):
        Reachy2Backend(turn_on_fails, turn_on=True)


def test_move_to_reaches_target_and_leaves_other_arm_still():
    backend = Reachy2Backend(FakeReachySDK(), realtime=False)
    before = backend.observe()
    target = _offset_pose(before["right"].pose, dx=0.015, dz=0.01)

    outcome = backend.move_to("right", target, threading.Event())

    assert outcome.failure_code is None
    after = backend.observe()
    assert after["right"].pose.position == pytest.approx(target.position, abs=0.005)
    assert after["left"].pose.position == pytest.approx(before["left"].pose.position, abs=1e-6)


def test_unreachable_target_does_not_move_arm_and_allows_recovery():
    backend = Reachy2Backend(FakeReachySDK(), realtime=False)
    before = backend.observe()["right"].pose

    failed = backend.move_to(
        "right", Pose("world", (5.0, 0.0, 0.0), before.orientation), threading.Event()
    )
    assert failed.failure_code is FailureCode.UNREACHABLE
    assert backend.observe()["right"].pose.position == pytest.approx(before.position, abs=1e-6)

    ok = backend.move_to("right", _offset_pose(before, dz=0.01), threading.Event())
    assert ok.failure_code is None


def test_stop_interrupts_realtime_motion():
    backend = Reachy2Backend(FakeReachySDK(), realtime=True)
    start_pose = backend.observe()["right"].pose
    target = _offset_pose(start_pose, dz=0.02)
    stop = threading.Event()
    threading.Timer(0.03, stop.set).start()

    outcome = backend.move_to("right", target, stop)

    assert outcome.failure_code is FailureCode.STOPPED
    reached = backend.observe()["right"].pose.position
    assert np.linalg.norm(np.array(reached) - np.array(target.position)) > 0.005


def test_close_empty_air_and_grasp_and_open_release():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=False)

    # Empty air close -> not held
    assert backend.close("left", threading.Event()).failure_code is None
    left_state = backend.observe()["left"]
    assert left_state.held is False
    assert left_state.contact == {}

    # Place an object at the right end-effector -> close grasps it
    right_pos = backend.observe()["right"].pose.position
    reachy.place_object("mug", right_pos, opening_pct=42.0)

    assert backend.close("right", threading.Event()).failure_code is None
    held_state = backend.observe()["right"]
    assert held_state.held is True
    assert held_state.contact["finger"] == pytest.approx(round((100.0 - 42.0) / 10.0, 4))

    # Open releases it
    assert backend.open("right", threading.Event()).failure_code is None
    released_state = backend.observe()["right"]
    assert released_state.held is False
    assert released_state.contact == {}


def test_close_reports_slipped_when_object_slips_out():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=False)
    right_pos = backend.observe()["right"].pose.position
    reachy.place_object("slick_block", right_pos, opening_pct=40.0, slip=True)

    outcome = backend.close("right", threading.Event())

    assert outcome.failure_code is FailureCode.SLIPPED
    assert "slipped" in outcome.detail
    assert backend.observe()["right"].held is False


class _StopAfter(threading.Event):
    """Set itself on the N-th poll, matching BackendContract's StopAfter."""

    def __init__(self, polls: int) -> None:
        super().__init__()
        self._polls = 0
        self._limit = polls

    def _poll(self) -> None:
        self._polls += 1
        if self._polls >= self._limit:
            self.set()

    def is_set(self) -> bool:
        self._poll()
        return super().is_set()

    def wait(self, timeout: float | None = None) -> bool:
        self._poll()
        return super().wait(timeout)


def test_stopped_close_does_not_report_held():
    backend = Reachy2Backend(FakeReachySDK(), realtime=False, min_gripper_steps=20)
    assert backend.open("right", threading.Event()).failure_code is None

    outcome = backend.close("right", _StopAfter(3))

    assert outcome.failure_code is FailureCode.STOPPED
    state = backend.observe()["right"]
    assert state.held is False
    assert state.contact == {}


def test_stopped_open_keeps_held_while_grasping():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=False, min_gripper_steps=20)
    right_pos = backend.observe()["right"].pose.position
    reachy.place_object("cube", right_pos, opening_pct=40.0)
    assert backend.close("right", threading.Event()).failure_code is None
    assert backend.observe()["right"].held is True

    outcome = backend.open("right", _StopAfter(3))

    assert outcome.failure_code is FailureCode.STOPPED
    state = backend.observe()["right"]
    assert state.held is True
    assert state.contact


def test_open_and_close_timeout_when_gripper_stuck():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=False)
    assert reachy.r_arm is not None
    reachy.r_arm.gripper.set_opening = lambda pct: None  # type: ignore[method-assign]

    # Stuck nearly closed: open never reaches the open threshold.
    reachy.r_arm.gripper.opening = 10.0
    opened = backend.open("right", threading.Event())
    assert opened.failure_code is FailureCode.TIMEOUT
    assert reachy.r_arm.gripper.opening == 10.0

    # Stuck fully open: close never reaches the closed threshold, and opening
    # stays at/above open_opening_pct so the stall is not mistaken for a grasp.
    reachy.r_arm.gripper.opening = 100.0
    closed = backend.close("right", threading.Event())
    assert closed.failure_code is FailureCode.TIMEOUT
    assert backend.observe()["right"].held is False


def test_hold_and_shutdown():
    reachy = FakeReachySDK()
    assert reachy.l_arm is not None and reachy.r_arm is not None
    backend = Reachy2Backend(reachy, realtime=False)
    before_l = reachy.l_arm.cancel_count
    before_r_gripper = reachy.r_arm.gripper.cancel_count

    backend.hold()
    assert reachy.l_arm.cancel_count == before_l + 1
    assert reachy.r_arm.gripper.cancel_count == before_r_gripper + 1

    assert reachy.is_connected() is True
    backend.shutdown()
    assert reachy.is_connected() is False


def test_forward_kinematics_always_passes_explicit_seven_radians():
    """Verify the reachy2-sdk 1.0.15 forward_kinematics(None) bug is bypassed."""
    recorded_calls: list[tuple[object, bool]] = []
    fake = FakeReachySDK()
    assert fake.r_arm is not None
    real_fk = fake.r_arm.forward_kinematics

    def checked_fk(joints_positions=None, degrees=True):
        recorded_calls.append((joints_positions, degrees))
        assert joints_positions is not None
        assert len(joints_positions) == 7
        assert degrees is False
        return real_fk(joints_positions, degrees=degrees)

    fake.r_arm.forward_kinematics = checked_fk  # type: ignore[method-assign]
    backend = Reachy2Backend(fake, realtime=False)
    pose = backend.observe()["right"].pose
    assert (
        backend.move_to("right", _offset_pose(pose, dz=0.01), threading.Event()).failure_code
        is None
    )
    assert recorded_calls


async def test_controller_and_adapter_end_to_end():
    reachy = FakeReachySDK()
    backend = Reachy2Backend(reachy, realtime=False)
    limits = SafetyLimits(
        workspace=Workspace("world", (-0.5, -0.8, -0.5), (0.8, 0.8, 0.8)),
        max_step_m=0.05,
        max_step_rad=0.25,
    )
    controller = RobotController(backend, limits)

    obs = await controller.run(parse_request({"skill": "observe"}, backend.hands), "t1")
    assert obs.success is True
    right_pose = obs.final_pose["right"]
    right_pos = tuple(right_pose["position"])
    reachy.place_object("cube", right_pos, opening_pct=35.0)

    closed = await controller.run(
        parse_request({"skill": "close", "args": {"hand": "right"}}, backend.hands),
        "t2",
    )
    assert closed.success is True
    assert closed.held["right"] is True

    moved = await controller.run(
        parse_request(
            {
                "skill": "move_to",
                "args": {
                    "hand": "right",
                    "pose": {
                        "frame": "world",
                        "units": "m",
                        "position": [
                            right_pos[0] + 0.01,
                            right_pos[1],
                            right_pos[2] + 0.01,
                        ],
                        "orientation": right_pose["orientation"],
                    },
                },
            },
            backend.hands,
        ),
        "t3",
    )
    assert moved.success is True

    opened = await controller.run(
        parse_request({"skill": "open", "args": {"hand": "right"}}, backend.hands),
        "t4",
    )
    assert opened.success is True
    assert opened.held["right"] is False
    await asyncio.wait_for(controller.aclose(), 1)

    # Smoke-check RobotAdapter metadata and cleanup
    adapter = RobotAdapter(backend, limits, name="Reachy 2 Test")
    assert adapter.get_metadata().name == "Reachy 2 Test"
    await asyncio.wait_for(adapter.close(), 1)


def test_disconnected_sdk_raises_connection_error(monkeypatch):
    disconnected = FakeReachySDK()
    disconnected.disconnect()
    with pytest.raises(ConnectionError, match="not connected"):
        Reachy2Backend(disconnected, realtime=False)

    monkeypatch.setattr("reachy2_sdk.ReachySDK", lambda host, **kw: disconnected)
    with pytest.raises(ConnectionError, match="failed to connect"):
        connect("unreachable-host")


def test_numeric_parameter_validation():
    with pytest.raises(ValueError, match="control_dt"):
        Reachy2Backend(FakeReachySDK(), control_dt=0.0, realtime=False)
    with pytest.raises(ValueError, match="settle_time_s"):
        Reachy2Backend(FakeReachySDK(), settle_time_s=-0.1, realtime=False)
    with pytest.raises(ValueError, match="position_tolerance_m"):
        Reachy2Backend(FakeReachySDK(), position_tolerance_m=0.0, realtime=False)
    with pytest.raises(ValueError, match="orientation_tolerance_rad"):
        Reachy2Backend(FakeReachySDK(), orientation_tolerance_rad=0.0, realtime=False)
    with pytest.raises(ValueError, match="gripper_step_pct"):
        Reachy2Backend(FakeReachySDK(), gripper_step_pct=0.0, realtime=False)
    with pytest.raises(ValueError, match="closed_opening_pct"):
        Reachy2Backend(
            FakeReachySDK(), closed_opening_pct=80.0, open_opening_pct=80.0, realtime=False
        )
    with pytest.raises(ValueError, match="contact_stall_s"):
        Reachy2Backend(FakeReachySDK(), contact_stall_s=0.0, realtime=False)
    with pytest.raises(ValueError, match="grasp_settle_s"):
        Reachy2Backend(FakeReachySDK(), contact_stall_s=0.2, grasp_settle_s=0.2, realtime=False)


def _stalling_gripper(reachy: FakeReachySDK, freeze: Callable[[int], bool]) -> None:
    """Make the right gripper ignore set_opening on the calls where ``freeze(n)`` is true,
    so its opening reads the same as the previous step (a lag or stale state update)."""
    assert reachy.r_arm is not None
    gripper = reachy.r_arm.gripper
    gripper.opening = 100.0
    real_set_opening = gripper.set_opening
    calls = {"n": 0}

    def set_opening(percentage: float) -> None:
        calls["n"] += 1
        if not freeze(calls["n"]):
            real_set_opening(percentage)

    gripper.set_opening = set_opening  # type: ignore[method-assign]


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_brief_mid_range_stall_in_empty_air_is_not_a_slip(frames):
    reachy = FakeReachySDK()
    _stalling_gripper(reachy, lambda n: 4 <= n < 4 + frames)
    backend = Reachy2Backend(reachy, realtime=False)

    outcome = backend.close("right", threading.Event())

    assert outcome.failure_code is None
    assert reachy.r_arm.gripper.opening <= 2.0
    assert backend.observe()["right"].held is False


def test_stale_state_every_other_step_is_not_a_slip():
    reachy = FakeReachySDK()
    _stalling_gripper(reachy, lambda n: n % 2 == 0)
    backend = Reachy2Backend(reachy, realtime=False)

    outcome = backend.close("right", threading.Event())

    assert outcome.failure_code is None
    assert backend.observe()["right"].held is False


def test_sustained_stall_that_then_closes_reads_as_slipped():
    """Past contact_stall_s the backend cannot tell an object slipping out from a lag
    catching up, because the gripper only reports its opening. Documented limit."""
    reachy = FakeReachySDK()
    _stalling_gripper(reachy, lambda n: 4 <= n < 12)
    backend = Reachy2Backend(reachy, realtime=False)

    assert backend.close("right", threading.Event()).failure_code is FailureCode.SLIPPED


def test_lagging_empty_air_gripper_finishes_closing_without_false_hold():
    """Physical grippers lag behind target_pct; closing in empty air must not look like a stall."""
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    gripper = reachy.r_arm.gripper
    gripper.opening = 100.0
    real_set_opening = gripper.set_opening

    def lagging_set_opening(percentage: float) -> None:
        # Advance at most 6% per control step toward the commanded target (less than 10% step).
        step = max(percentage, gripper.opening - 6.0)
        real_set_opening(step)

    gripper.set_opening = lagging_set_opening  # type: ignore[method-assign]
    backend = Reachy2Backend(reachy, realtime=False, settle_time_s=1.0)

    outcome = backend.close("right", threading.Event())

    assert outcome.failure_code is None
    assert gripper.opening <= 2.0
    state = backend.observe()["right"]
    assert state.held is False
    assert state.contact == {}


def test_gripper_encoder_overshoot_is_clamped():
    reachy = FakeReachySDK()
    assert reachy.r_arm is not None
    seen_targets: list[float] = []
    real_set_opening = reachy.r_arm.gripper.set_opening

    def recording_set_opening(percentage: float) -> None:
        assert 0.0 <= percentage <= 100.0
        seen_targets.append(percentage)
        real_set_opening(percentage)

    reachy.r_arm.gripper.set_opening = recording_set_opening  # type: ignore[method-assign]
    backend = Reachy2Backend(reachy, realtime=False)

    # Overshoot above 100% during hold() and close()
    reachy.r_arm.gripper.opening = 100.5
    backend.hold()
    assert seen_targets[-1] == 100.0
    assert backend.close("right", threading.Event()).failure_code is None

    # Undershoot below 0% during hold() and open()
    reachy.r_arm.gripper.opening = -0.5
    backend.hold()
    assert seen_targets[-1] == 0.0
    assert backend.open("right", threading.Event()).failure_code is None
