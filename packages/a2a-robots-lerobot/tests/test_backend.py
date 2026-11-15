import asyncio
import sys
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
from a2a_robots_lerobot import (
    ArmSpec,
    FakeLeRobot,
    LeRobotArmBackend,
    PlacoKinematics,
    koch_arms,
    so101_arms,
)


def _offset_pose(pose: Pose, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
    x, y, z = pose.position
    return Pose(pose.frame, (x + dx, y + dy, z + dz), pose.orientation)


class _StopAfter(threading.Event):
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


def test_observe_defaults_and_camera_passthrough():
    robot = FakeLeRobot(bimanual=True, with_camera=True)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    obs = backend.observe()
    assert set(obs) == {"left", "right"}
    for hand in ("left", "right"):
        assert obs[hand].pose.frame == "world"
        assert obs[hand].contact == {}
        assert obs[hand].held is False


def test_constructor_validation_and_safety_guards():
    robot = FakeLeRobot(bimanual=False)
    with pytest.raises(ValueError, match="at least one hand"):
        LeRobotArmBackend(robot, {}, solvers=robot.kinematics)

    with pytest.raises(ValueError, match="use_degrees=True"):
        bad_deg = FakeLeRobot(bimanual=False, use_degrees=False)
        LeRobotArmBackend(bad_deg, so101_arms(), solvers=bad_deg.kinematics)

    # Disconnected robot fails unless connect=True
    disconnected = FakeLeRobot(bimanual=False, connected=False, calibrated=True)
    with pytest.raises(ConnectionError, match="not connected"):
        LeRobotArmBackend(disconnected, disconnected.arms, solvers=disconnected.kinematics)

    # connect=True connects with calibrate=False so interactive prompts never block
    backend = LeRobotArmBackend(
        disconnected,
        disconnected.arms,
        solvers=disconnected.kinematics,
        connect=True,
        realtime=False,
    )
    assert disconnected.connect_calls == [False]
    assert disconnected.calibrate_calls == 0
    backend.shutdown()
    assert disconnected.is_connected is False

    # Uncalibrated robot disconnects safely and raises
    uncalibrated = FakeLeRobot(bimanual=False, connected=False, calibrated=False)
    with pytest.raises(RuntimeError, match="not calibrated"):
        LeRobotArmBackend(
            uncalibrated,
            uncalibrated.arms,
            solvers=uncalibrated.kinematics,
            connect=True,
        )
    assert uncalibrated.connect_calls == [False]
    assert uncalibrated.calibrate_calls == 0
    assert uncalibrated.is_connected is False

    # Missing action/observation keys fail fast
    bad_spec = {"right": ArmSpec(arm_joints=("missing_joint",))}
    with pytest.raises(ValueError, match="action key"):
        LeRobotArmBackend(robot, bad_spec, solvers=robot.kinematics)


def test_numeric_parameter_validation():
    robot = FakeLeRobot()
    with pytest.raises(ValueError, match="control_dt"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, control_dt=0.0)
    with pytest.raises(ValueError, match="settle_time_s"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, settle_time_s=-0.1)
    with pytest.raises(ValueError, match="position_tolerance_m"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, position_tolerance_m=0.0)
    with pytest.raises(ValueError, match="orientation_tolerance_rad"):
        LeRobotArmBackend(
            robot, robot.arms, solvers=robot.kinematics, orientation_tolerance_rad=0.0
        )
    with pytest.raises(ValueError, match="contact_stall_s"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, contact_stall_s=0.0)
    with pytest.raises(ValueError, match="grasp_settle_s"):
        LeRobotArmBackend(
            robot,
            robot.arms,
            solvers=robot.kinematics,
            contact_stall_s=0.2,
            grasp_settle_s=0.2,
        )
    with pytest.raises(ValueError, match="startup_grace_s"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, startup_grace_s=-0.1)
    with pytest.raises(ValueError, match="grasp_opening_tol_pct"):
        LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, grasp_opening_tol_pct=-1.0)


def test_bimanual_move_to_sends_full_action_dict_and_holds_other_arm():
    robot = FakeLeRobot(bimanual=True)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    before = backend.observe()
    target = _offset_pose(before["right"].pose, dx=0.015, dz=0.01)

    outcome = backend.move_to("right", target, threading.Event())

    assert outcome.failure_code is None
    assert robot.sent_actions
    assert set(robot.sent_actions[-1]) == set(robot.action_features)
    after = backend.observe()
    assert after["right"].pose.position == pytest.approx(target.position, abs=0.005)
    assert after["left"].pose.position == pytest.approx(before["left"].pose.position, abs=1e-6)


def test_unreachable_target_does_not_move_arm_and_recovers():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    before = backend.observe()["right"].pose

    failed = backend.move_to(
        "right", Pose("world", (5.0, 0.0, 0.0), before.orientation), threading.Event()
    )
    assert failed.failure_code is FailureCode.UNREACHABLE
    assert backend.observe()["right"].pose.position == pytest.approx(before.position, abs=1e-6)

    ok = backend.move_to("right", _offset_pose(before, dz=0.01), threading.Event())
    assert ok.failure_code is None


def test_stop_interrupts_realtime_motion():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=True)
    start_pose = backend.observe()["right"].pose
    target = _offset_pose(start_pose, dz=0.02)
    stop = threading.Event()
    threading.Timer(0.03, stop.set).start()

    outcome = backend.move_to("right", target, stop)

    assert outcome.failure_code is FailureCode.STOPPED
    reached = backend.observe()["right"].pose.position
    assert np.linalg.norm(np.array(reached) - np.array(target.position)) > 0.005


def test_close_empty_air_grasp_and_open_release():
    robot = FakeLeRobot(bimanual=True)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)

    assert backend.close("left", threading.Event()).failure_code is None
    left_state = backend.observe()["left"]
    assert left_state.held is False
    assert left_state.contact == {}

    right_pos = backend.observe()["right"].pose.position
    robot.place_object("mug", right_pos, opening_pct=42.0)

    assert backend.close("right", threading.Event()).failure_code is None
    held_state = backend.observe()["right"]
    assert held_state.held is True
    assert held_state.contact["finger"] == pytest.approx(round((100.0 - 42.0) / 10.0, 4))

    assert backend.open("right", threading.Event()).failure_code is None
    released_state = backend.observe()["right"]
    assert released_state.held is False
    assert released_state.contact == {}


def test_close_reports_slipped_when_object_slips_out():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    right_pos = backend.observe()["right"].pose.position
    robot.place_object("slick_block", right_pos, opening_pct=40.0, slip=True)

    outcome = backend.close("right", threading.Event())

    assert outcome.failure_code is FailureCode.SLIPPED
    assert "slipped" in (outcome.detail or "")
    assert backend.observe()["right"].held is False


def test_stopped_open_clears_held_and_hold_does_not_slam_shut():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(
        robot, robot.arms, solvers=robot.kinematics, realtime=False, min_gripper_steps=20
    )
    right_pos = backend.observe()["right"].pose.position
    robot.place_object("cube", right_pos, opening_pct=40.0)
    assert backend.close("right", threading.Event()).failure_code is None
    assert backend.observe()["right"].held is True

    outcome = backend.open("right", _StopAfter(3))
    assert outcome.failure_code is FailureCode.STOPPED
    measured = robot.get_observation()["gripper.pos"]
    assert measured > 40.0
    assert backend.observe()["right"].held is False

    backend.hold()
    assert robot.goal_positions["gripper.pos"] == pytest.approx(measured, abs=0.5)


def test_hold_keeps_commanded_squeeze_while_grasping():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    right_pos = backend.observe()["right"].pose.position
    robot.place_object("cube", right_pos, opening_pct=40.0)
    assert backend.close("right", threading.Event()).failure_code is None
    assert backend.observe()["right"].held is True

    backend.hold()

    assert robot.goal_positions["gripper.pos"] == 0.0
    assert backend.observe()["right"].held is True


def test_hold_uses_robot_command_hold_when_supported():
    robot = FakeLeRobot(bimanual=False)
    hold_calls = {"n": 0}
    robot.supports_command_hold = True
    robot.hold = lambda: hold_calls.__setitem__("n", hold_calls["n"] + 1)  # type: ignore[attr-defined]

    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    sent_before = len(robot.sent_actions)
    backend.hold()

    assert hold_calls["n"] == 1
    assert len(robot.sent_actions) == sent_before


def _stalling_gripper(robot: FakeLeRobot, freeze: Callable[[int], bool]) -> None:
    robot._state["gripper.pos"] = 100.0
    real_send_action = robot.send_action
    calls = {"n": 0}

    def wrapped_send(action: dict[str, float]) -> dict[str, float]:
        calls["n"] += 1
        if freeze(calls["n"]):
            copied = dict(action)
            copied["gripper.pos"] = robot._state["gripper.pos"]
            return real_send_action(copied)
        return real_send_action(action)

    robot.send_action = wrapped_send  # type: ignore[method-assign]


def test_lagging_empty_air_gripper_does_not_false_grasp():
    robot = FakeLeRobot(bimanual=False)
    _stalling_gripper(robot, lambda n: n <= 2)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)

    outcome = backend.close("right", threading.Event())
    assert outcome.failure_code is None
    assert backend.observe()["right"].held is False


def test_gripperless_arm_returns_unreachable_on_open_and_close():
    robot = FakeLeRobot(bimanual=False)
    spec = robot.arms["right"]
    arms = {
        "right": ArmSpec(
            arm_joints=spec.arm_joints,
            action_keys=spec.action_keys,
            observation_keys=spec.observation_keys,
            gripper_action_key=None,
            check_orientation=False,
        )
    }
    backend = LeRobotArmBackend(robot, arms, solvers=robot.kinematics, realtime=False)
    assert backend.open("right", threading.Event()).failure_code is FailureCode.UNREACHABLE
    assert backend.close("right", threading.Event()).failure_code is FailureCode.UNREACHABLE
    assert backend.open("right", _StopAfter(1)).failure_code is FailureCode.STOPPED


def test_placo_kinematics_missing_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "lerobot.model", None)
    monkeypatch.setitem(sys.modules, "lerobot.model.kinematics", None)
    with pytest.raises(ModuleNotFoundError, match="lerobot\\[kinematics\\]"):
        PlacoKinematics("dummy.urdf")


def test_placo_kinematics_missing_placo_error(monkeypatch):
    from types import SimpleNamespace

    def fake_robot_kinematics(**_kwargs):
        raise ImportError("'placo' is required")

    monkeypatch.setitem(sys.modules, "lerobot", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lerobot.model", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "lerobot.model.kinematics",
        SimpleNamespace(RobotKinematics=fake_robot_kinematics),
    )
    with pytest.raises(ModuleNotFoundError, match="lerobot\\[kinematics\\]"):
        PlacoKinematics("dummy.urdf")


def test_koch_arms_preset_matches_fake_keys():
    robot = FakeLeRobot(bimanual=False)
    arms = koch_arms()
    backend = LeRobotArmBackend(robot, arms, solvers=robot.kinematics, realtime=False)
    assert backend.hands == frozenset({"right"})


async def test_controller_and_adapter_end_to_end():
    robot = FakeLeRobot(bimanual=False)
    backend = LeRobotArmBackend(robot, robot.arms, solvers=robot.kinematics, realtime=False)
    limits = SafetyLimits(
        workspace=Workspace("world", (-0.5, -0.5, -0.2), (0.6, 0.5, 0.6)),
        max_step_m=0.05,
        max_step_rad=0.25,
    )
    controller = RobotController(backend, limits)

    obs = await controller.run(parse_request({"skill": "observe"}, backend.hands), "t1")
    assert obs.success is True
    right_pose = obs.final_pose["right"]
    right_pos = tuple(right_pose["position"])
    robot.place_object("cube", right_pos, opening_pct=35.0)

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

    # Smoke-check RobotAdapter metadata
    adapter = RobotAdapter(
        LeRobotArmBackend(FakeLeRobot(), FakeLeRobot().arms, solvers=FakeLeRobot().kinematics),
        limits,
        name="LeRobot Test",
    )
    assert adapter.get_metadata().name == "LeRobot Test"
    await asyncio.wait_for(adapter.close(), 1)
