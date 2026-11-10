import asyncio
import threading

import pytest
from helpers import make_limits, pose

from a2a_robots import (
    FailureCode,
    FakeBackend,
    Pose,
    RequestError,
    RobotController,
    Skill,
    SkillCancelled,
    StopError,
)
from a2a_robots.skills import parse_request

HANDS = frozenset({"left", "right"})


def req(skill: str, **args):
    return parse_request({"skill": skill, "args": args}, HANDS)


def right_position(backend: FakeBackend):
    return backend.observe()["right"].pose.position


class BlockingHoldBackend(FakeBackend):
    """``hold`` blocks until the test releases it, like a slow real freeze."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hold_started = threading.Event()
        self.release_hold = threading.Event()

    def hold(self) -> None:
        self.hold_started.set()
        self.release_hold.wait(5)
        super().hold()


class IgnoresStopBackend(FakeBackend):
    """Keeps moving after stop until the test releases it, like a stuck SDK call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = threading.Event()

    def open(self, hand, stop):
        self.release.wait(5)
        return super().open(hand, threading.Event())


class FailingMoveBackend(FakeBackend):
    def open(self, hand, stop):
        raise RuntimeError("motor fault")


async def test_observe_reports_every_hand_in_the_result_schema(controller):
    result = (await controller.run(req("observe"), "t1")).to_dict()
    assert result["success"] is True
    assert result["failure_code"] is None
    assert set(result["final_pose"]) == {"left", "right"}
    assert result["final_pose"]["right"]["units"] == "m"
    assert result["held"] == {"left": False, "right": False}
    assert "limits" not in result


async def test_move_within_limits_reaches_the_target(controller, backend):
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.13)), "t1")
    assert result.success
    assert right_position(backend) == pytest.approx((0.3, 0.1, 0.13))


async def test_oversized_move_is_refused_before_anything_moves(controller, backend):
    before = right_position(backend)
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.3)), "t1")
    assert not result.success
    assert result.failure_code is FailureCode.STEP_TOO_LARGE
    assert result.limits == {"max_step_m": 0.05, "max_step_rad": 0.1}
    assert right_position(backend) == before


async def test_oversized_rotation_is_refused(controller):
    quarter_turn = (0.7071068, 0.7071068, 0, 0)
    result = await controller.run(
        req("move_to", hand="right", pose=pose(0.3, 0.1, 0.1, quarter_turn)), "t1"
    )
    assert result.failure_code is FailureCode.STEP_TOO_LARGE


async def test_target_outside_workspace_is_refused():
    controller = RobotController(FakeBackend(), make_limits(max_step_m=10))
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.9)), "t1")
    assert result.failure_code is FailureCode.OUT_OF_WORKSPACE


async def test_wrong_frame_is_a_request_error(controller):
    with pytest.raises(RequestError, match='frame must be "world"'):
        await controller.run(
            req("move_to", hand="right", pose=pose(0.3, 0.1, 0.1, frame="tool")), "t1"
        )


async def test_second_motion_fails_fast_with_busy_while_observe_still_works():
    backend = FakeBackend(steps=50, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    first = asyncio.create_task(controller.run(req("open", hand="left"), "t1"))
    await asyncio.sleep(0.05)

    busy = await asyncio.wait_for(controller.run(req("close", hand="right"), "t2"), 0.2)
    observed = await asyncio.wait_for(controller.run(req("observe"), "t3"), 0.2)

    assert busy.failure_code is FailureCode.BUSY
    assert observed.success
    assert (await first).success


async def test_stop_preempts_a_running_move_and_freezes_the_pose():
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.1)

    stopped = await asyncio.wait_for(controller.run(req("stop"), "t2"), 0.5)
    moved = await move

    assert stopped.success
    assert moved.failure_code is FailureCode.STOPPED
    assert 0.1 < right_position(backend)[2] < 0.14
    assert backend.hold_calls == 1
    # The robot is free again after a stop.
    assert (await controller.run(req("observe"), "t3")).success
    assert (await controller.run(req("open", hand="right"), "t4")).success


async def test_timeout_stops_the_hand_instead_of_leaving_it_moving():
    backend = FakeBackend(steps=1000, step_time_s=0.01)
    controller = RobotController(backend, make_limits(skill_timeout_s=0.1))
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    z = right_position(backend)[2]
    await asyncio.sleep(0.05)

    assert result.failure_code is FailureCode.TIMEOUT
    assert right_position(backend)[2] == z
    assert backend.hold_calls == 1


async def test_cancel_returns_once_the_hands_are_frozen():
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.05)

    await controller.cancel("t1")
    assert backend.hold_calls == 1
    z = right_position(backend)[2]
    with pytest.raises(SkillCancelled):
        await move
    await asyncio.sleep(0.05)

    assert right_position(backend)[2] == z
    assert (await controller.run(req("open", hand="right"), "t2")).success


async def test_cancelling_the_task_before_cancel_still_stops_the_backend_thread():
    """Some a2a-sdk versions cancel the running coroutine before calling cancel()."""
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.05)
    move.cancel()
    await controller.cancel("t1")
    with pytest.raises(asyncio.CancelledError):
        await move
    z = right_position(backend)[2]
    await asyncio.sleep(0.05)

    assert right_position(backend)[2] == z
    assert backend.hold_calls >= 1
    assert (await controller.run(req("open", hand="right"), "t2")).success


async def test_a_stop_the_backend_ignores_is_reported_and_blocks_new_motion():
    backend = IgnoresStopBackend()
    controller = RobotController(backend, make_limits(stop_timeout_s=0.05))
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.02)

    with pytest.raises(StopError, match="did not stop within 0.05 s"):
        await controller.run(req("stop"), "t2")
    assert backend.hold_calls == 1  # the hands are still told to hold
    busy = await controller.run(req("close", hand="left"), "t3")
    assert busy.failure_code is FailureCode.BUSY

    backend.release.set()
    await move
    assert (await controller.run(req("close", hand="left"), "t4")).success


async def test_a_cancel_the_backend_ignores_raises_stop_error():
    backend = IgnoresStopBackend()
    controller = RobotController(backend, make_limits(stop_timeout_s=0.05))
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.02)

    with pytest.raises(StopError):
        await controller.cancel("t1")
    backend.release.set()
    with pytest.raises(SkillCancelled):
        await move


async def test_a_cancelled_run_ends_only_after_the_motion_does():
    """a2a-adapter reports a failed cancel as canceled once the run has ended,
    so the run must outlast the motion, even past the skill timeout."""
    backend = IgnoresStopBackend()
    limits = make_limits(skill_timeout_s=0.1, stop_timeout_s=0.05)
    controller = RobotController(backend, limits)
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.02)

    with pytest.raises(StopError):
        await controller.cancel("t1")
    await asyncio.sleep(0.3)
    assert not move.done()

    backend.release.set()
    with pytest.raises(SkillCancelled):
        await asyncio.wait_for(move, 1)


async def test_a_timeout_the_backend_ignores_fails_the_skill():
    backend = IgnoresStopBackend(steps=2)
    controller = RobotController(backend, make_limits(skill_timeout_s=0.05, stop_timeout_s=0.05))
    with pytest.raises(StopError):
        await controller.run(req("open", hand="right"), "t1")
    # The timed-out run has ended, but its motion thread is still going.
    busy = await controller.run(req("close", hand="left"), "t2")
    assert busy.failure_code is FailureCode.BUSY
    assert busy.detail == "the last motion has not stopped"

    backend.release.set()
    await asyncio.sleep(0.1)  # the released motion finishes its steps
    assert (await controller.run(req("close", hand="left"), "t3")).success


async def test_a_backend_error_freezes_the_hands():
    backend = FailingMoveBackend()
    controller = RobotController(backend, make_limits())
    with pytest.raises(RuntimeError, match="motor fault"):
        await controller.run(req("open", hand="right"), "t1")
    assert backend.hold_calls == 1
    assert (await controller.run(req("close", hand="right"), "t2")).success


async def test_no_motion_starts_while_an_idle_stop_is_freezing():
    backend = BlockingHoldBackend()
    controller = RobotController(backend, make_limits())
    stop = asyncio.create_task(controller.run(req("stop"), "t1"))
    assert await asyncio.to_thread(backend.hold_started.wait, 1)
    before = right_position(backend)

    busy = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.13)), "t2")

    assert busy.failure_code is FailureCode.BUSY
    assert right_position(backend) == before
    backend.release_hold.set()
    assert (await stop).success
    assert (await controller.run(req("open", hand="right"), "t3")).success


async def test_no_motion_starts_after_a_preempted_motion_until_the_freeze_ends():
    backend = BlockingHoldBackend(steps=100)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.05)
    stop = asyncio.create_task(controller.run(req("stop"), "t2"))
    assert await asyncio.to_thread(backend.hold_started.wait, 1)

    # The stopped motion has released its busy flag, but hold() is still running.
    assert (await move).failure_code is FailureCode.STOPPED
    busy = await controller.run(req("open", hand="right"), "t3")

    assert busy.failure_code is FailureCode.BUSY
    backend.release_hold.set()
    assert (await stop).success
    assert (await controller.run(req("open", hand="right"), "t4")).success


async def test_cancel_without_a_task_id_stops_a_motion_started_without_one():
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(controller.run(req("open", hand="right"), None))
    await asyncio.sleep(0.05)
    await controller.cancel(None)
    with pytest.raises(SkillCancelled):
        await asyncio.wait_for(move, 0.5)


class SettlesShortBackend(FakeBackend):
    """Like a real controller, stops a little short of each target (0.6 mm in x)."""

    def move_to(self, hand, target, stop):
        outcome = super().move_to(hand, target, stop)
        if outcome.failure_code is None:
            with self._lock:
                p = self._state[hand].pose
                x, y, z = p.position
                self._state[hand].pose = Pose(p.frame, (x - 0.0006, y, z), p.orientation)
        return outcome


def step_x(dx: float, start: float = 0.3):
    return req("move_to", hand="right", pose=pose(start + dx, 0.1, 0.1))


async def test_full_step_after_a_successful_move_is_measured_from_its_target():
    backend = SettlesShortBackend()
    controller = RobotController(backend, make_limits())
    assert (await controller.run(step_x(0.04), "t1")).success

    # 0.0499 m from the last target, but 0.0505 m from where the hand stopped.
    result = await controller.run(step_x(0.0899), "t2")

    assert result.success
    assert right_position(backend)[0] == pytest.approx(0.3893)


async def test_steps_are_measured_from_the_observed_pose_after_a_stop():
    backend = SettlesShortBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(controller.run(step_x(0.05), "t1"))
    await asyncio.sleep(0.1)
    await controller.run(req("stop"), "t2")
    assert (await move).failure_code is FailureCode.STOPPED

    # Within one step of the stopped move's target, but not of where the hand is.
    result = await controller.run(step_x(0.09), "t3")

    assert result.failure_code is FailureCode.STEP_TOO_LARGE


@pytest.mark.parametrize("skill", ["open", "close"])
async def test_steps_are_measured_from_the_observed_pose_after_the_fingers_move(skill):
    backend = SettlesShortBackend()
    controller = RobotController(backend, make_limits())
    assert (await controller.run(step_x(0.04), "t1")).success
    assert (await controller.run(req(skill, hand="right"), "t2")).success

    result = await controller.run(step_x(0.0899), "t3")

    assert result.failure_code is FailureCode.STEP_TOO_LARGE


async def test_a_refused_move_keeps_the_last_target():
    backend = SettlesShortBackend()
    controller = RobotController(backend, make_limits())
    assert (await controller.run(step_x(0.04), "t1")).success
    assert (await controller.run(step_x(0.2), "t2")).failure_code is FailureCode.STEP_TOO_LARGE
    assert (await controller.run(step_x(0.0899), "t3")).success


async def test_the_last_target_is_per_hand():
    backend = SettlesShortBackend()
    controller = RobotController(backend, make_limits())
    assert (await controller.run(step_x(0.04), "t1")).success
    left = backend.observe()["left"].pose.position
    far_left = req("move_to", hand="left", pose=pose(left[0] + 0.0506, left[1], left[2]))
    assert (await controller.run(far_left, "t2")).failure_code is FailureCode.STEP_TOO_LARGE


async def test_cancel_for_another_task_does_not_stop_the_motion():
    backend = FakeBackend(steps=20, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.03)
    await controller.cancel("other")
    assert (await move).success


async def test_grasp_reports_held_and_contact(controller):
    result = await controller.run(req("close", hand="right"), "t1")
    assert result.success
    assert result.held["right"] is True
    assert result.contact["right"]
    released = await controller.run(req("open", hand="right"), "t2")
    assert released.held["right"] is False


async def test_slip_is_a_completed_result_with_success_false():
    controller = RobotController(
        FakeBackend(reachable_objects={"right"}, slipping={"right"}), make_limits()
    )
    result = await controller.run(req("close", hand="right"), "t1")
    assert result.skill is Skill.CLOSE
    assert not result.success
    assert result.failure_code is FailureCode.SLIPPED
    assert result.held["right"] is False


async def test_aclose_stops_and_shuts_down_the_backend(controller, backend):
    await controller.aclose()
    assert backend.shut_down
    assert backend.hold_calls == 1


async def test_aclose_shuts_down_even_when_the_stop_times_out():
    backend = IgnoresStopBackend()
    controller = RobotController(backend, make_limits(stop_timeout_s=0.05))
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.02)
    with pytest.raises(StopError):
        await controller.aclose()
    assert backend.shut_down
    backend.release.set()
    await move
