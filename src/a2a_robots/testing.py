"""Conformance tests that every ``RobotBackend`` must pass.

A robot package subclasses ``BackendContract`` in its own tests and implements
``make_backend``; pytest then runs every check below against that backend::

    from a2a_robots.testing import BackendContract

    class TestMyRobotContract(BackendContract):
        def make_backend(self):
            return MyRobotBackend(...)

The checks call the backend directly, without the controller, so they show
whether the backend itself keeps the contract in ``a2a_robots.backend``:
observe is safe during motion, every motion method polls ``stop`` and returns
``STOPPED`` once it is set, and ``hold`` keeps the hands where they are.

Stops are requested after a fixed number of polls rather than after a delay,
so the checks do not depend on timing. This module does not import pytest.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

from .backend import MotionOutcome, RobotBackend
from .skills import FailureCode, Pose


class StopAfter(threading.Event):
    """A stop event that sets itself on the ``polls``-th time the backend
    checks it, as if ``stop`` arrived during the motion.

    Every ``is_set()`` and ``wait()`` call counts as a poll and may set the
    event, so a backend's own final ``stop.is_set()`` check (for example to
    tell a stop from another failure) can trigger the stop. Pass
    ``polls=math.inf`` to only count polls.
    """

    def __init__(self, polls: float, on_poll: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.polls = 0
        self._limit = polls
        self._on_poll = on_poll

    def _poll(self) -> None:
        self.polls += 1
        if self._on_poll is not None:
            self._on_poll()
        if self.polls >= self._limit:
            self.set()

    def is_set(self) -> bool:
        self._poll()
        return super().is_set()

    def wait(self, timeout: float | None = None) -> bool:
        self._poll()
        return super().wait(timeout)


class BackendContract:
    """Subclass as ``Test<Name>Contract`` and implement ``make_backend``.

    Override ``move_target`` if a small move along x is not reachable for the
    robot, and the tolerances if it settles less precisely.
    """

    frame = "world"
    position_tolerance_m = 0.005
    stop_polls = 3  # stop on the third poll, so the motion is under way
    observe_timeout_s = 1.0

    def make_backend(self) -> RobotBackend:
        raise NotImplementedError

    def move_target(self, backend: RobotBackend, hand: str) -> Pose:
        """A reachable target that takes more than one control step."""
        pose = backend.observe()[hand].pose
        x, y, z = pose.position
        return Pose(pose.frame, (x + 0.01, y, z), pose.orientation)

    # Helpers

    def _backend(self) -> RobotBackend:
        backend = self.make_backend()
        self._backends = [*getattr(self, "_backends", []), backend]
        return backend

    def teardown_method(self) -> None:
        for backend in getattr(self, "_backends", []):
            backend.shutdown()
        self._backends = []

    def _hand(self, backend: RobotBackend) -> str:
        return sorted(backend.hands)[0]

    def _motions(self, backend: RobotBackend, hand: str):
        target = self.move_target(backend, hand)
        return {
            "move_to": lambda stop: backend.move_to(hand, target, stop),
            "open": lambda stop: backend.open(hand, stop),
            "close": lambda stop: backend.close(hand, stop),
        }

    def _distance(self, a: Pose, b: Pose) -> float:
        return math.dist(a.position, b.position)

    # Checks

    def test_observe_reports_every_hand(self) -> None:
        backend = self._backend()
        assert backend.hands, "the backend has no hands"
        states = backend.observe()
        assert set(states) == set(backend.hands)
        for hand, state in states.items():
            assert state.pose.frame == self.frame, f"{hand}: frame {state.pose.frame!r}"
            norm = math.sqrt(sum(q * q for q in state.pose.orientation))
            assert abs(norm - 1) < 1e-3, f"{hand}: orientation is not a unit quaternion"
            assert isinstance(state.held, bool)
            assert all(isinstance(k, str) and v >= 0 for k, v in state.contact.items())

    def test_move_to_reaches_the_target(self) -> None:
        backend = self._backend()
        hand = self._hand(backend)
        target = self.move_target(backend, hand)
        outcome = backend.move_to(hand, target, threading.Event())
        assert outcome == MotionOutcome(), f"move_to failed: {outcome}"
        reached = backend.observe()[hand].pose
        assert self._distance(reached, target) <= self.position_tolerance_m

    def test_a_motion_stopped_before_it_starts_does_not_move(self) -> None:
        for name in ("move_to", "open", "close"):
            backend = self._backend()
            hand = self._hand(backend)
            before = backend.observe()[hand].pose
            stop = threading.Event()
            stop.set()
            outcome = self._motions(backend, hand)[name](stop)
            assert outcome.failure_code is FailureCode.STOPPED, f"{name}: {outcome}"
            after = backend.observe()[hand].pose
            assert self._distance(before, after) <= self.position_tolerance_m, name

    def test_a_motion_polls_stop_while_it_runs(self) -> None:
        """A full motion checks stop more than ``stop_polls`` times, so it does
        not only check at the start and the end."""
        for name in ("move_to", "open", "close"):
            backend = self._backend()
            counter = StopAfter(math.inf)
            self._motions(backend, self._hand(backend))[name](counter)
            assert counter.polls > self.stop_polls, (
                f"{name} polled stop {counter.polls} times; it must poll at every control step"
            )

    def test_a_motion_stopped_partway_returns_stopped(self) -> None:
        for name in ("move_to", "open", "close"):
            backend = self._backend()
            hand = self._hand(backend)
            target = self.move_target(backend, hand)
            stop = StopAfter(self.stop_polls)
            outcome = self._motions(backend, hand)[name](stop)
            assert outcome.failure_code is FailureCode.STOPPED, f"{name}: {outcome}"
            if name == "move_to":
                reached = backend.observe()[hand].pose
                assert self._distance(reached, target) > self.position_tolerance_m, (
                    "move_to reached the target after stop was set"
                )

    def test_observe_is_safe_during_motion(self) -> None:
        backend = self._backend()
        hand = self._hand(backend)
        observed = threading.Event()

        def observe_from_another_thread() -> None:
            if observed.is_set():
                return
            thread = threading.Thread(target=backend.observe, daemon=True)
            thread.start()
            thread.join(self.observe_timeout_s)
            assert not thread.is_alive(), "observe blocked while a motion was running"
            observed.set()

        stop = StopAfter(self.stop_polls, on_poll=observe_from_another_thread)
        backend.move_to(hand, self.move_target(backend, hand), stop)
        assert observed.is_set(), "move_to never polled stop"

    def test_hold_keeps_the_hands_where_they_are(self) -> None:
        backend = self._backend()
        hand = self._hand(backend)
        backend.move_to(hand, self.move_target(backend, hand), StopAfter(self.stop_polls))
        backend.hold()
        before = backend.observe()
        backend.hold()  # holding twice is harmless
        after = backend.observe()
        for name in backend.hands:
            assert self._distance(before[name].pose, after[name].pose) <= 1e-6, name

    def test_the_robot_moves_again_after_a_stop(self) -> None:
        backend = self._backend()
        hand = self._hand(backend)
        backend.move_to(hand, self.move_target(backend, hand), StopAfter(self.stop_polls))
        backend.hold()
        target = self.move_target(backend, hand)
        outcome = backend.move_to(hand, target, threading.Event())
        assert outcome == MotionOutcome(), f"move_to after a stop failed: {outcome}"
        reached = backend.observe()[hand].pose
        assert self._distance(reached, target) <= self.position_tolerance_m
