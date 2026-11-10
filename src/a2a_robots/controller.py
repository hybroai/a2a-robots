"""Runs skills on one robot: busy flag, preemption, timeouts, results.

One controller owns one backend, so every chat served by the process shares
the same busy flag and two chats cannot move the same robot at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading

from a2a_adapter import UserFacingError

from .backend import MotionOutcome, RobotBackend
from .limits import SafetyLimits
from .skills import FailureCode, Pose, RequestError, Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)


class StopError(UserFacingError):
    """The backend did not stop in time, so the robot may still be moving."""


class SkillCancelled(Exception):
    """The motion was stopped by ``cancel`` for its own task."""


class RobotController:
    def __init__(self, backend: RobotBackend, limits: SafetyLimits) -> None:
        self.backend = backend
        self.limits = limits
        self._stop = threading.Event()
        self._busy_task: str | None = None
        self._stopping = 0
        self._worker: asyncio.Future[MotionOutcome] | None = None
        self._cancelled: set[str] = set()
        # Target of each hand's last successful move. A backend stops within its
        # settle tolerance, so the next step is measured from here instead of
        # the observed pose. Cleared by anything that may leave the hand elsewhere.
        self._commanded: dict[str, Pose] = {}

    async def run(self, request: SkillRequest, task_id: str | None) -> SkillResult:
        if request.skill is Skill.OBSERVE:
            return await self._result(request.skill)
        if request.skill is Skill.STOP:
            await self.stop()
            return await self._result(request.skill)
        return await self._run_motion(request, task_id)

    async def stop(self) -> None:
        """Stop any motion now and freeze every hand where it is.

        No motion can start until the hands are frozen, including in the gap
        after a stopped motion releases the busy flag and before ``hold`` ends.
        Raises ``StopError`` if the motion does not stop within
        ``stop_timeout_s``; the hands are still told to hold.
        """
        self._stopping += 1
        try:
            self._stop.set()
            stopped = await self._wait_for_worker()
            await asyncio.to_thread(self.backend.hold)
            self._commanded.clear()
        finally:
            self._stopping -= 1
        if not stopped:
            raise StopError(
                f"the robot did not stop within {self.limits.stop_timeout_s} s; "
                "it may still be moving"
            )

    async def cancel(self, task_id: str | None) -> None:
        """Stop the motion started by ``task_id``, if it is the one running, and
        return once the hands are frozen. That run then raises ``SkillCancelled``.
        Raises ``StopError`` if the robot did not stop in time."""
        # A run without a task id is stored as "", so None matches it here.
        key = task_id or ""
        if key != self._busy_task:
            return
        self._cancelled.add(key)  # before the await, so the run sees it
        await self.stop()

    async def aclose(self) -> None:
        try:
            await self.stop()
        finally:
            await asyncio.to_thread(self.backend.shutdown)

    async def _run_motion(self, request: SkillRequest, task_id: str | None) -> SkillResult:
        if request.pose is not None and request.pose.frame != self.limits.workspace.frame:
            raise RequestError(f'pose.frame must be "{self.limits.workspace.frame}"')
        # No await between this check and claiming the flag, so it is atomic on the loop.
        if self._stopping:
            return await self._result(request.skill, FailureCode.BUSY, "the robot is stopping")
        if self._busy_task is not None:
            return await self._result(
                request.skill, FailureCode.BUSY, "another skill is running on this robot"
            )
        if self._worker is not None and not self._worker.done():
            return await self._result(
                request.skill, FailureCode.BUSY, "the last motion has not stopped"
            )
        key = self._busy_task = task_id or ""
        self._stop.clear()
        try:
            if request.skill is Skill.MOVE_TO:
                assert request.hand is not None and request.pose is not None
                states = await asyncio.to_thread(self.backend.observe)
                current = self._commanded.get(request.hand, states[request.hand].pose)
                refusal = self.limits.check_move(current, request.pose)
                if refusal is not None:
                    return await self._result(request.skill, *refusal, limits=True)
                call = functools.partial(
                    self.backend.move_to, request.hand, request.pose, self._stop
                )
            elif request.skill is Skill.OPEN:
                call = functools.partial(self.backend.open, request.hand, self._stop)
            else:
                call = functools.partial(self.backend.close, request.hand, self._stop)
            self._commanded.pop(request.hand, None)
            try:
                outcome = await self._run_worker(call)
            except StopError:
                if key not in self._cancelled:
                    raise
                # A timeout during a cancel; cancel() reports the failed stop.
            if key in self._cancelled:
                await self._motion_ended()
                raise SkillCancelled(f"{request.skill} was cancelled")
            if (
                request.skill is Skill.MOVE_TO
                and outcome.failure_code is None
                and not self._stop.is_set()
            ):
                self._commanded[request.hand] = request.pose
            return await self._result(request.skill, outcome.failure_code, outcome.detail)
        finally:
            self._busy_task = None
            self._cancelled.discard(key)

    async def _run_worker(self, call: functools.partial[MotionOutcome]) -> MotionOutcome:
        self._worker = asyncio.ensure_future(asyncio.to_thread(call))
        try:
            return await asyncio.wait_for(asyncio.shield(self._worker), self.limits.skill_timeout_s)
        except TimeoutError:
            await self.stop()
            return MotionOutcome(
                FailureCode.TIMEOUT, f"skill exceeded {self.limits.skill_timeout_s} s"
            )
        except asyncio.CancelledError:
            # tasks/cancel cancels this coroutine, not the thread: stop the hand too.
            # A stop that times out is reported by cancel(), not here.
            with contextlib.suppress(StopError):
                await asyncio.shield(self.stop())
            raise
        except Exception:
            # The backend failed mid-motion: freeze the hands before reporting it.
            try:
                await self.stop()
            except Exception:
                logger.exception("Could not freeze the robot after a backend error")
            raise

    async def _motion_ended(self) -> None:
        """Wait, without a bound, for the motion thread to end.

        When ``cancel`` fails, a2a-adapter reports the task canceled if the run
        has already ended, so a cancelled run must not end before the robot
        stops. The executor waits a bounded time and then fails the task, and
        cancelling this wait ends the run.
        """
        worker = self._worker
        if worker is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(worker)

    async def _wait_for_worker(self) -> bool:
        """Wait for the motion thread to end; False if it is still running."""
        worker = self._worker
        if worker is None or worker.done():
            return True
        try:
            await asyncio.wait_for(asyncio.shield(worker), self.limits.stop_timeout_s)
        except TimeoutError:
            return False
        except Exception:
            pass  # the motion's own error is reported by its run
        return True

    async def _result(
        self,
        skill: Skill,
        failure_code: FailureCode | None = None,
        detail: str | None = None,
        *,
        limits: bool = False,
    ) -> SkillResult:
        states = await asyncio.to_thread(self.backend.observe)
        return SkillResult(
            skill=skill,
            success=failure_code is None,
            failure_code=failure_code,
            detail=detail,
            final_pose={hand: s.pose.to_dict() for hand, s in states.items()},
            contact={hand: dict(s.contact) for hand, s in states.items()},
            held={hand: s.held for hand, s in states.items()},
            limits=self.limits.as_dict() if limits else None,
        )
