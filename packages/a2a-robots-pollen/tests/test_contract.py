from a2a_robots.testing import BackendContract
from a2a_robots_pollen import FakeReachySDK, Reachy2Backend


class TestReachy2Contract(BackendContract):
    def make_backend(self) -> Reachy2Backend:
        return Reachy2Backend(FakeReachySDK(), realtime=False)
