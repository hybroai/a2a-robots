from a2a_robots import FakeBackend
from a2a_robots.testing import BackendContract


class TestFakeBackendContract(BackendContract):
    def make_backend(self):
        return FakeBackend()
