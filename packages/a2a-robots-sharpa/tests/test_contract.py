from test_mujoco_backend import backend

from a2a_robots.testing import BackendContract


class TestSharpaMujocoContract(BackendContract):
    def make_backend(self):
        return backend()
