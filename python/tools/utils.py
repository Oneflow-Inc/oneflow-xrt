import os


def get_env(name, default=""):
    return os.getenv(name, default)


class Env(object):
    def __init__(self):
        pass

    @property
    def cmake_build_type(self):
        if get_env("DEBUG") in ["ON", "1"]:
            return "Debug"
        elif get_env("CMAKE_BUILD_TYPE") != "":
            return get_env("CMAKE_BUILD_TYPE")
        else:
            return "Release"

    @property
    def build_xla(self):
        return get_env("BUILD_XLA") in ["ON", "1"]

    @property
    def build_tensorrt(self):
        return get_env("BUILD_TENSORRT") in ["ON", "1"]

    @property
    def build_openvino(self):
        return get_env("BUILD_OPENVINO") in ["ON", "1"]

    @property
    def tensorrt_root(self):
        return get_env("TENSORRT_ROOT")

    @property
    def openvino_root(self):
        return get_env("OPENVINO_ROOT")


env = Env()
