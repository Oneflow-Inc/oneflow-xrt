import os
from setuptools import find_packages, setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext

cwd = os.path.dirname(os.path.abspath(__file__))


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


class BuildExt(build_ext):
    def build_extension(self, ext):
        os.makedirs(self.build_temp, exist_ok=True)
        os.chdir(self.build_temp)

        cmake_args = ["-DCMAKE_BUILD_TYPE=" + env.cmake_build_type]

        if ext.name == "oneflow_xrt_xla":
            cmake_args += ["-DWITH_XLA=ON"]
        elif ext.name == "oneflow_xrt_tensorrt":
            cmake_args += [
                "-DWITH_TENSORRT=ON",
                f"-DTENSORRT_ROOT={env.tensorrt_root}",
            ]
        elif ext.name == "oneflow_xrt_openvino":
            cmake_args += [
                "-DWITH_OPENVINO=ON",
                f"-DOPENVINO_ROOT={env.openvino_root}",
            ]
        else:
            pass

        source_dir = os.path.join(cwd, "../..")
        self.spawn(["cmake", source_dir] + cmake_args)

        build_args = ["--config", env.cmake_build_type, "--", "-j"]
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(os.path.join(cwd, ".."))


def setup_extension(package_name, description):
    setup(
        name=package_name,
        version="0.0.1",
        description=description,
        ext_modules=[Extension(package_name, sources=[])],
        cmdclass={"build_ext": BuildExt},
        zip_safe=False,
        packages=find_packages(),
        package_data={
            package_name: [
                f"{package_name}/*.so*",
                f"{package_name}/*.dylib*",
                f"{package_name}/*.dll",
                f"{package_name}/*.lib",
            ]
        },
    )
