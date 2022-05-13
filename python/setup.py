# Environment variables:
#
#  DEBUG
#    build with -O0 and -g
#
#  WITH_XLA
#    build with TensorFlow-XLA
#
#  WITH_TENSORRT
#    build with TensorRT
#
#  WITH_OPENVINO
#    build with OpenVINO
#
#  TENSORRT_ROOT
#    specify where TensorRT is installed
#
#  OPENVINO_ROOT
#    specify where OpenVINO runtime is installed

import os
import pathlib
from setuptools import find_packages, setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

cwd = os.path.dirname(os.path.abspath(__file__))


def get_env(name, default=""):
    return os.getenv(name, default)


with_xla = False
with_tensorrt = False
with_openvino = False

if get_env("WITH_XLA") in ["ON", "1"]:
    with_xla = True
if get_env("WITH_TENSORRT") in ["ON", "1"]:
    tensorrt_root = get_env("TENSORRT_ROOT")
    assert (
        tensorrt_root != ""
    ), "should specify where TensorRT is installed when build with TensorRT"
    with_tensorrt = True
if get_env("WITH_OPENVINO") in ["ON", "1"]:
    openvino_root = get_env("OPENVINO_ROOT")
    assert (
        openvino_root != ""
    ), "should specify where OpenVINO runtime is installed when build with OpenVINO"
    with_openvino = True

cmake_build_type = "Release"
if get_env("DEBUG") in ["ON", "1"]:
    cmake_build_type = "Debug"
elif get_env("CMAKE_BUILD_TYPE") != "":
    cmake_build_type = get_env("CMAKE_BUILD_TYPE")
else:
    cmake_build_type = "Release"


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class BuildExt(build_ext):
    def build_extension(self, ext):
        os.makedirs(self.build_temp, exist_ok=True)
        os.chdir(self.build_temp)

        cmake_args = ["-DCMAKE_BUILD_TYPE=" + cmake_build_type]

        if ext.name == "oneflow_xrt_xla":
            cmake_args += ["-DWITH_XLA=ON"]
        elif ext.name == "oneflow_xrt_tensorrt":
            cmake_args += [
                "-DWITH_TENSORRT=ON",
                f"-DTENSORRT_ROOT={get_env('TENSORRT_ROOT')}",
            ]
        elif ext.name == "oneflow_xrt_openvino":
            cmake_args += [
                "-DWITH_OPENVINO=ON",
                f"-DOPENVINO_ROOT={get_env('OPENVINO_ROOT')}",
            ]
        else:
            pass

        source_dir = os.path.join(cwd, "..")
        self.spawn(["cmake", f"{source_dir}"] + cmake_args)

        build_args = ["--config", cmake_build_type, "--", "-j"]
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(cwd)


def setup_stub(package_name, description):
    setup(
        name=package_name,
        version="0.0.1",
        description=(
            "an OneFlow extension that provides an easy to use, "
            "flexible and unified way to integrate third-party computing engines"
        ),
        ext_modules=[CMakeExtension(package_name)],
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


setup_stub(
    "oneflow_xrt",
    description=(
        "an OneFlow extension that provides an easy to use, "
        "flexible and unified way to integrate third-party computing engines"
    ),
)

if with_xla:
    setup_stub(
        "oneflow_xrt_xla", description=("oneflow_xrt's xla extension"),
    )
elif with_tensorrt:
    setup_stub(
        "oneflow_xrt_tensorrt", description=("oneflow_xrt's tensorrt extension"),
    )
elif with_openvino:
    setup_stub(
        "oneflow_xrt_openvino", description=("oneflow_xrt's openvino extension"),
    )
else:
    pass
