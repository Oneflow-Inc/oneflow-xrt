# Environment variables:
#
#  DEBUG
#    build with -O0 and -g
#
#  BUILD_XLA
#    build XRT TensorFlow-XLA
#
#  BUILD_TENSORRT
#    build XRT TensorRT
#
#  BUILD_OPENVINO
#    build XRT OpenVINO
#
#  TENSORRT_ROOT
#    specify where TensorRT is installed
#
#  OPENVINO_ROOT
#    specify where OpenVINO runtime is installed

from setuptools import find_packages, setup
from setuptools import Extension
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install
import os

from tools.utils import env


cwd = os.path.dirname(os.path.abspath(__file__))


class build_ext(setuptools.command.build_ext.build_ext):
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

        source_dir = os.path.join(cwd, "..")
        self.spawn(["cmake", source_dir] + cmake_args)

        build_args = ["--config", env.cmake_build_type, "--", "-j"]
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(cwd)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("build_ext")
        # clear build lib dir
        import glob
        import shutil

        for filename in glob.glob(f"{self.build_lib}/*"):
            try:
                os.remove(filename)
            except OSError:
                shutil.rmtree(filename, ignore_errors=True)
        super().run()


class install(setuptools.command.install.install):
    def finalize_options(self):
        super().finalize_options()
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

    def run(self):
        super().run()


def setup_extension(package_name, description):
    if not os.path.exists(package_name):
        os.makedirs(package_name)
    setup(
        name=package_name,
        version="0.0.1",
        description=description,
        ext_modules=[Extension(package_name, sources=[])],
        cmdclass={"build_ext": build_ext, "build_py": build_py, "install": install},
        zip_safe=False,
        packages=[package_name],
        package_data={package_name: ["*.so*", "*.dylib*", "*.dll", "*.lib",]},
    )


if env.build_xla:
    setup_extension(
        "oneflow_xrt_xla", description=("oneflow_xrt's xla extension"),
    )
elif env.build_tensorrt:
    assert (
        env.tensorrt_root != ""
    ), "should specify TENSORRT_ROOT where TensorRT runtime is installed when build with TensorRT"
    setup_extension(
        "oneflow_xrt_tensorrt", description=("oneflow_xrt's tensorrt extension"),
    )
elif env.build_openvino:
    assert (
        env.openvino_root != ""
    ), "should specify OPENVINO_ROOT where OpenVINO runtime is installed when build with OpenVINO"
    setup_extension(
        "oneflow_xrt_openvino", description=("oneflow_xrt's openvino extension"),
    )
else:
    setup_extension(
        "oneflow_xrt",
        description=(
            "an OneFlow extension that provides an easy to use, "
            "flexible and unified way to integrate third-party computing engines"
        ),
    )
