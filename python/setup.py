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

from tools.utils import setup_extension, env

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
