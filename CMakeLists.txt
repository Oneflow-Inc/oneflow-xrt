cmake_minimum_required(VERSION 3.18.0)

option(WITH_XLA "Option to build with XLA" OFF)
option(WITH_TENSORRT "Option to build with TensorRT" OFF)
option(WITH_OPENVINO "Option to build with OpenVINO" OFF)
option(BUILD_PYTHON "Option to build python module" ON)

project(oneflow-xrt CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
)
include(oneflow)
include(third_party)

if(WITH_XLA)
  include(tensorflow-xla)
endif()
if(WITH_TENSORRT)
  include(tensorrt)
endif()
if(WITH_OPENVINO)
  include(openvino)
endif()

include_directories(${ONEFLOW_INCLUDE_DIR})
add_subdirectory(oneflow_xrt)
