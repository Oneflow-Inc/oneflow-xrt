cmake_minimum_required(VERSION 3.18.0)

option(BUILD_XLA "Option to build with XLA" OFF)
option(BUILD_TENSORRT "Option to build with TensorRT" OFF)
option(BUILD_OPENVINO "Option to build with OpenVINO" OFF)
option(BUILD_PYTHON "Option to build python module" ON)
option(AUTO_INSTALL_ONEFLOW "Option to install oneflow automatically with pip" OFF)

project(oneflow-xrt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(THIRD_PARTY_MIRROR aliyun CACHE STRING "")

include(third_party)

if(BUILD_XLA)
  include(tensorflow-xla)
endif()
if(BUILD_TENSORRT)
  include(tensorrt)
endif()
if(BUILD_OPENVINO)
  include(openvino)
endif()

set(INSTALL_DIR
    "${PROJECT_BINARY_DIR}/install"
    CACHE STRING "")
add_subdirectory(oneflow_xrt)
