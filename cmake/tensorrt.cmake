if(NOT WITH_CUDA)
  message(FATAL_ERROR "Should recompile OneFlow with BUILD_CUDA=ON")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include
        $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/include)

find_library(
  TENSORRT_LIBRARIES
  NAMES nvinfer
  PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARIES)
  message(
    FATAL_ERROR "TensorRT was not found. You can set TENSORRT_ROOT to specify the search path.")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIBRARIES}")

include_directories(${TENSORRT_INCLUDE_DIR})
list(APPEND XRT_THIRD_PARTY_LIBRARIES ${TENSORRT_LIBRARIES})
