include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

file(GLOB XRT_PROTO_FILES *.proto)
set(PROTOBUF_IMPORT_DIRS ${ONEFLOW_INCLUDE_DIR})
protobuf_generate_cpp(XRT_PROTO_SRCS XRT_PROTO_HDRS ${XRT_PROTO_FILES})

add_library(oneflow_xrt_proto STATIC ${XRT_PROTO_HDRS} ${XRT_PROTO_SRCS})
add_dependencies(oneflow_xrt_proto protobuf)
target_link_libraries(oneflow_xrt_proto PUBLIC ${PROTOBUF_STATIC_LIBRARIES})
target_include_directories(oneflow_xrt_proto PRIVATE ${ONEFLOW_INCLUDE_DIR})

file(GLOB_RECURSE XRT_SRCS
  api/*.cpp
  common/*.cpp
  graph/*.cpp
  compiler/compilation_cache.cpp
  compiler/kernel/*.cpp
  compiler/passes/*.cpp
  int8_calibration/*.cpp
  xrt_launch_kernel.cpp
  xrt_launch_op.cpp
)
add_library(oneflow_xrt ${XRT_SRCS})
add_dependencies(oneflow_xrt ${XRT_THIRD_PARTY_DEPENDICES})
target_link_libraries(oneflow_xrt PRIVATE oneflow_xrt_proto ${XRT_THIRD_PARTY_LIBRARIES})
target_include_directories(oneflow_xrt PRIVATE ${ONEFLOW_INCLUDE_DIR})

if(BUILD_XLA)
  file(GLOB_RECURSE XRT_XLA_SRCS compiler/xla/*.cpp)
  add_library(oneflow_xrt_xla ${XRT_XLA_SRCS})
  add_dependencies(
      oneflow_xrt_xla
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_XLA_THIRD_PARTY_DEPENDICES})
  target_link_libraries(
      oneflow_xrt_xla
      oneflow_xrt
      oneflow_xrt_proto
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_XLA_THIRD_PARTY_LIBRARIES})
  target_include_directories(
      oneflow_xrt_xla PRIVATE ${TENSORFLOW_XLA_INCLUDE_INSTALL_DIR} ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_TENSORRT)
  file(GLOB_RECURSE XRT_TENSORRT_SRCS compiler/tensorrt/*.cpp)
  add_library(oneflow_xrt_tensorrt ${XRT_TENSORRT_SRCS})
  add_dependencies(
      oneflow_xrt_tensorrt
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_TENSORRT_THIRD_PARTY_DEPENDICES})
  target_link_libraries(
      oneflow_xrt_tensorrt
      oneflow_xrt
      oneflow_xrt_proto
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_TENSORRT_THIRD_PARTY_LIBRARIES})
  target_include_directories(
      oneflow_xrt_tensorrt PRIVATE ${TENSORRT_INCLUDE_DIR} ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_OPENVINO)
  file(GLOB_RECURSE XRT_OPENVINO_SRCS compiler/openvino/*.cpp)
  add_library(oneflow_xrt_openvino ${XRT_OPENVINO_SRCS})
  add_dependencies(
      oneflow_xrt_openvino
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_OPENVINO_THIRD_PARTY_DEPENDICES})
  target_link_libraries(
      oneflow_xrt_openvino
      oneflow_xrt
      oneflow_xrt_proto
      ${XRT_COMMON_THIRD_PARTY_LIBRARIES}
      ${XRT_OPENVINO_THIRD_PARTY_LIBRARIES})
  target_include_directories(
      oneflow_xrt_openvino PRIVATE ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
