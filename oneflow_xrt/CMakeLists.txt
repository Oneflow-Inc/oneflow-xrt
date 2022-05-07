include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

file(GLOB XRT_PROTO_FILES *.proto)
set(PROTOBUF_IMPORT_DIRS ${ONEFLOW_INCLUDE_DIR})
protobuf_generate_cpp(XRT_PROTO_SRCS XRT_PROTO_HDRS ${XRT_PROTO_FILES})

add_library(oneflow_xrt_proto STATIC ${XRT_PROTO_HDRS} ${XRT_PROTO_SRCS})
add_dependencies(oneflow_xrt_proto protobuf)
target_link_libraries(oneflow_xrt_proto PUBLIC ${PROTOBUF_STATIC_LIBRARIES})

file(GLOB_RECURSE XRT_SRCS
  api/*.cpp
  common/*.cpp
  graph/*.cpp
  compiler/compilation_cache.cpp
  compiler/kernel/*.cpp
  compiler/passes/*.cpp
  xrt_launch_kernel.cpp
  xrt_launch_op.cpp
)

if(WITH_XLA)
  file(GLOB_RECURSE XRT_XLA_SRCS compiler/xla/*.cpp)
  list(APPEND XRT_SRCS ${XRT_XLA_SRCS})
endif()

if(WITH_TENSORRT)
  file(GLOB_RECURSE XRT_TENSORRT_SRCS compiler/tensorrt/*.cpp)
  list(APPEND XRT_SRCS ${XRT_TENSORRT_SRCS})
endif()

if(WITH_OPENVINO)
  file(GLOB_RECURSE XRT_OPENVINO_SRCS compiler/openvino/*.cpp)
  list(APPEND XRT_SRCS ${XRT_OPENVINO_SRCS})
endif()

add_library(oneflow_xrt STATIC ${XRT_SRCS})
add_dependencies(oneflow_xrt ${XRT_THIRD_PARTY_DEPENDICES})
target_link_libraries(oneflow_xrt oneflow_xrt_proto ${XRT_THIRD_PARTY_LIBRARIES})

add_subdirectory(python)
