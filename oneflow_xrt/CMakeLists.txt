include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

file(GLOB XRT_FLATBUFFERS_FILES *.fbs)
flatbuffers_generate_cpp(xrt_fbs ${XRT_FLATBUFFERS_FILES})

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
add_library(oneflow_xrt SHARED ${XRT_SRCS})
add_dependencies(oneflow_xrt ${XRT_THIRD_PARTY_DEPENDICES} xrt_fbs)
target_link_libraries(
    oneflow_xrt PUBLIC
    ${XRT_THIRD_PARTY_LIBRARIES} ${FLATBUFFERS_STATIC_LIBRARIES}
    -Wl,--version-script ${PROJECT_SOURCE_DIR}/version_script.lds)
target_include_directories(
    oneflow_xrt PRIVATE
    ${ONEFLOW_INCLUDE_DIR} ${FLATBUFFERS_INCLUDE_DIR})

if(BUILD_XLA)
  file(GLOB_RECURSE XRT_XLA_SRCS compiler/xla/*.cpp)
  add_library(oneflow_xrt_xla SHARED ${XRT_XLA_SRCS})
  add_dependencies(
      oneflow_xrt_xla
      ${XRT_XLA_THIRD_PARTY_DEPENDICES}
      oneflow_xrt)
  target_link_libraries(
      oneflow_xrt_xla
      ${XRT_XLA_THIRD_PARTY_LIBRARIES}
      oneflow_xrt)
  target_include_directories(
      oneflow_xrt_xla PRIVATE
      ${TENSORFLOW_XLA_INCLUDE_INSTALL_DIR} ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_TENSORRT)
  file(GLOB_RECURSE XRT_TENSORRT_SRCS compiler/tensorrt/*.cpp)
  add_library(oneflow_xrt_tensorrt SHARED ${XRT_TENSORRT_SRCS})
  add_dependencies(
      oneflow_xrt_tensorrt
      ${XRT_TENSORRT_THIRD_PARTY_DEPENDICES}
      oneflow_xrt)
  target_link_libraries(
      oneflow_xrt_tensorrt
      ${XRT_TENSORRT_THIRD_PARTY_LIBRARIES}
      oneflow_xrt)
  target_include_directories(
      oneflow_xrt_tensorrt PRIVATE ${TENSORRT_INCLUDE_DIR} ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_OPENVINO)
  file(GLOB_RECURSE XRT_OPENVINO_SRCS compiler/openvino/*.cpp)
  add_library(oneflow_xrt_openvino SHARED ${XRT_OPENVINO_SRCS})
  add_dependencies(
      oneflow_xrt_openvino
      ${XRT_OPENVINO_THIRD_PARTY_DEPENDICES}
      oneflow_xrt)
  target_link_libraries(
      oneflow_xrt_openvino
      ${XRT_OPENVINO_THIRD_PARTY_LIBRARIES}
      oneflow_xrt)
  target_include_directories(
      oneflow_xrt_openvino PRIVATE ${ONEFLOW_INCLUDE_DIR})
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
