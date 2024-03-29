set(ONEFLOW_XRT_PYTHON_DIR ${PROJECT_SOURCE_DIR}/python)
include(pybind11)

function(ONEFLOW_XRT_ADD_MODULE target_name)
  pybind11_add_module(${target_name} SHARED ${ARGN})
  set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "default")
  set_target_properties(${target_name} PROPERTIES PREFIX "_")
  target_include_directories(${target_name} PRIVATE ${Python_INCLUDE_DIRS})
endfunction()

set(XRT_PYTHON_SRCS
  stub.cpp
  graph.cpp
  options.cpp
  int8_calibration.cpp
)
oneflow_xrt_add_module(oneflow_xrt_internal ${XRT_PYTHON_SRCS})
set_target_properties(oneflow_xrt_internal
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_XRT_PYTHON_DIR}/oneflow_xrt")

target_include_directories(oneflow_xrt_internal PRIVATE ${ONEFLOW_INCLUDE_DIR})
target_link_libraries(oneflow_xrt_internal PUBLIC oneflow_xrt glog::glog)

function(ONEFLOW_XRT_ADD_STUB target_name)
  oneflow_xrt_add_module(${target_name}_internal ${ARGN})
  set_target_properties(${target_name}_internal
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_XRT_PYTHON_DIR}/${target_name}")

  target_link_libraries(${target_name}_internal PUBLIC
      -Wl,--no-as-needed
      ${target_name}
      -Wl,--as-needed
      glog::glog)

  add_custom_target(
      ${target_name}_create_python_module
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/tools/create_python_module.py
              ${ONEFLOW_XRT_PYTHON_DIR} ${target_name}
      DEPENDS ${Python_EXECUTABLE}
  )
  add_dependencies(${target_name}_internal ${target_name}_create_python_module)
endfunction()

if(BUILD_XLA)
  oneflow_xrt_add_stub(oneflow_xrt_xla xla_stub.cpp)
endif()

if(BUILD_TENSORRT)
  oneflow_xrt_add_stub(oneflow_xrt_tensorrt tensorrt_stub.cpp)
endif()

if(BUILD_OPENVINO)
  oneflow_xrt_add_stub(oneflow_xrt_openvino openvino_stub.cpp)
endif()
