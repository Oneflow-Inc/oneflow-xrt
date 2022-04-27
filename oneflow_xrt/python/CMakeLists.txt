set(ONEFLOW_XRT_PYTHON_DIR ${PROJECT_SOURCE_DIR}/python)
include(pybind11)

file(GLOB XRT_PYTHON_SRCS *.cpp)
pybind11_add_module(oneflow_xrt_internal ${XRT_PYTHON_SRCS})

set_target_properties(oneflow_xrt_internal PROPERTIES CXX_VISIBILITY_PRESET "default")
set_target_properties(oneflow_xrt_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_xrt_internal
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_XRT_PYTHON_DIR}/oneflow_xrt")

target_include_directories(oneflow_xrt_internal PRIVATE ${Python_INCLUDE_DIRS})

if(APPLE)
   target_link_libraries(oneflow_xrt_internal PRIVATE
       -Wl,-force_load
       oneflow_xrt
       oneflow
       ${Python3_LIBRARIES}
       pybind11::headers
       ${XRT_THIRD_PARTY_LIBRARIES})
elseif(UNIX)
  target_link_libraries(oneflow_xrt_internal PRIVATE
       -Wl,--whole-archive
       oneflow_xrt
       -Wl,--no-whole-archive
       oneflow
       ${Python3_LIBRARIES}
       pybind11::headers
       ${XRT_THIRD_PARTY_LIBRARIES})
elseif(WIN32)
  target_link_libraries(oneflow_xrt_internal PRIVATE
       oneflow_xrt
       oneflow
       ${Python3_LIBRARIES}
       pybind11::headers
       ${XRT_THIRD_PARTY_LIBRARIES})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:oneflow_xrt")
endif()
