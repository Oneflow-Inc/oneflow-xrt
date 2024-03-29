# The following are set after configuration is done: WITH_CUDA
# ONEFLOW_INCLUDE_DIR ONEFLOW_ROOT_DIR

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import oneflow.sysconfig; print(oneflow.sysconfig.with_cuda())"
  OUTPUT_VARIABLE OneFlow_WITH_CUDA
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${OneFlow_WITH_CUDA}" STREQUAL "True")
  message(STATUS "Build WITH_CUDA=ON")
  set(WITH_CUDA ON)
  add_definitions(-DWITH_CUDA)
else()
  message(STATUS "Build WITH_CUDA=OFF")
  set(WITH_CUDA OFF)
endif()

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import oneflow.sysconfig; print(oneflow.sysconfig.get_include())"
  OUTPUT_VARIABLE ONEFLOW_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import oneflow.sysconfig; print(oneflow.sysconfig.get_lib())"
  OUTPUT_VARIABLE ONEFLOW_ROOT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import oneflow.sysconfig; \
                 print(oneflow.sysconfig.get_liboneflow_link_flags()[0].replace('-L', ''))"
  OUTPUT_VARIABLE OneFlow_LIBRARIE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import oneflow.sysconfig; \
                   print(oneflow.sysconfig.get_liboneflow_link_flags()[1].replace('-l:', ''))"
  OUTPUT_VARIABLE OneFlow_LIBRARIE_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(OneFlow_LIBRARIE_DIR AND OneFlow_LIBRARIE_NAME)
  find_library(
    ONEFLOW_LIBRARIES
    NAMES ${OneFlow_LIBRARIE_NAME}
    PATHS ${OneFlow_LIBRARIE_DIR})
  if(NOT ONEFLOW_LIBRARIES)
    file(GLOB OneFlow_LIBRARIE_PATH
         "${OneFlow_LIBRARIE_DIR}/lib${OneFlow_LIBRARIE_NAME}-*.*")

    list(LENGTH OneFlow_LIBRARIE_PATH OneFlow_LIBRARIE_NUMS)

    if(OneFlow_LIBRARIE_NUMS MATCHES 0)
      message(FATAL_ERROR "There is no oneflow library.")
    endif()

    if(NOT OneFlow_LIBRARIE_NUMS MATCHES 1)
      message(
        FATAL_ERROR
          "There are more than one oneflow library, I dont't konw which one is better."
      )
    endif()
    string(REGEX REPLACE ".+/lib(.+)\\..*" "\\1" OneFlow_LIBRARIE_NEW_NAME
                         ${OneFlow_LIBRARIE_PATH})
    find_library(
      ONEFLOW_LIBRARIES
      NAMES ${OneFlow_LIBRARIE_NEW_NAME}
      PATHS ${OneFlow_LIBRARIE_DIR})
  endif()
endif()

if(NOT ONEFLOW_LIBRARIES)
  message(
    FATAL_ERROR
      "can not find oneflow libraries in directory: ${OneFlow_LIBRARIE_DIR}")
endif()

add_library(oneflow UNKNOWN IMPORTED)
set_property(TARGET oneflow PROPERTY IMPORTED_LOCATION ${ONEFLOW_LIBRARIES})
