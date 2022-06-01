include(ExternalProject)

set(FLATBUFFERS_URL https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz)

set(FLATBUFFERS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install/flatbuffers)
set(FLATBUFFERS_INSTALL_INCLUDEDIR include)
set(FLATBUFFERS_INSTALL_LIBDIR lib)
set(FLATBUFFERS_INSTALL_BINDIR bin)

use_mirror(VARIABLE FLATBUFFERS_URL URL ${FLATBUFFERS_URL})

set(FLATBUFFERS_INCLUDE_DIR ${FLATBUFFERS_INSTALL_PREFIX}/${FLATBUFFERS_INSTALL_INCLUDEDIR})
set(FLATBUFFERS_LIBRARY_DIR ${FLATBUFFERS_INSTALL_PREFIX}/${FLATBUFFERS_INSTALL_LIBDIR})
set(FLATBUFFERS_BINARY_DIR ${FLATBUFFERS_INSTALL_PREFIX}/${FLATBUFFERS_INSTALL_BINDIR})

set(FLATC_EXECUTABLE_NAME flatc)
set(FLATBUFFERS_FLATC_EXECUTABLE ${FLATBUFFERS_BINARY_DIR}/${FLATC_EXECUTABLE_NAME})

set(FLATBUFFERS_LIBRARY_NAMES libflatbuffers.a)
foreach(LIBRARY_NAME ${FLATBUFFERS_LIBRARY_NAMES})
  list(APPEND FLATBUFFERS_STATIC_LIBRARIES ${FLATBUFFERS_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

ExternalProject_Add(
  flatbuffers
  PREFIX flatbuffers
  URL ${FLATBUFFERS_URL}
  URL_MD5 c62ffefb3d4548b127cca14ce047f16c
  UPDATE_COMMAND bash -c "rm -f BUILD || true"
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers/src/flatbuffers
  BUILD_BYPRODUCTS ${FLATBUFFERS_STATIC_LIBRARIES}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_PREFIX}
             -DCMAKE_INSTALL_INCLUDEDIR=${FLATBUFFERS_INSTALL_INCLUDEDIR}
             -DCMAKE_INSTALL_LIBDIR=${FLATBUFFERS_INSTALL_LIBDIR}
             -DCMAKE_INSTALL_BINDIR=${FLATBUFFERS_INSTALL_BINDIR}
             -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
             -DFLATBUFFERS_BUILD_TESTS=OFF)

function(FLATBUFFERS_GENERATE_CPP targetName)
   foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(FIL_WE ${FIL} NAME_WE)

      add_custom_target(${targetName} DEPENDS ${ABS_FIL} ${FLATBUFFERS_FLATC_EXECUTABLE})
      add_custom_command(
        TARGET ${targetName}
        COMMAND  ${FLATBUFFERS_FLATC_EXECUTABLE}
        ARGS -c -b --gen-object-api --reflect-names ${ABS_FIL}
        DEPENDS ${ABS_FIL} ${FLATBUFFERS_FLATC_EXECUTABLE}
        VERBATIM)
    endforeach()
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_generated.h" PROPERTIES GENERATED TRUE)
endfunction()
