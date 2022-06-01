set(ABSL_URL https://github.com/Oneflow-Inc/abseil-cpp/archive/d0.tar.gz)

if(USE_MIRROR)
  use_mirror(VARIABLE ABSL_URL URL ${ABSL_URL})
endif()

include(FetchContent)

FetchContent_Declare(absl URL ${ABSL_URL})

set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(absl)
