set(ABSL_URL https://github.com/Oneflow-Inc/abseil-cpp/archive/d0.tar.gz)

include(FetchContent)

FetchContent_Declare(absl URL ${ABSL_URL})

set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(absl)
