include(protobuf)
include(glog)
include(absl)

set(XRT_COMMON_THIRD_PARTY_LIBRARIES
  glog::glog
  absl::algorithm
  absl::base
  absl::debugging
  absl::flat_hash_map
  absl::flags
  absl::memory
  absl::meta
  absl::numeric
  absl::strings
  absl::synchronization
  absl::time
  absl::utility
  absl::span
)
set(XRT_THIRD_PARTY_DEPENDICES protobuf)

set(XRT_THIRD_PARTY_LIBRARIES ${XRT_COMMON_THIRD_PARTY_LIBRARIES})
if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND XRT_THIRD_PARTY_LIBRARIES CUDA::cudart_static)
endif()
