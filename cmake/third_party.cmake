include(protobuf)
include(glog)
include(absl)

set(XRT_THIRD_PARTY_LIBRARIES
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

if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(XRT_THIRD_PARTY_LIBRARIES CUDA::cudart_static ${XRT_THIRD_PARTY_LIBRARIES})
endif()
