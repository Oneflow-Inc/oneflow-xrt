/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow_xrt/common/device.h"

#include "glog/logging.h"
#include "oneflow/core/common/device_type.pb.h"
#ifdef WITH_CUDA
#include "cuda_runtime.h"
#endif

namespace oneflow {
namespace xrt {

XrtDevice OfDeviceToXrtDevice(const std::string& device) {
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(device));
  return OfDeviceToXrtDevice(device_type);
}

XrtDevice OfDeviceToXrtDevice(const DeviceType& device) {
  switch (device) {
    case DeviceType::kCUDA:
      return XrtDevice::GPU_CUDA;
    case DeviceType::kCPU:
      return XrtDevice::CPU_X86;
    default:
      LOG(WARNING) << "unsupported oneflow device type (" << device
                   << ") is encountered, so use the default CPU device instead";
      return XrtDevice::CPU_X86;
  }
}

DeviceType XrtDeviceToOfDevice(const XrtDevice& device) {
  if (device == XrtDevice::GPU_CUDA) {
    return DeviceType::kCUDA;
  } else if (device == XrtDevice::CPU_X86) {
    return DeviceType::kCPU;
  } else {
    LOG(FATAL) << "unsupported xrt device " << device;
    return DeviceType::kCPU;
  }
}

int GetDeviceId(const XrtDevice& device) {
  switch (device) {
    case XrtDevice::CPU_X86:
      return 0;
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      int device_id = 0;
      CHECK_EQ(cudaSuccess, cudaGetDevice(&device_id));
      return device_id;
#endif
    }
    case XrtDevice::GPU_CL:
    case XrtDevice::CPU_ARM:
      return 0;
  }
  return 0;  // let compiler warning free
}

void SetDeviceId(const XrtDevice& device, const int device_id) {
  switch (device) {
    case XrtDevice::CPU_X86:
      return;
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      CHECK_EQ(cudaSuccess, cudaSetDevice(device_id));
      return;
#endif
    }
    case XrtDevice::GPU_CL:
    case XrtDevice::CPU_ARM:
      return;
  }
}

}  // namespace xrt
}  // namespace oneflow
