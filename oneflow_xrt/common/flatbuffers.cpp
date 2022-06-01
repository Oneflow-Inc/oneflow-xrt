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
#include "oneflow_xrt/common/flatbuffers.h"
#include "glog/logging.h"

namespace oneflow {
namespace xrt {

std::string XrtEngine_Name(const XrtEngine& engine) {
  return EnumNameXrtEngine(engine);
}
std::string XrtDevice_Name(const XrtDevice& device) {
  return EnumNameXrtDevice(device);
}

XrtEngine XrtEngine_Parse(const std::string& val) {
  const auto* names = EnumNamesXrtEngine();
  for (int i = XrtEngine_MIN; i < XrtEngine_MAX; ++i) {
    if (val == std::string(names[i])) {
      return EnumValuesXrtEngine()[i];
    }
  }
  LOG(FATAL) << "Unknow engine: " << val;
}

XrtDevice XrtDevice_Parse(const std::string& val) {
  const auto* names = EnumNamesXrtDevice();
  for (int i = XrtDevice_MIN; i < XrtDevice_MAX; ++i) {
    if (val == std::string(names[i])) {
      return EnumValuesXrtDevice()[i];
    }
  }
  LOG(FATAL) << "Unknow device: " << val;
}

}  // namespace xrt
}  // namespace oneflow
