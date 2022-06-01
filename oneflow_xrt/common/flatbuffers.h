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
#ifndef ONEFLOW_XRT_COMMON_FLATBUFFERS_H_
#define ONEFLOW_XRT_COMMON_FLATBUFFERS_H_

#include "oneflow_xrt/xrt_generated.h"

namespace oneflow {
namespace xrt {

std::string XrtEngine_Name(const XrtEngine& engine);
std::string XrtDevice_Name(const XrtDevice& engine);

XrtEngine XrtEngine_Parse(const std::string& val);
XrtDevice XrtDevice_Parse(const std::string& val);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMMON_FLATBUFFERS_H_
