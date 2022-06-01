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
#include "oneflow_xrt/common/protobuf.h"

namespace oneflow {
namespace xrt {
namespace protobuf {

std::string PrintToTextString(const google::protobuf::Message& m) {
  std::string result;
  google::protobuf::TextFormat::PrintToString(m, &result);
  return result;
}

}  // namespace protobuf
}  // namespace xrt
}  // namespace oneflow
