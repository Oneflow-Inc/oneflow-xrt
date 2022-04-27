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
#ifndef ONEFLOW_XRT_COMPILER_KERNEL_OP_CONTEXT_H_
#define ONEFLOW_XRT_COMPILER_KERNEL_OP_CONTEXT_H_

#include "oneflow/core/framework/attr_map.h"

namespace oneflow {
namespace xrt {

class OpContext {
 public:
  explicit OpContext(const AttrMap& attrs) : attrs_(attrs) {}

  template <typename T>
  T Attr(const std::string& name) {
    return CHECK_JUST(attrs_.GetAttr<T>(name));
  }

 private:
  AttrMap attrs_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_KERNEL_OP_CONTEXT_H_
