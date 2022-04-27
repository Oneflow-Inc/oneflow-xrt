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
#ifndef ONEFLOW_XRT_COMPILER_KERNEL_OP_KERNEL_REGISTRY_ID_H_
#define ONEFLOW_XRT_COMPILER_KERNEL_OP_KERNEL_REGISTRY_ID_H_

#include <functional>
#include <iostream>
#include <string>

#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

struct OpKernelAttributes;
class OpKernelBase;

struct ModelUpdateRegId {
  using Factory = bool;
};
struct OpKernelAttrRegId {
  using Factory = OpKernelAttributes;
};
struct OpKernelRegId {
  using Factory = std::function<OpKernelBase*()>;
};

struct OpKernelAttrRegKey {
  std::string op_name;
  XrtEngine engine;
};

struct OpKernelRegKey {
  std::string op_name;
  XrtEngine engine;
  XrtDevice device;
};

inline bool operator==(const OpKernelAttrRegKey& lhs,
                       const OpKernelAttrRegKey& rhs) {
  return lhs.op_name == rhs.op_name && lhs.engine == rhs.engine;
}
inline bool operator==(const OpKernelRegKey& lhs, const OpKernelRegKey& rhs) {
  return lhs.op_name == rhs.op_name && lhs.engine == rhs.engine &&
         lhs.device == rhs.device;
}

inline std::ostream& operator<<(std::ostream& ost,
                                const OpKernelAttrRegKey& key) {
  ost << "OpKernelAttrRegKey(" << key.op_name << ", "
      << XrtEngine_Name(key.engine) << ")";
  return ost;
}

inline std::ostream& operator<<(std::ostream& ost, const OpKernelRegKey& key) {
  ost << "OpKernelRegKey(" << key.op_name << ", " << XrtEngine_Name(key.engine)
      << ", " << XrtDevice_Name(key.device) << ")";
  return ost;
}

}  // namespace xrt
}  // namespace oneflow

namespace std {

template <>
struct hash<oneflow::xrt::OpKernelAttrRegKey> {
  size_t operator()(const oneflow::xrt::OpKernelAttrRegKey& key) const {
    return std::hash<std::string>()(key.op_name) ^
           std::hash<int64_t>()(static_cast<int64_t>(key.engine));
  }
};

template <>
struct hash<oneflow::xrt::OpKernelRegKey> {
  size_t operator()(const oneflow::xrt::OpKernelRegKey& key) const {
    return std::hash<std::string>()(key.op_name) ^
           std::hash<int64_t>()(static_cast<int64_t>(key.engine)) ^
           std::hash<int64_t>()(static_cast<int64_t>(key.device));
  }
};

}  // namespace std

#endif  // ONEFLOW_XRT_KERNEL_OP_KERNEL_REGISTRY_ID_H_
