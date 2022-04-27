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
#ifndef ONEFLOW_XRT_COMPILER_COMPILATION_CACHE_H_
#define ONEFLOW_XRT_COMPILER_COMPILATION_CACHE_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "oneflow/core/common/shape.h"
#include "oneflow_xrt/compiler/executable.h"
#include "oneflow_xrt/compiler/parameter.h"

namespace oneflow {
namespace xrt {

struct Signature {
  // builder name
  std::string builder_name;
  // device ordinal
  int device_ordinal;

  // the signature should be recompute if the entry shape has been changed
  std::vector<Shape> entry_shapes;
};

bool operator==(const Signature& lhs, const Signature& rhs);

struct SignatureHash {
  size_t operator()(const Signature& signature) const;
};

class CompilationCache {
 public:
  Executable* GetRecord(const Signature& signature) const;

  void Record(const Signature& signature,
              const std::shared_ptr<Executable>& result);

  void Release();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<Signature, std::shared_ptr<Executable>, SignatureHash>
      records_;
};

Signature ComputeSignature(const std::string& name, const int device_ordinal,
                           const std::vector<xrt::Parameter>& entry_params);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_COMPILATION_CACHE_H_
