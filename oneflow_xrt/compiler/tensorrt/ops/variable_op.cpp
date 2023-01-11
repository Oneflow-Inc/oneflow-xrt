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
#include "oneflow_xrt/api/api_internal.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class VariableOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    const auto& buffer = RegisteredBuffer(ctx->op_name());
    xrt::Parameter param(ctx->op_name() + "/out",
                         const_cast<void*>(buffer.first), buffer.second,
                         DataType::kFloat);
    ctx->SetVariable("out", TrtValue::Parameter(ctx->builder(), param));
  }
};

REGISTER_TRT_OP_KERNEL(Variable, VariableOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
