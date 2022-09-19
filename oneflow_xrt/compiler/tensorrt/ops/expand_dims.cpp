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
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"
#include "oneflow_xrt/compiler/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ExpandDimsOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->InputShape("in_0");
    int32_t axis = ctx->Attr<int32_t>("axis");
    if (axis < 0) {
      axis = axis + in_shape.NumAxes() + 1;
    }

    auto dim_vec = in_shape.dim_vec();
    dim_vec.insert(dim_vec.begin() + axis, 1);
    ctx->SetSoleOutput(
        helpers::Reshape(ctx, ctx->Input("in_0"), Shape(dim_vec)));
  }
};

REGISTER_TRT_OP_KERNEL(expand_dims, ExpandDimsOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
