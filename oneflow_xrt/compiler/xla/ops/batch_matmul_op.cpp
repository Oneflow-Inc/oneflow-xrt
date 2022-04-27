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
#include "oneflow_xrt/compiler/xla/ops/op_context.h"
#include "oneflow_xrt/compiler/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class BatchMatMulOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    Shape shape_a = ctx->InputShape("a_0");
    Shape shape_b = ctx->InputShape("b_0");
    CHECK_EQ(shape_a.NumAxes(), shape_b.NumAxes());
    CHECK_GT(shape_a.NumAxes(), 2);

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");

    xla::XlaOp a = ctx->Input("a_0");
    xla::XlaOp b = ctx->Input("b_0");

    xla::XlaOp out = xla::BatchDot(a, transpose_a, b, transpose_b);
    if (ctx->HasInput("_add_to_output_0")) {
      out = xla::Add(out, ctx->Input("_add_to_output_0"));
    }
    ctx->SetOutput("out_0", out);
  }
};

REGISTER_XLA_OP_KERNEL(batch_matmul, BatchMatMulOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
