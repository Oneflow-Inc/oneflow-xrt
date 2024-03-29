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
#include "oneflow/core/common/shape_view.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"
#include "oneflow_xrt/compiler/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::MatrixOperation GetMatrixOperation(nvinfer1::ITensor* x,
                                             bool transpose) {
  if (x->getDimensions().nbDims < 2) {
    return nvinfer1::MatrixOperation::kVECTOR;
  }
  return transpose ? nvinfer1::MatrixOperation::kTRANSPOSE
                   : nvinfer1::MatrixOperation::kNONE;
}

class MatMulOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape a_shape = ctx->InputShape("a_0");
    Shape b_shape = ctx->InputShape("b_0");
    CHECK_GE(a_shape.NumAxes(), 1) << "Tensor a's dim should >= 1";
    CHECK_GE(b_shape.NumAxes(), 1) << "Tensor b's dim should >= 1";

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");
    nvinfer1::ITensor* a = ctx->Input("a_0");
    nvinfer1::ITensor* b = ctx->Input("b_0");

    if ((a_shape.NumAxes() > 2 || b_shape.NumAxes() > 2) &&
        a_shape.NumAxes() != b_shape.NumAxes()) {
      int64_t max_axes = std::max(a_shape.NumAxes(), b_shape.NumAxes());
      if (a_shape.NumAxes() < max_axes) {
        a = helpers::Reshape(ctx, a,
                             CreateLeftExtendedShape(a_shape, max_axes));
      }
      if (b_shape.NumAxes() < max_axes) {
        b = helpers::Reshape(ctx, b,
                             CreateLeftExtendedShape(b_shape, max_axes));
      }
    }

    auto op0 = GetMatrixOperation(a, transpose_a);
    auto op1 = GetMatrixOperation(b, transpose_b);

    auto* layer = ctx->builder()->addMatrixMultiply(*a, op0, *b, op1);
    layer->setName(ctx->op_name().c_str());
    nvinfer1::ITensor* out = layer->getOutput(0);

    if (ctx->HasInput("_add_to_output_0")) {
      auto* add_layer =
          ctx->builder()->addElementWise(*out, *ctx->Input("_add_to_output_0"),
                                         nvinfer1::ElementWiseOperation::kSUM);
      ctx->SetSoleOutput(add_layer->getOutput(0));
    } else {
      ctx->SetSoleOutput(out);
    }
  }
};

REGISTER_TRT_OP_KERNEL(matmul, MatMulOp).EnableTrainPhase().Finalize();
REGISTER_TRT_OP_KERNEL(broadcast_matmul, MatMulOp)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(batch_matmul, MatMulOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
