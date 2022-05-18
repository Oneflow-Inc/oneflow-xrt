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
#include "NvInfer.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"
#include "oneflow_xrt/compiler/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <nvinfer1::ElementWiseOperation element_wise_op>
class ScalarBinaryOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    const Shape& in_shape = ctx->SoleInputShape();
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = ctx->Attr<int64_t>("int_operand");
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = ctx->Attr<double>("float_operand");
    }
    DataType data_type = ctx->SoleInputType();
    Shape shape(DimVector(in_shape.NumAxes(), 1));
    std::string name = ctx->op_name() + ".scalar";
    nvinfer1::Weights constant =
        helpers::Constant(ctx, value, shape, data_type, name);
    auto* constant_layer =
        ctx->builder()->addConstant(ShapeToXrtDims(shape), constant);
    constant_layer->setName(name.c_str());

    nvinfer1::ITensor* scalar = constant_layer->getOutput(0);
    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addElementWise(*in, *scalar, element_wise_op);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(scalar_add,
                       ScalarBinaryOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(scalar_sub,
                       ScalarBinaryOp<nvinfer1::ElementWiseOperation::kSUB>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(scalar_mul,
                       ScalarBinaryOp<nvinfer1::ElementWiseOperation::kPROD>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(scalar_div,
                       ScalarBinaryOp<nvinfer1::ElementWiseOperation::kDIV>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(scalar_pow,
                       ScalarBinaryOp<nvinfer1::ElementWiseOperation::kPOW>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
