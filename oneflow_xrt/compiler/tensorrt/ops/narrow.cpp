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

class NarrowOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->InputShape("in_0");
    const int64_t dim = ctx->Attr<int64_t>("dim");
    const int64_t start = ctx->Attr<int64_t>("start");
    const int64_t length = ctx->Attr<int64_t>("length");

    nvinfer1::ITensor* indices = GetIndices(ctx, start, length);
    nvinfer1::ITensor* in = ctx->Input("in_0");
    auto gather_layer =
        ctx->builder()->addGather(*in, *indices, static_cast<int32_t>(dim));

    nvinfer1::ITensor* gather = gather_layer->getOutput(0);
    auto shuffle_layer = ctx->builder()->addShuffle(*gather);

    DimVector dim_vec;
    dim_vec.insert(dim_vec.end(), in_shape.dim_vec().cbegin(),
                   in_shape.dim_vec().cbegin() + dim);
    dim_vec.insert(dim_vec.end(), length);
    dim_vec.insert(dim_vec.end(), in_shape.dim_vec().cbegin() + dim + 1,
                   in_shape.dim_vec().cend());
    shuffle_layer->setReshapeDimensions(ShapeToXrtDims(Shape(dim_vec)));
    shuffle_layer->setName(ctx->op_name().c_str());

    ctx->SetSoleOutput(shuffle_layer->getOutput(0));
  }

  nvinfer1::ITensor* GetIndices(TrtOpContext* ctx, int64_t start,
                                int64_t length) {
    std::vector<int32_t> indices(length);
    for (int i = 0; i < length; ++i) {
      indices[i] = start + i;
    }
    Parameter param(ctx->op_name() + "_constant_indices", indices.data(),
                    Shape{length}, DataType::kInt32);
    int64_t handle = ctx->builder()->AddParameter(param);
    nvinfer1::Weights weight = ctx->builder()->GetWeight(handle);
    auto indices_layer =
        ctx->builder()->addConstant(ShapeToXrtDims(param.shape()), weight);
    return indices_layer->getOutput(0);
  }
};

REGISTER_TRT_OP_KERNEL(narrow, NarrowOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
