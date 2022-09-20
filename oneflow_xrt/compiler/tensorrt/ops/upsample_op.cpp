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
#include "oneflow_xrt/common/shape_util.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"
#include "oneflow_xrt/compiler/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <nvinfer1::ResizeMode resize_mode>
class Upsample2dOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    CHECK_EQ(in_shape.NumAxes(), 4);

    const double height_scale = ctx->Attr<double>("height_scale");
    const double width_scale = ctx->Attr<double>("width_scale");
    std::vector<int64_t> output_size =
        ctx->Attr<std::vector<int64_t>>("output_size");
    CHECK(output_size.empty())
        << "Upsample output_size is not supported in TensorRT";

    nvinfer1::ITensor* in = ctx->SoleInput();
    nvinfer1::IResizeLayer* layer = ctx->builder()->addResize(*in);
    layer->setName(ctx->op_name().c_str());

    std::vector<float> scales{1.0, 1.0, height_scale, width_scale};
    layer->setScales(scales.data(), 4);
    layer->setResizeMode(resize_mode);
    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
    layer->setCoordinateTransformation(
        nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(upsample_nearest_2d,
                       Upsample2dOp<nvinfer1::ResizeMode::kNEAREST>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
