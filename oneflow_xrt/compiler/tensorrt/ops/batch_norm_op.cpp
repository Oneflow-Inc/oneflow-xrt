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

template <typename T>
static T* GetWeightPtr(const nvinfer1::Weights& weight) {
  return reinterpret_cast<T*>(const_cast<void*>(weight.values));
}

class NormalizationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->InputShape("x_0");
    CHECK(in_shape.NumAxes() >= 2 && in_shape.NumAxes() < 5)
        << "Only support 2, 3 or 4 dimension input, but got "
        << in_shape.NumAxes();

    float epsilon = ctx->Attr<float>("epsilon");

    nvinfer1::Weights gamma = ctx->Weight("gamma_0");
    nvinfer1::Weights beta = ctx->Weight("beta_0");
    nvinfer1::Weights moving_mean = ctx->Weight("moving_mean_0");
    nvinfer1::Weights moving_variance = ctx->Weight("moving_variance_0");

    float* gamma_ptr = GetWeightPtr<float>(gamma);
    float* beta_ptr = GetWeightPtr<float>(beta);
    const float* moving_mean_ptr = GetWeightPtr<float>(moving_mean);
    const float* moving_variance_ptr = GetWeightPtr<float>(moving_variance);

    for (int i = 0; i < gamma.count; ++i) {
      *gamma_ptr /= std::sqrt(*moving_variance_ptr + epsilon);
      *beta_ptr -= *moving_mean_ptr * (*gamma_ptr);
      gamma_ptr += 1;
      beta_ptr += 1;
      moving_mean_ptr += 1;
      moving_variance_ptr += 1;
    }

    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ITensor* input = ctx->Input("x_0");
    // IScaleLayer only support 3 dim or 4 dim input
    if (in_shape.NumAxes() < 3) {
      Shape shape{1, 1, 1, 1};
      for (int i = 0; i < in_shape.NumAxes(); ++i) {
        shape.Set(i, in_shape.At(i));
      }
      input = helpers::Reshape(ctx, input, shape);
    }

    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
    nvinfer1::IScaleLayer* layer =
        ctx->builder()->addScale(*input, mode, beta, gamma, power);
    layer->setName(ctx->op_name().c_str());
    nvinfer1::ITensor* out = layer->getOutput(0);

    if (in_shape.NumAxes() < 3) {
      out = helpers::Reshape(ctx, out, in_shape);
    }
    if (ctx->HasInput("_add_to_output_0")) {
      auto* add_layer =
          ctx->builder()->addElementWise(*out, *ctx->Input("_add_to_output_0"),
                                         nvinfer1::ElementWiseOperation::kSUM);
      ctx->SetOutput("y_0", add_layer->getOutput(0));
    } else {
      ctx->SetOutput("y_0", out);
    }
  }
};

REGISTER_TRT_OP_KERNEL(normalization, NormalizationOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
