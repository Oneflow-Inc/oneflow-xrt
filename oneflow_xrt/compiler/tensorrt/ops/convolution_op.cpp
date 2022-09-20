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

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <int Ndims>
class ConvolutionNdOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* in = ctx->Input("in_0");
    nvinfer1::Weights weight = ctx->Weight("weight_0");

    nvinfer1::Weights bias;
    if (ctx->HasInput("bias_0")) {
      bias = ctx->Weight("bias_0");
    } else {
      bias = nvinfer1::Weights{nvinfer1::DataType::kFLOAT /* type */,
                               nullptr /* values */, 0 /* count */};
    }

    CHECK_EQ(ctx->Attr<std::string>("data_format"), "channels_first");
    const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& pads = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int groups = ctx->Attr<int32_t>("groups");
    CHECK_EQ(kernel_size.size(), Ndims);
    CHECK_EQ(strides.size(), Ndims);
    CHECK_EQ(pads.size(), Ndims);
    CHECK_EQ(dilation.size(), Ndims);

    int filters = ctx->Attr<int32_t>("filters");
    auto* layer = ctx->builder()->addConvolutionNd(
        *in, filters, IntListToXrtDims(kernel_size), weight, bias);
    layer->setName(ctx->op_name().c_str());

    layer->setStrideNd(IntListToXrtDims(strides));
    layer->setDilationNd(IntListToXrtDims(dilation));
    layer->setNbGroups(groups);

    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    layer->setPrePadding(IntListToXrtDims(pads));
    layer->setPostPadding(IntListToXrtDims(pads));
    ctx->SetOutput("out_0", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(conv1d, ConvolutionNdOp<1>).Finalize();
REGISTER_TRT_OP_KERNEL(conv2d, ConvolutionNdOp<2>).Finalize();
REGISTER_TRT_OP_KERNEL(conv3d, ConvolutionNdOp<3>).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
