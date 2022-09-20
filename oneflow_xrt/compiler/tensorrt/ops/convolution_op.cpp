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

class Convolution1dOp : public TrtOpKernel {
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
    CHECK_EQ(kernel_size.size(), 1);
    CHECK_EQ(strides.size(), 1);
    CHECK_EQ(pads.size(), 1);
    CHECK_EQ(dilation.size(), 1);

    int filters = ctx->Attr<int32_t>("filters");

    const auto& in_shape = ctx->InputShape("in_0");
    std::vector<int64_t> shape(in_shape.NumAxes() + 1, 1);
    for (int i = 0; i < in_shape.NumAxes(); ++i) {
      shape[i] = in_shape.At(i);
    }
    in = helpers::Reshape(ctx, in, AsShape(shape));
    auto* layer = ctx->builder()->addConvolutionNd(
        *in, filters, ShapeToXrtDims(Shape{kernel_size[0], 1}), weight, bias);
    layer->setName(ctx->op_name().c_str());

    layer->setStrideNd(ShapeToXrtDims(Shape{strides[0], 1}));
    layer->setDilationNd(ShapeToXrtDims(Shape{dilation[0], 1}));
    layer->setNbGroups(groups);

    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    layer->setPrePadding(ShapeToXrtDims(Shape{pads[0], 0}));
    layer->setPostPadding(ShapeToXrtDims(Shape{pads[0], 0}));

    const auto& out_shape =
        XrtDimsToShape(layer->getOutput(0)->getDimensions());
    CHECK_EQ(out_shape.size(), 4);
    CHECK_EQ(out_shape[3], 1);
    ctx->SetOutput(
        "out_0", helpers::Reshape(
                     ctx, layer->getOutput(0),
                     Shape{out_shape.At(0), out_shape.At(1), out_shape.At(2)}));
  }
};

REGISTER_TRT_OP_KERNEL(conv1d, Convolution1dOp).Finalize();
REGISTER_TRT_OP_KERNEL(conv2d, ConvolutionNdOp<2>).Finalize();
REGISTER_TRT_OP_KERNEL(conv3d, ConvolutionNdOp<3>).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
