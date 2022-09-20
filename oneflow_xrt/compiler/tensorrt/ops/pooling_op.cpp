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

template <nvinfer1::PoolingType pooling_type>
class TfPoolingOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    CHECK_GE(in_shape.NumAxes(), 3);
    CHECK_LE(in_shape.NumAxes(), 5);

    const auto& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");

    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addPooling(
        *in, pooling_type, nvinfer1::DimsHW(pool_size[0], pool_size[1]));
    layer->setName(ctx->op_name().c_str());

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));

    const std::string& padding = ctx->Attr<std::string>("padding");
    // The default padding mode is valid for TensorRT.
    if (padding != "valid") {
      if (padding == "customized") {
        const auto& padding_before =
            ctx->Attr<std::vector<int32_t>>("padding_before");
        const auto& padding_after =
            ctx->Attr<std::vector<int32_t>>("padding_after");
        const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
        if (ceil_mode) {
          layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
        } else {
          layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
        }
        layer->setPrePadding(
            nvinfer1::DimsHW(padding_before[0], padding_before[1]));
        layer->setPostPadding(
            nvinfer1::DimsHW(padding_after[0], padding_after[1]));
      } else if (padding == "same_lower") {
        layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_LOWER);
      } else if (padding == "same_upper") {
        layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
      } else {
        UNIMPLEMENTED();
      }
    }
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(tf_max_pool_2d, TfPoolingOp<nvinfer1::PoolingType::kMAX>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(tf_avg_pool_2d,
                       TfPoolingOp<nvinfer1::PoolingType::kAVERAGE>)
    .EnableTrainPhase()
    .Finalize();

template <nvinfer1::PoolingType pooling_type>
class PoolingOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    CHECK_GE(in_shape.NumAxes(), 3);
    CHECK_LE(in_shape.NumAxes(), 5);

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    std::vector<int32_t> kernel_size =
        ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    if (kernel_size.size() == 1) {
      kernel_size.emplace_back(1);
    }
    if (padding.size() == 1) {
      padding.emplace_back(0);
    }
    if (stride.size() == 1) {
      stride.emplace_back(1);
    }

    nvinfer1::ITensor* in = ctx->SoleInput();
    if (in_shape.NumAxes() < 4) {
      std::vector<int64_t> dims(4, 1);
      for (int i = 0; i < in_shape.NumAxes(); ++i) {
        dims[i] = in_shape.At(i);
      }
      in = helpers::Reshape(ctx, in, AsShape(dims));
    }

    nvinfer1::IPoolingLayer* layer = ctx->builder()->addPoolingNd(
        *in, pooling_type, IntListToXrtDims(kernel_size));
    if (pooling_type == nvinfer1::PoolingType::kMAX) {
      std::vector<int32_t> dilation =
          ctx->Attr<std::vector<int32_t>>("dilation");
      CHECK(dilation == std::vector<int32_t>(dilation.size(), 1))
          << "Pooling dilation is not supported in TensorRT";
    } else {
      const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
      layer->setAverageCountExcludesPadding(!count_include_pad);
    }
    layer->setName(ctx->op_name().c_str());

    auto padding_mode = ceil_mode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP
                                  : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    layer->setPaddingMode(padding_mode);
    layer->setPaddingNd(IntListToXrtDims(padding));
    layer->setStrideNd(IntListToXrtDims(stride));

    auto* layer_out = layer->getOutput(0);
    if (in_shape.NumAxes() < 4) {
      auto out_shape = XrtDimsToShape(layer_out->getDimensions());
      out_shape.resize(in_shape.NumAxes());
      layer_out = helpers::Reshape(ctx, layer_out, out_shape);
    }
    ctx->SetSoleOutput(layer_out);
  }
};

REGISTER_TRT_OP_KERNEL(max_pool_2d, PoolingOp<nvinfer1::PoolingType::kMAX>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(avg_pool_2d, PoolingOp<nvinfer1::PoolingType::kAVERAGE>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
