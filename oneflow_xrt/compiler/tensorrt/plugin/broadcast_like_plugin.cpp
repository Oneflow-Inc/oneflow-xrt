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
#include "oneflow_xrt/compiler/tensorrt/plugin/broadcast_like_plugin.h"

#include "oneflow_xrt/compiler/tensorrt/trt_shape.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::DimsExprs BroadcastLikePlugin::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  // inputs are x, like
  CHECK_EQ(nb_inputs, 2);
  CHECK_EQ(output_index, 0);
  return inputs[1];
}

nvinfer1::DataType BroadcastLikePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  CHECK_EQ(nb_inputs, 2);
  return input_types[0];
}

bool BroadcastLikePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  const auto& desc = in_out[pos];
  return desc.type == in_out[0].type &&
         desc.format == nvinfer1::TensorFormat::kLINEAR;
}

int BroadcastLikePlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                 const nvinfer1::PluginTensorDesc* output_desc,
                                 const void* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) TRT_NOEXCEPT {
  return 0;
}

nvinfer1::IPluginV2DynamicExt* BroadcastLikePlugin::clone() const TRT_NOEXCEPT {
  auto* plugin = new BroadcastLikePlugin(*this);
  plugin->setPluginNamespace(this->getPluginNamespace());
  return plugin;
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
