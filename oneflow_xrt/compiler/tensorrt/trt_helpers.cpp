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
#include "oneflow_xrt/compiler/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace helpers {

bool DimsEqual(const nvinfer1::Dims& dim1, const nvinfer1::Dims& dim2) {
  if (dim1.nbDims != dim2.nbDims) {
    return false;
  }
  for (int i = 0; i < dim1.nbDims; ++i) {
    if (dim1.d[i] != dim2.d[i]) {
      return false;
    }
  }
  return true;
}

nvinfer1::Weights Constant(TrtOpContext* ctx, const Scalar& value,
                           const Shape& shape, DataType data_type,
                           const std::string& name) {
  switch (data_type) {
#define TRT_HELPERS_CONSTANT_SWITCH_ENTRY(T, type)                 \
  case type: {                                                     \
    std::vector<T> v(shape.elem_cnt(), CHECK_JUST(value.As<T>())); \
    Parameter param(name, v.data(), shape, type);                  \
    int64_t handle = ctx->builder()->AddParameter(param);          \
    return ctx->builder()->GetWeight(handle);                      \
  }
    OF_PP_FOR_EACH_TUPLE(TRT_HELPERS_CONSTANT_SWITCH_ENTRY,
                         ARITHMETIC_DATA_TYPE_SEQ)
#undef TRT_HELPERS_CONSTANT_SWITCH_ENTRY
    default: {
      UNIMPLEMENTED() << "Constant does not support data type "
                      << DataType_Name(data_type);
      return nvinfer1::Weights();
    }
  }
}  // namespace helpers

nvinfer1::ITensor* Reshape(TrtOpContext* ctx, nvinfer1::ITensor* in,
                           const Shape& shape) {
  nvinfer1::Dims dims = ShapeToXrtDims(shape);
  if (DimsEqual(in->getDimensions(), dims)) {
    return in;
  }
  auto* layer = ctx->builder()->addShuffle(*in);
  layer->setReshapeDimensions(dims);
  return layer->getOutput(0);
}

nvinfer1::ITensor* Reshape(TrtOpContext* ctx, nvinfer1::Weights in,
                           const Shape& shape) {
  nvinfer1::Dims dims = ShapeToXrtDims(shape);
  auto* layer = ctx->builder()->addConstant(dims, in);
  return layer->getOutput(0);
}

nvinfer1::ITensor* Transpose(TrtOpContext* ctx, nvinfer1::ITensor* in,
                             const std::vector<int>& permute) {
  CHECK_LE(permute.size(), nvinfer1::Dims::MAX_DIMS)
      << "Exceed the allowed maximum dimension.";
  nvinfer1::Permutation permutation;
  for (int i = 0; i < permute.size(); ++i) {
    permutation.order[i] = permute[i];
  }
  auto* layer = ctx->builder()->addShuffle(*in);
  layer->setFirstTranspose(permutation);
  return layer->getOutput(0);
}

nvinfer1::ITensor* Transpose(TrtOpContext* ctx, nvinfer1::Weights in,
                             const Shape& shape,
                             const std::vector<int>& permute) {
  CHECK_LE(permute.size(), nvinfer1::Dims::MAX_DIMS)
      << "Exceed the allowed maximum dimension.";
  nvinfer1::Permutation permutation;
  for (int i = 0; i < permute.size(); ++i) {
    permutation.order[i] = permute[i];
  }

  nvinfer1::ITensor* in_tensor = Reshape(ctx, in, shape);
  auto* layer = ctx->builder()->addShuffle(*in_tensor);
  layer->setFirstTranspose(permutation);
  return layer->getOutput(0);
}

}  // namespace helpers

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
