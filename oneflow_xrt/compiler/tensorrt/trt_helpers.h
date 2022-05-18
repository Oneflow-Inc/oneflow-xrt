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
#ifndef ONEFLOW_XRT_COMPILER_TENSORRT_TRT_HELPERS_H_
#define ONEFLOW_XRT_COMPILER_TENSORRT_TRT_HELPERS_H_

#include "oneflow/core/common/scalar.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/trt_shape.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace helpers {

bool DimsEqual(const nvinfer1::Dims& dim1, const nvinfer1::Dims& dim2);

nvinfer1::Weights Constant(TrtOpContext* ctx, const Scalar& value,
                           const Shape& shape, DataType data_type,
                           const std::string& name);

nvinfer1::ITensor* Reshape(TrtOpContext* ctx, nvinfer1::ITensor* in,
                           const Shape& shape);

nvinfer1::ITensor* Reshape(TrtOpContext* ctx, nvinfer1::Weights in,
                           const Shape& shape);

nvinfer1::ITensor* Transpose(TrtOpContext* ctx, nvinfer1::ITensor* in,
                             const std::vector<int>& permute);

nvinfer1::ITensor* Transpose(TrtOpContext* ctx, nvinfer1::Weights in,
                             const Shape& shape,
                             const std::vector<int>& permute);

}  // namespace helpers

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_TENSORRT_TRT_HELPERS_H_
