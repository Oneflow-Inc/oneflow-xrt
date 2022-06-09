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
#include "oneflow_xrt/compiler/tensorrt/trt_shape.h"  // DataTypeToTrtDataType

namespace oneflow {
namespace xrt {
namespace tensorrt {

class CastOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    DataType dest_dtype = ctx->Attr<DataType>("dtype");
    DataType src_dtype = ctx->SoleInputType();
    nvinfer1::ITensor* in = ctx->SoleInput();
    if (src_dtype == dest_dtype) {
      ctx->SetSoleOutput(in);
    } else {
      auto* layer = ctx->builder()->addIdentity(*in);
      layer->setOutputType(0, DataTypeToTrtDataType(dest_dtype));
      layer->setName(ctx->op_name().c_str());
      ctx->SetSoleOutput(layer->getOutput(0));
    }
  }
};

REGISTER_TRT_OP_KERNEL(cast, CastOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
