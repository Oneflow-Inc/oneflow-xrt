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
#include "oneflow_xrt/compiler/openvino/ops/op_context.h"
#include "oneflow_xrt/compiler/openvino/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class XrtEntryOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    ctx->SetOutput("value", ctx->Variable());
  }
};

class XrtReturnOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    ctx->SetVariable(ctx->Input("value"));
  }
};

REGISTER_OPENVINO_OP_KERNEL(XrtEntry, XrtEntryOp).Finalize();
REGISTER_OPENVINO_OP_KERNEL(XrtReturn, XrtReturnOp).Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
