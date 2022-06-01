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
#ifndef ONEFLOW_XRT_COMPILER_OPENVINO_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_COMPILER_OPENVINO_OPS_OP_KERNEL_H_

#include "oneflow_xrt/common/registry.h"
#include "oneflow_xrt/compiler/kernel/op_kernel.h"
#include "oneflow_xrt/compiler/kernel/op_kernel_registry.h"
#include "oneflow_xrt/compiler/openvino/ops/op_context.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoOpKernel : public OpKernel<OpenvinoOpContext> {
 public:
  virtual void Compile(OpenvinoOpContext* ctx) = 0;

  OpenvinoOpKernel() = default;
  virtual ~OpenvinoOpKernel() = default;
};

#define REGISTER_OPENVINO_OP_KERNEL(OpName, KernelType)    \
  static OpKernelRegistrar _openvino_op_kernel_##OpName##_ \
      __attribute__((unused)) =                            \
          OpKernelRegistrar(#OpName)                       \
              .SetEngine(XrtEngine_OPENVINO)              \
              .SetDevice({XrtDevice_CPU_X86})             \
              .SetFactory([]() -> OpKernelBase* { return new KernelType; })

inline std::shared_ptr<OpenvinoOpKernel> BuildOpKernel(
    const std::string& op_name) {
  OpKernelRegKey reg_key{op_name, XrtEngine_OPENVINO, XrtDevice_CPU_X86};
  const auto& f = XRT_REGISTER_LOOKUP(OpKernelRegId, reg_key);
  auto* openvino_kernel = dynamic_cast<OpenvinoOpKernel*>(f());
  CHECK(openvino_kernel) << "failed to build openvino op kernel for "
                         << reg_key;
  return std::shared_ptr<OpenvinoOpKernel>(openvino_kernel);
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_OPENVINO_OPS_OP_KERNEL_H_
