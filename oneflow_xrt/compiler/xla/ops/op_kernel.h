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
#ifndef ONEFLOW_XRT_COMPILER_XLA_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_COMPILER_XLA_OPS_OP_KERNEL_H_

#include "oneflow_xrt/common/device.h"
#include "oneflow_xrt/common/registry.h"
#include "oneflow_xrt/compiler/kernel/op_kernel.h"
#include "oneflow_xrt/compiler/kernel/op_kernel_registry.h"
#include "oneflow_xrt/compiler/xla/ops/op_context.h"
#include "oneflow_xrt/compiler/xla/xla_macro.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaOpKernel : public OpKernel<XlaOpContext> {
 public:
  virtual void Compile(XlaOpContext* ctx) = 0;

  XlaOpKernel() = default;
  virtual ~XlaOpKernel() = default;
};

#define REGISTER_XLA_OP_KERNEL(OpName, KernelType)    \
  static OpKernelRegistrar _xla_op_kernel_##OpName##_ \
      __attribute__((unused)) =                       \
          OpKernelRegistrar(#OpName)                  \
              .SetEngine(XrtEngine::XLA)              \
              .EnableTrainPhase()                     \
              .SetFactory([]() -> OpKernelBase* { return new KernelType; })

inline std::shared_ptr<XlaOpKernel> BuildOpKernel(const XrtDevice& device,
                                                  const std::string& op_name) {
  OpKernelRegKey reg_key{op_name, XrtEngine::XLA, device};
  const auto& f = XRT_REGISTER_LOOKUP(OpKernelRegId, reg_key);
  auto* xla_kernel = dynamic_cast<XlaOpKernel*>(f());
  CHECK(xla_kernel) << "failed to build xla op kernel for " << reg_key;
  return std::shared_ptr<XlaOpKernel>(xla_kernel);
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_XLA_OPS_OP_KERNEL_H_
