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
#ifndef ONEFLOW_XRT_COMPILER_KERNEL_OP_KERNEL_REGISTRY_H_
#define ONEFLOW_XRT_COMPILER_KERNEL_OP_KERNEL_REGISTRY_H_

#include "oneflow_xrt/common/registry.h"
#include "oneflow_xrt/compiler/kernel/op_kernel.h"
#include "oneflow_xrt/compiler/kernel/op_kernel_registry_id.h"

namespace oneflow {
namespace xrt {

struct OpKernelAttributes {
  bool train_phase_enabled = false;
  bool is_model_update_op = false;
  std::set<std::string> mutable_variables;
};

class OpKernelRegistrar {
 public:
  explicit OpKernelRegistrar(const std::string& name) : op_name_(name) {}

  OpKernelRegistrar& SetFactory(const std::function<OpKernelBase*()>& factory) {
    factory_ = factory;
    return *this;
  }

  OpKernelRegistrar& SetEngine(const XrtEngine& engine) {
    engine_ = engine;
    return *this;
  }

  OpKernelRegistrar& SetDevice(const std::vector<XrtDevice>& device) {
    device_ = device;
    return *this;
  }

  OpKernelRegistrar& SetMutableVariables(
      const std::set<std::string>& variables) {
    attrs_.mutable_variables = variables;
    return *this;
  }

  OpKernelRegistrar& MarkModelUpdateOp() {
    attrs_.is_model_update_op = true;
    return *this;
  }

  OpKernelRegistrar& EnableTrainPhase() {
    attrs_.train_phase_enabled = true;
    return *this;
  }

  OpKernelRegistrar& Finalize() {
    if (attrs_.is_model_update_op) {
      XRT_REGISTER(ModelUpdateRegId, op_name_, attrs_.is_model_update_op);
    }
    XRT_REGISTER(OpKernelAttrRegId, (OpKernelAttrRegKey{op_name_, engine_}),
                 attrs_);
    for (const auto& device : device_) {
      XRT_REGISTER(OpKernelRegId, (OpKernelRegKey{op_name_, engine_, device}),
                   factory_);
    }
    return *this;
  }

 private:
  std::function<OpKernelBase*()> factory_;

  std::string op_name_;
  XrtEngine engine_;
  std::vector<XrtDevice> device_{XrtDevice_CPU_X86, XrtDevice_GPU_CUDA};

  OpKernelAttributes attrs_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_KERNEL_OP_KERNEL_REGISTRY_H_
