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
#include "oneflow_xrt/graph/node_util.h"

#include <string>

#include "oneflow_xrt/common/registry.h"
#include "oneflow_xrt/compiler/kernel/op_kernel_registry.h"
#include "oneflow_xrt/compiler/kernel/op_kernel_registry_id.h"

namespace oneflow {
namespace xrt {

bool IsCanbeCompiledNode(const XrtNode* node, const XrtEngine& engine,
                         const XrtDevice& device) {
  return XRT_REGISTER_HAS(OpKernelRegId,
                          (OpKernelRegKey{node->type(), engine, device}));
}

bool IsModelUpdateNode(const XrtNode* node) {
  return XRT_REGISTER_HAS(ModelUpdateRegId, node->type());
}

bool IsMutableVariable(const Argument& argument, const std::string& op_type,
                       const XrtEngine& engine) {
  const auto& kernel_attrs = XRT_REGISTER_LOOKUP(
      OpKernelAttrRegId, (OpKernelAttrRegKey{op_type, engine}));
  const auto& mutable_variables = kernel_attrs.mutable_variables;
  return mutable_variables.count(argument.meta_data().consume_key);
}

bool IsNodeInput(const XrtNode* node, const Argument& argument) {
  for (XrtEdge* edge : node->in_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

bool IsNodeOutput(const XrtNode* node, const Argument& argument) {
  for (XrtEdge* edge : node->out_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
