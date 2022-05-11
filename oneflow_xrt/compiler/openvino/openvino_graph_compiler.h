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
#ifndef ONEFLOW_XRT_COMPILER_OPENVINO_OPENVINO_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_COMPILER_OPENVINO_OPENVINO_GRAPH_COMPILER_H_

#include "oneflow_xrt/compiler/graph_compiler.h"
#include "oneflow_xrt/compiler/openvino/ngraph_shape.h"
#include "oneflow_xrt/compiler/openvino/openvino_executable.h"
#include "oneflow_xrt/compiler/openvino/ops/op_context.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit OpenvinoGraphCompiler(const std::string& name)
      : GraphCompiler::Impl(name) {}

  virtual ~OpenvinoGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph* graph, const std::vector<Parameter>& entry_params,
      const std::vector<Parameter>& return_params,
      const std::vector<InputOutputAlias>& aliases) override;

 private:
  void SetupKernelContextParam(const XrtNode* node,
                               OpenvinoOpContext::Param* context_param);

  void PopulateEntryParams(
      const std::vector<Parameter>& entry_params,
      std::unordered_map<Argument, Parameter>& entry_params_map,
      std::unordered_map<Argument, int>& entry_params_index_map);

  Argument ArgFromParameter(const Parameter& param);

 private:
  std::unordered_map<std::string, Argument> arguments_;
  std::unordered_map<Argument, std::shared_ptr<ngraph::Node>> operands_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_OPENVINO_OPENVINO_GRAPH_COMPILER_H_
