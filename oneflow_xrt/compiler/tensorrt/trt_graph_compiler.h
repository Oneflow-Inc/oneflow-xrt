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
#ifndef ONEFLOW_XRT_COMPILER_TENSORRT_TRT_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_COMPILER_TENSORRT_TRT_GRAPH_COMPILER_H_

#include "NvInfer.h"
#include "oneflow_xrt/compiler/graph_compiler.h"
#include "oneflow_xrt/compiler/tensorrt/ops/op_context.h"
#include "oneflow_xrt/compiler/tensorrt/trt_builder.h"
#include "oneflow_xrt/compiler/tensorrt/trt_executable.h"
#include "oneflow_xrt/compiler/tensorrt/trt_value.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TrtGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit TrtGraphCompiler(const std::string& name)
      : GraphCompiler::Impl(name) {
    builder_ = std::make_shared<TrtBuilder>(name);
  }

  virtual ~TrtGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph* graph, const std::vector<Parameter>& entry_params,
      const std::vector<Parameter>& return_params,
      const std::vector<InputOutputAlias>& aliases) override;

 private:
  void SetupKernelContextParam(const XrtNode* node,
                               TrtOpContext::Param* context_param);

  void PopulateEntryParams(const std::vector<Parameter>& entry_params);

  Argument ArgFromParameter(const Parameter& param);

 private:
  std::shared_ptr<TrtBuilder> builder_;

  std::unordered_map<std::string, Argument> arguments_;
  std::unordered_map<Argument, TrtValue> operands_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_TENSORRT_TRT_GRAPH_COMPILER_H_
