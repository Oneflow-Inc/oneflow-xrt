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
#include "oneflow_xrt/compiler/tensorrt/trt_graph_compiler.h"

#include "oneflow_xrt/compiler/tensorrt/ops/op_kernel.h"
#include "oneflow_xrt/graph/node_util.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

void TrtGraphCompiler::PopulateEntryParams(
    const std::vector<Parameter>& entry_params) {
  for (const Parameter& param : entry_params) {
    Argument arg = ArgFromParameter(param);
    TrtValue value = TrtValue::Parameter(builder_.get(), param);
    operands_[arg] = std::move(value);
    arguments_.emplace(param.name(), arg);
  }
}

Argument TrtGraphCompiler::ArgFromParameter(const Parameter& param) {
  return Argument(param.name(), param.shape(), param.data_type());
}

void TrtGraphCompiler::SetupKernelContextParam(
    const XrtNode* node, TrtOpContext::Param* context_param) {
  std::unordered_map<Argument, TrtValue> input_ops;
  std::unordered_map<std::string /* produce/consume key */, Argument>
      input_output_args;
  std::vector<std::string> output_names;
  if (node->IsEntryNode()) {
    const Argument& arg = arguments_.at(node->name());
    input_output_args.emplace("variable", arg);
    input_ops.emplace(arg, operands_.at(arg));
  } else if (node->IsReturnNode()) {
    const Argument& arg = arguments_.at(node->name());
    input_output_args.emplace("variable", arg);
    output_names.emplace_back(node->name());
  }
  for (const XrtEdge* edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      CHECK_GT(operands_.count(arg), 0);
      const TrtValue& operand = operands_.at(arg);
      input_ops.emplace(arg, operand);
      const std::string& k = arg.meta_data().consume_key;
      input_output_args.emplace(k, arg);
    }
  }
  for (const XrtEdge* edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      const std::string& k = arg.meta_data().produce_key;
      input_output_args.emplace(k, arg);
      output_names.emplace_back(k);
    }
  }
  size_t num_outputs = input_output_args.size() - input_ops.size();
  CHECK_GE(num_outputs, 0) << "Outputs number should >= 0";
  context_param->op_name = node->name();
  context_param->builder = builder_.get();
  context_param->attrs = node->attrs();
  context_param->arguments = std::move(input_output_args);
  context_param->inputs = std::move(input_ops);
  context_param->output_names = std::move(output_names);
  context_param->num_outputs = num_outputs;
}

std::shared_ptr<Executable> TrtGraphCompiler::Compile(
    const XrtGraph* graph, const std::vector<Parameter>& entry_params,
    const std::vector<Parameter>& return_params,
    const std::vector<InputOutputAlias>& aliases) {
  // build entry trt values
  PopulateEntryParams(entry_params);
  std::vector<Argument> return_args(return_params.size());
  for (int i = 0; i < return_params.size(); ++i) {
    return_args[i] = ArgFromParameter(return_params[i]);
    arguments_.emplace(return_params[i].name(), return_args[i]);
  }

  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    if (node->IsNoOpNode()) {
      return;
    }
    TrtOpContext::Param param;
    SetupKernelContextParam(node, &param);
    TrtOpContext op_context(param);
    // do compile
    auto op_kernel = BuildOpKernel(node->type());
    op_kernel->Compile(&op_context);

    // always insert the new output into `operands_`
    const auto& outputs = op_context.outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      operands_[it->first] = it->second;
    }
  });

  for (const auto& arg : return_args) {
    const TrtValue& value = operands_.at(arg);
    builder_->MarkOutput(value.handle());
  }
  for (const auto& arg : arguments_) {
    TrtValue& value = operands_.at(arg.second);
    if (IsTensorKind(value.ValueKind(builder_.get()))) {
      value.AsTensor(builder_.get())->setName(arg.first.data());
    }
  }
  return std::make_shared<TrtExecutable>(
      builder_->name(), builder_->ReleaseBuilder(), builder_->ReleaseNetwork(),
      builder_->host_weights());
}

REGISTER_GRAPH_COMPILER(XrtEngine::TENSORRT, TrtGraphCompiler);

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
