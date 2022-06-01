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
#ifndef ONEFLOW_XRT_COMPILER_OPENVINO_OPS_OP_CONTEXT_H_
#define ONEFLOW_XRT_COMPILER_OPENVINO_OPS_OP_CONTEXT_H_

#include <inference_engine.hpp>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow_xrt/compiler/kernel/op_context.h"
#include "oneflow_xrt/compiler/parameter.h"
#include "oneflow_xrt/graph/argument.h"
#include "oneflow_xrt/xrt_generated.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoOpContext : public OpContext {
 public:
  struct Param {
    std::string op_name;

    // attributes related to the op
    AttrMap attrs;
    // Input operands
    std::unordered_map<Argument, std::shared_ptr<ngraph::Node>> inputs;
    int input_size;

    std::unordered_map<std::string, Argument> arguments;
  };

  explicit OpenvinoOpContext(
      const Param& param,
      const std::unordered_map<Argument, Parameter>& entry_params_map)
      : OpContext(param.attrs),
        param_(param),
        graph_inputs_(),
        graph_weight_(),
        outputs_(),
        entry_params_map_(entry_params_map) {}

  virtual ~OpenvinoOpContext() = default;

  const Param& param() const { return param_; }

  const std::string& op_name() const { return param_.op_name; }

  // Return input named `name` as tensor
  std::shared_ptr<ngraph::Node> Input(const std::string& name);
  std::shared_ptr<ngraph::Node> Input(const Argument& arg);
  std::shared_ptr<ngraph::Node> Weight(const std::string& name);
  std::shared_ptr<ngraph::Node> Weight(const Argument& arg);
  // Return output named `name` as tensor
  std::shared_ptr<ngraph::Node> Output(const std::string& name);
  std::shared_ptr<ngraph::Node> Output(const Argument& arg);

  std::shared_ptr<ngraph::Node> Variable();

  int num_inputs() const { return param_.input_size; }

  // Return inputs as OpenvinoValues
  const std::unordered_map<Argument, std::shared_ptr<ngraph::Node>>&
  graph_inputs() const {
    return graph_inputs_;
  }
  const std::unordered_map<Argument, std::shared_ptr<ngraph::Node>>&
  graph_weight() const {
    return graph_weight_;
  }
  // Return output as OpenvinoValues
  const std::unordered_map<Argument, std::shared_ptr<ngraph::Node>>& outputs()
      const {
    return outputs_;
  }

  void SetOutput(const std::string& name,
                 const std::shared_ptr<ngraph::Node>& ngraph_node);

  void SetVariable(const std::shared_ptr<ngraph::Node>& ngraph_node);

  // Return input `name` shape as Shape
  Shape InputShape(const std::string& name) const;
  // Return output `name` shape as Shape
  Shape OutputShape(const std::string& name) const;

  // Input data type
  DataType InputType(const std::string& name) const;
  // Output data type
  DataType OutputType(const std::string& name) const;

 private:
  OpenvinoOpContext() = delete;
  Argument ArgumentFromKey(const std::string& key) const;

  Param param_;
  // Output operands
  std::unordered_map<Argument, std::shared_ptr<ngraph::Node>> graph_inputs_;
  std::unordered_map<Argument, std::shared_ptr<ngraph::Node>> graph_weight_;
  std::unordered_map<Argument, std::shared_ptr<ngraph::Node>> outputs_;

  const std::unordered_map<Argument, Parameter>& entry_params_map_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_OPS_OP_CONTEXT_H_
