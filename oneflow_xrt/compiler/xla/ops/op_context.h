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
#ifndef ONEFLOW_XRT_COMPILER_XLA_OPS_OP_CONTEXT_H_
#define ONEFLOW_XRT_COMPILER_XLA_OPS_OP_CONTEXT_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow_xrt/common/device.h"
#include "oneflow_xrt/compiler/kernel/op_context.h"
#include "oneflow_xrt/graph/argument.h"
#include "oneflow_xrt/xrt.pb.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaValue {
 public:
  XlaValue() : initialized_(false) {}
  // Construct from Constant shape.
  static XlaValue Constant(const xla::Shape shape);
  // Construct from XlaOp handle.
  static XlaValue XlaOp(const xla::XlaOp handle);

  // Return the XlaOp handle if the builder is matched with the handle.
  xla::XlaOp AsXlaOp(xla::XlaBuilder* builder) const;

  friend class XlaOpContext;

 private:
  bool initialized_;
  // XlaOp handle should be initialized if the value is
  // constructed from another XlaOp.
  xla::XlaOp handle_;
  // Shape of the xla value.
  xla::Shape shape_;
};

class XlaOpContext : public OpContext {
 public:
  struct Param {
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder* builder;

    XrtDevice device;
    // attributes related to the op
    AttrMap attrs;
    // input operands
    std::unordered_map<Argument, XlaValue> inputs;
    std::vector<std::string> output_names;
    int num_outputs;

    std::unordered_map<std::string, Argument> arguments;
  };

  explicit XlaOpContext(const Param& param)
      : OpContext(param.attrs), param_(param) {}

  virtual ~XlaOpContext() = default;

  const XrtDevice& device() const { return param_.device; }
  // returns XlaBuilder
  xla::XlaBuilder* builder() const;

  const std::string& SoleOutputName() const;

  // returns input named `name` as XlaOp
  xla::XlaOp Input(const std::string& name);
  xla::XlaOp Input(const Argument& arg);
  xla::XlaOp SoleInput();

  // returns output named `name` as XlaOp
  xla::XlaOp Output(const std::string& name);
  xla::XlaOp Output(const Argument& arg);
  xla::XlaOp SoleOutput();

  xla::XlaOp Variable();

  int num_inputs() const { return param_.inputs.size(); }
  int num_outputs() const { return param_.num_outputs; }
  // returns inputs as XlaValues
  const std::unordered_map<Argument, XlaValue>& inputs() const {
    return param_.inputs;
  }
  // returns output as XlaValues
  const std::unordered_map<Argument, XlaValue>& outputs() const {
    return outputs_;
  }

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;
  // setup the output `output_name` with XlaOp
  void SetOutput(const std::string& name, const xla::XlaOp& handle);
  // setup the output `output_name` with XlaValue
  void SetOutput(const std::string& name, const XlaValue& handle);
  void SetSoleOutput(const xla::XlaOp& handle);

  void SetVariable(const xla::XlaOp& handle);

  // returns input `name` shape as Shape
  Shape InputShape(const std::string& name) const;
  Shape SoleInputShape() const;
  // returns output `name` shape as Shape
  Shape OutputShape(const std::string& name) const;
  Shape SoleOutputShape() const;

  // input data type
  DataType InputType(const std::string& name) const;
  DataType SoleInputType() const;
  // output data type
  DataType OutputType(const std::string& name) const;
  DataType SoleOutputType() const;

  const Param& param() const { return param_; }

 private:
  XlaOpContext() = delete;
  Argument ArgumentFromKey(const std::string& key) const;

  Param param_;
  // output operands
  std::unordered_map<Argument, XlaValue> outputs_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_XLA_OPS_OP_CONTEXT_H_
