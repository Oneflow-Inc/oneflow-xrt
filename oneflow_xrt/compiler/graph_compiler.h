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
#ifndef ONEFLOW_XRT_COMPILER_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_COMPILER_GRAPH_COMPILER_H_

#include <functional>

#include "oneflow_xrt/common/registry.h"
#include "oneflow_xrt/compiler/executable.h"
#include "oneflow_xrt/compiler/parameter.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

class InputOutputAlias {
 public:
  InputOutputAlias(const std::vector<int>& output_index, int param_number,
                   const std::vector<int>& param_index)
      : param_number_(param_number),
        param_index_(param_index),
        output_index_(output_index) {}

  int param_number() const { return param_number_; }

  const std::vector<int>& param_index() const { return param_index_; }

  const std::vector<int>& output_index() const { return output_index_; }

 private:
  int param_number_;
  std::vector<int> param_index_;
  std::vector<int> output_index_;
};

class GraphCompiler {
 public:
  // internal compiler interface class. It should be inherited and the
  // `Compile` function should be overrided
  class Impl {
   public:
    explicit Impl(const std::string& name) : name_(name) {}
    virtual ~Impl() = default;

    void set_device(const XrtDevice& device) { device_ = device; }

    void set_device_ordinal(int32_t device_ordinal) {
      CHECK_GE(device_ordinal, 0) << "device ordinal should >= 0.";
      device_ordinal_ = device_ordinal;
    }

    virtual std::shared_ptr<Executable> Compile(
        const XrtGraph* graph, const std::vector<Parameter>& entry_params,
        const std::vector<Parameter>& return_params,
        const std::vector<InputOutputAlias>& aliases) = 0;

   protected:
    std::string name_ = "";
    XrtDevice device_;
    int32_t device_ordinal_ = 0;
  };

 public:
  using Factory = std::function<Impl*(const std::string&)>;
  static auto Registry() -> common::Registry<GraphCompiler, XrtEngine>* {
    return common::Registry<GraphCompiler, XrtEngine>::Global();
  }

  GraphCompiler(const std::string& name, const XrtEngine& engine,
                const XrtDevice& device, int32_t device_ordinal)
      : engine_(engine) {
    impl_.reset(GraphCompiler::Registry()->Lookup(engine_)(name));
    CHECK(impl_) << "internal compiler should be built correctly for engine "
                 << engine_;
    impl_->set_device(device);
    impl_->set_device_ordinal(device_ordinal);
  }

  std::shared_ptr<Executable> Compile(
      const XrtGraph* graph, const std::vector<Parameter>& entry_params,
      const std::vector<Parameter>& return_params,
      const std::vector<InputOutputAlias>& aliases) {
    return impl_->Compile(graph, entry_params, return_params, aliases);
  }

  const XrtEngine& engine() const { return engine_; }

 private:
  XrtEngine engine_;
  std::shared_ptr<Impl> impl_;
};

#define REGISTER_GRAPH_COMPILER(Engine, Compiler)                        \
  namespace {                                                            \
  struct _XrtGraphCompiler {                                             \
    _XrtGraphCompiler() {                                                \
      GraphCompiler::Registry()->Register(                               \
          Engine, [](const std::string& name) -> GraphCompiler::Impl* {  \
            return new Compiler(name);                                   \
          });                                                            \
    }                                                                    \
  };                                                                     \
  static _XrtGraphCompiler _xrt_graph_compiler_ __attribute__((unused)); \
  }  // namespace

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_GRAPH_COMPILER_H_
