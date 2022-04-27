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
#ifndef ONEFLOW_XRT_COMPILER_EXECUTABLE_H_
#define ONEFLOW_XRT_COMPILER_EXECUTABLE_H_

#include <string>
#include <vector>

#include "oneflow_xrt/compiler/parameter.h"
#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

struct ExecutableRunOptions {
  // specify the compute stream if the engine supports multiple streams
  // the default compute stream will be used if `stream` is nullptr
  void* stream = nullptr;

  int32_t device_ordinal = -1;

  int32_t host_num_threads = -1;

  // memory footprint limit
  int64_t host_memory_limit = -1;
  int64_t device_memory_limit = -1;

  int64_t random_seed = -1;

  // maximum batch size for tensorrt
  int32_t max_batch_size = 1;

  // enable tensorrt mixed-precision
  bool tensorrt_fp16 = false;
  bool tensorrt_int8 = false;

  std::string tensorrt_int8_calibration = "";

  // populate the return parameters to reuse their storages while running
  // the executable
  std::vector<Parameter> return_params;
};

class Executable {
 public:
  Executable(const std::string& name, const XrtEngine& engine)
      : name_(name), engine_(engine) {}
  virtual ~Executable() = default;

  const XrtEngine& engine() const { return engine_; }

  const std::string& name() const { return name_; }

  virtual bool Run(const std::vector<Parameter>& inputs,
                   const ExecutableRunOptions& run_options,
                   bool block_until_done = true) = 0;

  bool RunAsync(const std::vector<Parameter> inputs,
                const ExecutableRunOptions& run_options) {
    return Run(inputs, run_options, false);
  }

  const std::vector<Parameter>& Results() const { return results_; }

 protected:
  std::string name_;
  XrtEngine engine_;
  std::vector<Parameter> results_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_EXECUTABLE_H_
