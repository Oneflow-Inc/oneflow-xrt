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
#ifndef ONEFLOW_XRT_COMPILER_TENSORRT_TRT_EXECUTABLE_H_
#define ONEFLOW_XRT_COMPILER_TENSORRT_TRT_EXECUTABLE_H_

#include <vector>

#include "NvInfer.h"
#include "oneflow_xrt/compiler/executable.h"
#include "oneflow_xrt/compiler/parameter.h"
#include "oneflow_xrt/compiler/tensorrt/trt_int8_calibrator.h"
#include "oneflow_xrt/compiler/tensorrt/trt_unique_ptr.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

class TrtExecutable : public Executable {
 public:
  explicit TrtExecutable(
      const std::string& name, nv::unique_ptr<nvinfer1::ICudaEngine>&& engine,
      const std::map<std::string, std::shared_ptr<std::vector<uint8_t>>>&
          host_weights)
      : Executable(name, XrtEngine::TENSORRT),
        engine_(std::move(engine)),
        host_weights_(host_weights) {}

  explicit TrtExecutable(
      const std::string& name, nv::unique_ptr<nvinfer1::IBuilder>&& builder,
      nv::unique_ptr<nvinfer1::INetworkDefinition>&& network,
      const std::map<std::string, std::shared_ptr<std::vector<uint8_t>>>&
          host_weights)
      : Executable(name, XrtEngine::TENSORRT),
        builder_(std::move(builder)),
        network_(std::move(network)),
        host_weights_(host_weights) {}

  virtual ~TrtExecutable() = default;

  bool Run(const std::vector<Parameter>& inputs,
           const ExecutableRunOptions& run_options,
           bool block_until_done = true) override;

 private:
  nvinfer1::ICudaEngine* CreateExecutableEngine(
      const ExecutableRunOptions& run_options, const int batch_size = 1,
      TRTInt8Calibrator* calibrator = nullptr);

  bool ExecuteEngine(const int batch_size, void** buffers, void* stream,
                     bool block_until_done);

  std::string LoadCalibrationTable(const std::string& calibration_path);

  int32_t GetBindingIndex(const std::string& name) const;

 private:
  nv::unique_ptr<nvinfer1::ICudaEngine> engine_;
  nv::unique_ptr<nvinfer1::IBuilder> builder_;
  nv::unique_ptr<nvinfer1::INetworkDefinition> network_;
  nv::unique_ptr<nvinfer1::IExecutionContext> execution_context_;

  std::shared_ptr<TRTInt8Calibrator> calibrator_;

  std::map<std::string, std::shared_ptr<std::vector<uint8_t>>> host_weights_;
  std::map<std::string, int32_t> bindings_;
};

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_TENSORRT_TRT_EXECUTABLE_H_
