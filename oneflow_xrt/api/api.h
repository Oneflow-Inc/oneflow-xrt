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
#ifndef ONEFLOW_XRT_API_H_
#define ONEFLOW_XRT_API_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow_xrt/compiler/parameter.h"
#include "oneflow_xrt/compiler/passes/options.h"
#include "oneflow_xrt/compiler/passes/shape_inference_options.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

extern std::shared_ptr<XrtGraph> BuildGraph(const FunctionProto& function);
extern std::shared_ptr<XrtGraph> BuildGraph(const Job& job);

std::shared_ptr<XrtGraph> RunClusterSubGraphPass(
    XrtGraph* graph, const ClusteringOptions& options);

extern std::shared_ptr<Job> RunRebuildJobPass(XrtGraph* graph,
                                              const Job& origin,
                                              const ReBuildJobOptions& options);

extern void RunShapeInferencePass(XrtGraph* graph,
                                  const ShapeInferenceOptions& options);

Parameter BuildParameter(const std::string& name,
                         const user_op::Tensor* tensor);

#ifdef WITH_TENSORRT
void CacheInt8Calibration();
void WriteInt8Calibration(const std::string& path);
#endif  // WITH_TENSORRT

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_API_H_