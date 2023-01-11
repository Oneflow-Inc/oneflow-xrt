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
#ifndef ONEFLOW_XRT_COMPILER_PASSES_OPTIONS_H_
#define ONEFLOW_XRT_COMPILER_PASSES_OPTIONS_H_

#include <string>

#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

struct ClusteringOptions {
  XrtEngine engine = XrtEngine::DEFAULT;
  XrtDevice device = XrtDevice::CPU_X86;

  // minimum node number in each cluster after clustering. If the number of
  // nodes contained by a cluster is less than `minimum_nodes` or grater than
  // `maximum_nodes`, then this cluster will be discard and not compiled
  int32_t minimum_nodes = 0x1;
  int32_t maximum_nodes = 0x7fffffff;

  // ignore strict dependencies analysis
  bool ignore_pipeline = true;

  // maximum iteration count for iteratively clustering. -1 means
  // that it will always iteratively merge as much as possible until no
  // node can be merged
  int32_t max_iteration = 20;

  std::string dump_subgraph_dir = "";
};

struct ReBuildJobOptions {
  bool use_fp16 = false;
  bool use_int8 = false;

  std::string int8_calibration = "";

  bool force_compile = false;
  bool strict_types = false;
  bool force_precision_constraints = true;

  int64_t max_batch_size = 1;
  int64_t max_workspace_size = -1;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_PASSES_OPTIONS_H_
