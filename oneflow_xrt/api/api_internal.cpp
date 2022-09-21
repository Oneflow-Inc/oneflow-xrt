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
#include "oneflow_xrt/api/api_internal.h"

#include <unordered_map>

#include "oneflow_xrt/compiler/passes/build_subgraph_pass.h"
#include "oneflow_xrt/compiler/passes/mark_cluster_id_pass.h"
#include "oneflow_xrt/compiler/passes/trainable_propagation_pass.h"

namespace oneflow {
namespace xrt {

std::shared_ptr<XrtGraph> RunClusterSubGraphPass(
    const XrtGraph* graph, const ClusteringOptions& options) {
  std::shared_ptr<XrtGraph> new_graph;
  new_graph = TrainablePropagationPass(graph);
  new_graph = RunMarkClusterIdPass(new_graph.get(), options);
  return RunBuildSubGraphPass(new_graph.get(), options);
}

namespace {

std::unordered_map<std::string, std::pair<const void*, Shape>>*
registered_buffers() {
  static std::unordered_map<std::string, std::pair<const void*, Shape>>
      registered_buffers;
  return &registered_buffers;
}

}  // namespace

void RegisterBuffer(const std::string& op_name, const Shape& shape,
                    const void* data) {
  registered_buffers()->emplace(op_name, std::make_pair(data, shape));
}

const std::pair<const void*, Shape>& RegisteredBuffer(
    const std::string& op_name) {
  const auto& it = registered_buffers()->find(op_name);
  CHECK(it != registered_buffers()->end())
      << "buffer not found which name is " << op_name;
  return it->second;
}

}  // namespace xrt
}  // namespace oneflow
