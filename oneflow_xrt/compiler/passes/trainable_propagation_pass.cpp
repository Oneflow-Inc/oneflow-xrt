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
#include "oneflow_xrt/compiler/passes/trainable_propagation_pass.h"

namespace oneflow {
namespace xrt {

std::shared_ptr<XrtGraph> TrainablePropagationPass(const XrtGraph* graph) {
  auto new_graph = graph->clone();
  algorithm::TopologyVisit(*new_graph, [&](XrtNode* node) {
    if (node->trainable()) {
      return;
    }
    bool trainable = false;
    for (const auto& edge : node->in_edges()) {
      if (edge->start()->trainable()) {
        trainable = true;
        break;
      }
    }
    node->set_trainable(trainable);
  });
  return new_graph;
}

}  // namespace xrt
}  // namespace oneflow
