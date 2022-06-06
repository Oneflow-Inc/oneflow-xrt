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
#ifndef ONEFLOW_XRT_API_SERVING_H_
#define ONEFLOW_XRT_API_SERVING_H_

#include <string>
#include <vector>

namespace oneflow {
namespace xrt {

std::string CompileJob(
    const std::string& job, const std::vector<std::string>& engine,
    bool use_fp16 = false, bool use_int8 = false, size_t max_batch_size = 1,
    size_t max_workspace_size = -1, bool strict_types = false,
    bool force_precision_constraints = true, bool force_compile = false,
    size_t cluster_minimum_nodes = 1, size_t cluster_maximum_nodes = 0x7fffffff,
    bool cluster_ignore_pipeline = true, size_t cluster_max_iteration = 20,
    const std::string& dump_subgraph_dir = "");

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_API_SERVING_H_
