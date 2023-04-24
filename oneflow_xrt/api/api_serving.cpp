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
#include "oneflow_xrt/api/api_serving.h"

#include "oneflow_xrt/api/api_internal.h"

namespace oneflow {
namespace xrt {

std::string CompileJob(const std::string& job,
                       const std::vector<std::string>& engine, bool use_fp16,
                       bool use_int8, size_t max_batch_size,
                       size_t max_workspace_size, bool strict_types,
                       bool force_precision_constraints, bool force_compile,
                       size_t cluster_minimum_nodes,
                       size_t cluster_maximum_nodes,
                       bool cluster_ignore_pipeline,
                       size_t cluster_max_iteration,
                       bool cluster_strict_sbp_policy,
                       const std::string& dump_subgraph_dir) {
  Job job_proto;
  if (!job_proto.ParseFromString(job)) {
    LOG(FATAL) << "invalid serialized job";
  }
  auto graph = BuildGraph(job_proto);
  ClusteringOptions cluster_options;
  cluster_options.minimum_nodes = cluster_minimum_nodes;
  cluster_options.maximum_nodes = cluster_maximum_nodes;
  cluster_options.ignore_pipeline = cluster_ignore_pipeline;
  cluster_options.max_iteration = cluster_max_iteration;
  cluster_options.strict_sbp_policy = cluster_strict_sbp_policy;
  cluster_options.dump_subgraph_dir = dump_subgraph_dir;
  for (const auto& e : engine) {
    XrtEngine xrt_engine;
    XrtEngine_Parse(e, &xrt_engine);
    cluster_options.engine = xrt_engine;
    graph = RunClusterSubGraphPass(graph.get(), cluster_options);
  }

  ReBuildJobOptions options;
  options.use_fp16 = use_fp16;
  options.use_int8 = use_int8;
  options.max_batch_size = max_batch_size;
  options.max_workspace_size = max_workspace_size;
  options.strict_types = strict_types;
  options.force_precision_constraints = force_precision_constraints;
  options.force_compile = force_compile;
  options.dump_subgraph_dir = dump_subgraph_dir;

  auto new_job = RunRebuildJobPass(graph.get(), job_proto, options);
  return new_job->SerializeAsString();
}

}  // namespace xrt
}  // namespace oneflow
