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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oneflow_xrt/api/api.h"

namespace py = pybind11;

using namespace oneflow;
using namespace oneflow::xrt;

extern void InitXrtGraphApis(py::module_& m);
extern void InitClusteringOptionsApis(py::module_& m);
extern void InitReBuildJobOptionsApis(py::module_& m);
extern void InitInt8CalibrationApis(py::module_& m);

PYBIND11_MODULE(_oneflow_xrt_internal, m) {
  m.def("rebuild_job",
        [](XrtGraph* graph, const std::string& serialized_origin_job,
           const ReBuildJobOptions& options) {
          Job origin_job;
          if (!origin_job.ParseFromString(serialized_origin_job)) {
            PyErr_SetString(PyExc_TypeError,
                            "the second argument is not a valid job");
          }
          auto job = RunRebuildJobPass(graph, origin_job, options);
          return py::bytes(job->SerializeAsString());
        });
  m.def("cluster_subgraph", &RunClusterSubGraphPass);

  InitXrtGraphApis(m);
  InitClusteringOptionsApis(m);
  InitReBuildJobOptionsApis(m);
  InitInt8CalibrationApis(m);
}
