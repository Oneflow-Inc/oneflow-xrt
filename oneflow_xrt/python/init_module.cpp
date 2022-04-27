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

#include "google/protobuf/text_format.h"
#include "oneflow_xrt/api/api.h"

namespace py = pybind11;

using namespace oneflow;
using namespace oneflow::xrt;
using google::protobuf::TextFormat;

extern void InitXrtGraphApis(py::module_& m);
extern void InitClusteringOptionsApis(py::module_& m);

PYBIND11_MODULE(_oneflow_xrt_internal, m) {
  m.def("rebuild_job", [](XrtGraph* graph, const std::string& origin,
                          const ReBuildJobOptions& options) {
    Job _origin;
    if (!_origin.ParseFromString(origin)) {
      PyErr_SetString(PyExc_TypeError, "origin is not a valid job");
    }
    auto job = RunRebuildJobPass(graph, _origin, options);
    std::string output;
    TextFormat::PrintToString(*job, &output);
    return output;
  });
  m.def("cluster_subgraph", &RunClusterSubGraphPass);

  InitXrtGraphApis(m);
  InitClusteringOptionsApis(m);
}
