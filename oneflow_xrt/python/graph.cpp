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

using namespace oneflow::xrt;

void InitXrtGraphApis(py::module_& m) {
  py::class_<XrtGraph, std::shared_ptr<XrtGraph>>(m, "Graph")
      .def(py::init([](const std::string& serialized_job) {
        oneflow::Job job;
        if (!job.ParseFromString(serialized_job)) {
          PyErr_SetString(PyExc_RuntimeError,
                          "the first argument is not a valid job");
        }
        return BuildGraph(job);
      }));
}
