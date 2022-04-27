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
#include "oneflow_xrt/compiler/passes/options.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace oneflow::xrt;

void InitClusteringOptionsApis(py::module_& m) {
  py::class_<ClusteringOptions, std::shared_ptr<ClusteringOptions>>(
      m, "ClusteringOptions")
      .def(py::init())
      .def_property(
          "engine", /*getter*/
          [](const ClusteringOptions& opt) {
            return XrtEngine_Name(opt.engine);
          },
          /*setter*/
          [](ClusteringOptions& opt, const std::string& engine) {
            XrtEngine _engine;
            XrtEngine_Parse(engine, &_engine);
            opt.engine = _engine;
          })
      .def_property(
          "device", /*getter*/
          [](const ClusteringOptions& opt) {
            return XrtDevice_Name(opt.device);
          },
          /*setter*/
          [](ClusteringOptions& opt, const std::string& device) {
            XrtDevice _device;
            XrtDevice_Parse(device, &_device);
            opt.device = _device;
          })
      .def_property(
          "minimum_nodes", /*getter*/
          [](const ClusteringOptions& opt) { return opt.minimum_nodes; },
          /*setter*/
          [](ClusteringOptions& opt, const int32_t& minimum_nodes) {
            opt.minimum_nodes = minimum_nodes;
          })
      .def_property(
          "maximum_nodes", /*getter*/
          [](const ClusteringOptions& opt) { return opt.maximum_nodes; },
          /*setter*/
          [](ClusteringOptions& opt, const int32_t& maximum_nodes) {
            opt.maximum_nodes = maximum_nodes;
          })
      .def_property(
          "ignore_pipeline", /*getter*/
          [](const ClusteringOptions& opt) { return opt.ignore_pipeline; },
          /*setter*/
          [](ClusteringOptions& opt, const bool& ignore_pipeline) {
            opt.ignore_pipeline = ignore_pipeline;
          })
      .def_property(
          "max_iteration", /*getter*/
          [](const ClusteringOptions& opt) { return opt.max_iteration; },
          /*setter*/
          [](ClusteringOptions& opt, const int32_t& max_iteration) {
            opt.max_iteration = max_iteration;
          })
      .def_property(
          "dump_subgraph_dir", /*getter*/
          [](const ClusteringOptions& opt) { return opt.dump_subgraph_dir; },
          /*setter*/
          [](ClusteringOptions& opt, const std::string& dump_subgraph_dir) {
            opt.dump_subgraph_dir = dump_subgraph_dir;
          });

  py::class_<ReBuildJobOptions, std::shared_ptr<ReBuildJobOptions>>(
      m, "ReBuildJobOptions")
      .def(py::init())
      .def_property(
          "engine", /*getter*/
          [](const ReBuildJobOptions& opt) {
            return XrtEngine_Name(opt.engine);
          },
          /*setter*/
          [](ReBuildJobOptions& opt, const std::string& engine) {
            XrtEngine _engine;
            XrtEngine_Parse(engine, &_engine);
            opt.engine = _engine;
          })
      .def_property(
          "use_fp16", /*getter*/
          [](const ReBuildJobOptions& opt) { return opt.use_fp16; },
          /*setter*/
          [](ReBuildJobOptions& opt, const bool& use_fp16) {
            opt.use_fp16 = use_fp16;
          })
      .def_property(
          "use_int8", /*getter*/
          [](const ReBuildJobOptions& opt) { return opt.use_int8; },
          /*setter*/
          [](ReBuildJobOptions& opt, const bool& use_int8) {
            opt.use_int8 = use_int8;
          })
      .def_property(
          "force_compile", /*getter*/
          [](const ReBuildJobOptions& opt) { return opt.force_compile; },
          /*setter*/
          [](ReBuildJobOptions& opt, const bool& force_compile) {
            opt.force_compile = force_compile;
          })
      .def_property(
          "int8_calibration", /*getter*/
          [](const ReBuildJobOptions& opt) { return opt.int8_calibration; },
          /*setter*/
          [](ReBuildJobOptions& opt, const std::string& int8_calibration) {
            opt.int8_calibration = int8_calibration;
          })
      .def_property(
          "max_workspace_size", /*getter*/
          [](const ReBuildJobOptions& opt) { return opt.max_workspace_size; },
          /*setter*/
          [](ReBuildJobOptions& opt, const int64_t& max_workspace_size) {
            opt.max_workspace_size = max_workspace_size;
          });
}
