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
#include "oneflow_xrt/api/api.h"

#include <fstream>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "oneflow_xrt/compiler/passes/build_subgraph_pass.h"
#include "oneflow_xrt/compiler/passes/mark_cluster_id_pass.h"
#ifdef WITH_TENSORRT
#include "oneflow_xrt/compiler/tensorrt/trt_int8_calibrator.h"
#endif  // WITH_TENSORRT

namespace oneflow {
namespace xrt {

std::shared_ptr<XrtGraph> RunClusterSubGraphPass(
    const XrtGraph* graph, const ClusteringOptions& options) {
  auto new_graph = RunMarkClusterIdPass(graph, options);
  return RunBuildSubGraphPass(new_graph.get(), options);
}

Parameter BuildParameter(const std::string& name,
                         const user_op::Tensor* tensor) {
  Shape shape;
  tensor->shape().ToShape(&shape);
  return Parameter(name, const_cast<void*>(tensor->dptr()), shape,
                   tensor->data_type());
}

#ifdef WITH_TENSORRT
void CacheInt8Calibration() {
  const auto& calib_resources = TRTInt8CalibratorResource::All();
  for (const auto& res : calib_resources) {
    std::lock_guard<std::mutex> lock(res.second->mutex_);
    if (!res.second->calibrator_->isDone()) {
      res.second->calibrator_->waitAndSetDone();
      res.second->thread_->join();
    }
    res.second->calibrator_->ReleaseDevBuffers();
  }
}

void WriteInt8Calibration(const std::string& path) {
  const auto& calib_resources = TRTInt8CalibratorResource::All();
  for (const auto& res : calib_resources) {
    CHECK(res.second->calibrator_->isDone())
        << "Calibration table maybe has not been generated "
        << "since the calibrator has not been done.";

    const std::string& calibration_table_data =
        res.second->calibrator_->getCalibrationTableAsString();
    CHECK(calibration_table_data.size()) << "Calibration table data is empty.";

    std::string calib_store_path =
        absl::StrCat(path, "/", res.first /*calibrator name*/);
    std::ofstream ofile(calib_store_path, std::ios::out);
    CHECK(ofile.good()) << "Could not open calibration file: "
                        << calib_store_path;
    ofile << calibration_table_data;
    ofile.close();
  }
}
#endif  // WITH_TENSORRT

}  // namespace xrt
}  // namespace oneflow
