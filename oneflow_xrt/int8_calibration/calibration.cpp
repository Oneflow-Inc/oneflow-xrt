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
#include "oneflow_xrt/int8_calibration/calibration.h"

#include <fstream>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "oneflow/core/vm/vm_sync.h"

namespace oneflow {
namespace xrt {

static std::unordered_map<std::string, Int8CalibratorResource*> resources;

/*static*/ bool Int8CalibratorResource::Record(const std::string& name,
                                               Int8CalibratorResource* res) {
  return resources.emplace(name, res).second;
}

/*static*/ Int8CalibratorResource* Int8CalibratorResource::Lookup(
    const std::string& name) {
  const auto& it = resources.find(name);
  return it == resources.end() ? NULL : it->second;
}

/*static*/ const std::unordered_map<std::string, Int8CalibratorResource*>&
Int8CalibratorResource::All() {
  return resources;
}

void CacheInt8Calibration() {
  // synchronize oneflow virtual machine to ensure that the kernel has been
  // complete executed
  vm::CurrentRankSync();
  const auto& calib_resources = Int8CalibratorResource::All();
  for (const auto& res : calib_resources) {
    res.second->WaitAndSetDone();
  }
}

void CacheAndWriteInt8Calibration(const std::string& path) {
  // synchronize oneflow virtual machine to ensure that the kernel has been
  // complete executed
  vm::CurrentRankSync();
  const auto& calib_resources = Int8CalibratorResource::All();
  for (const auto& res : calib_resources) {
    if (!res.second->IsDone()) {
      res.second->WaitAndSetDone();
    }
    const std::string& calibration_table_data =
        res.second->GetCalibrationTableAsString();
    std::string calib_store_path =
        absl::StrCat(path, "/", res.first /*calibrator name*/);
    std::ofstream ofile(calib_store_path, std::ios::out);
    CHECK(ofile.good()) << "Could not open calibration file: "
                        << calib_store_path;
    ofile << calibration_table_data;
    ofile.close();
  }
}

}  // namespace xrt
}  // namespace oneflow
