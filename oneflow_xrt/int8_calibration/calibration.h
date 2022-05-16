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
#ifndef ONEFLOW_XRT_INT8_CALIBRATION_CALIBRATION_H_
#define ONEFLOW_XRT_INT8_CALIBRATION_CALIBRATION_H_

#include <string>
#include <unordered_map>

namespace oneflow {
namespace xrt {

class Int8CalibratorResource {
 public:
  static bool Record(const std::string& name, Int8CalibratorResource* res);
  static Int8CalibratorResource* Lookup(const std::string& name);

  static const std::unordered_map<std::string, Int8CalibratorResource*>& All();

  virtual void WaitAndSetDone() = 0;
  virtual bool IsDone() const = 0;
  virtual std::string GetCalibrationTableAsString() const = 0;
};

void CacheInt8Calibration();
void CacheAndWriteInt8Calibration(const std::string& path);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_INT8_CALIBRATION_CALIBRATION_H_
