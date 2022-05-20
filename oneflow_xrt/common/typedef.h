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
#ifndef ONEFLOW_XRT_COMMON_TYPEDEF_H_
#define ONEFLOW_XRT_COMMON_TYPEDEF_H_

namespace oneflow {
namespace xrt {

constexpr char const _XrtLaunchOpType[] = "XrtLaunch";
constexpr char const _XrtEntryOpType[] = "XrtEntry";
constexpr char const _XrtReturnOpType[] = "XrtReturn";
constexpr char const _XrtNoOpType[] = "XrtNoOp";
constexpr char const _XrtUnsupportedOpType[] = "XrtUnsupported";

constexpr char const _XrtLaunchPrefix[] = "_xrt_launch_";
constexpr char const _XrtEntryName[] = "_xrt_entry";
constexpr char const _XrtReturnName[] = "_xrt_return";

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMMON_TYPEDEF_H_
