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
#ifndef ONEFLOW_XRT_COMPILER_PASSES_SHAPE_INFERENCE_CONTEXT_H_
#define ONEFLOW_XRT_COMPILER_PASSES_SHAPE_INFERENCE_CONTEXT_H_

#include <map>
#include <string>

#include "google/protobuf/map.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {
namespace xrt {

struct ShapeInferenceContext {
 public:
  ShapeInferenceContext(
      const std::map<std::string, BlobDesc>* entry_physical_blob_descs,
      const std::map<std::string, BlobDesc>* logical_blob_descs,
      const ParallelContext* parallel_ctx,  // NOLINT
      const ParallelDesc* parallel_desc,
      const google::protobuf::Map<std::string, NdSbpSignature>*
          nd_sbp_signatures)
      : entry_physical_blob_descs_(entry_physical_blob_descs),
        logical_blob_descs_(logical_blob_descs),
        parallel_ctx_(parallel_ctx),
        parallel_desc_(parallel_desc),
        nd_sbp_signatures_(nd_sbp_signatures) {}

  ShapeInferenceContext(
      const std::map<std::string, BlobDesc>* entry_physical_blob_descs,
      const google::protobuf::Map<std::string, BlobDescProto>*
          logical_blob_descs,
      const ParallelContext* parallel_ctx,  // NOLINT
      const ParallelDesc* parallel_desc,
      const google::protobuf::Map<std::string, NdSbpSignature>*
          nd_sbp_signatures)
      : entry_physical_blob_descs_(entry_physical_blob_descs),
        parallel_ctx_(parallel_ctx),
        parallel_desc_(parallel_desc),
        nd_sbp_signatures_(nd_sbp_signatures) {
    for (const auto& it : *logical_blob_descs) {
      internal_logical_blob_descs_.emplace(it.first, BlobDesc(it.second));
    }
    logical_blob_descs_ = &internal_logical_blob_descs_;
  }

  const std::map<std::string, BlobDesc>* entry_physical_blob_descs() const {
    return entry_physical_blob_descs_;
  }
  const std::map<std::string, BlobDesc>* logical_blob_descs() const {
    return logical_blob_descs_;
  }
  const ParallelContext* parallel_context() const { return parallel_ctx_; }
  const ParallelDesc* parallel_desc() const { return parallel_desc_; }

  const google::protobuf::Map<std::string, NdSbpSignature>* nd_sbp_signatures()
      const {
    return nd_sbp_signatures_;
  }

  std::map<std::string, BlobDesc>* infered_physical_blob_descs() {
    return &infered_physical_blob_descs_;
  }

 private:
  const std::map<std::string, BlobDesc>* entry_physical_blob_descs_;
  const std::map<std::string, BlobDesc>* logical_blob_descs_;

  const ParallelContext* parallel_ctx_;
  const ParallelDesc* parallel_desc_;

  const google::protobuf::Map<std::string, NdSbpSignature>* nd_sbp_signatures_;

  std::map<std::string, BlobDesc> internal_logical_blob_descs_;
  std::map<std::string, BlobDesc> infered_physical_blob_descs_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_PASSES_SHAPE_INFERENCE_CONTEXT_H_
