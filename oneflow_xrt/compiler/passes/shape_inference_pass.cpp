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
#include <string>
#include <vector>

#include "glog/logging.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow_xrt/compiler/passes/shape_inference_context.h"
#include "oneflow_xrt/graph/argument.h"
#include "oneflow_xrt/graph/graph.h"

namespace oneflow {
namespace xrt {

namespace shape_inference {

void InferPhysicalShape(const XrtGraph* graph, ShapeInferenceContext& context) {
  auto* infered_physical_blob_descs = context.infered_physical_blob_descs();
  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    if (node->IsEntryNode()) {
      // entry node must have output edge
      CHECK_GT(node->out_edges().size(), 0);
      const XrtEdge* edge = node->out_edges().front();
      const auto& arg_name = edge->argument().name();
      const auto& it = context.entry_physical_blob_descs()->find(node->name());
      CHECK(it != context.entry_physical_blob_descs()->end());
      infered_physical_blob_descs->emplace(arg_name, it->second);
    } else if (node->IsReturnNode()) {
      // return node must have input edge
      CHECK_GT(node->in_edges().size(), 0);
      const XrtEdge* edge = node->in_edges().front();
      const auto& arg_name = edge->argument().name();
      const auto& it = infered_physical_blob_descs->find(arg_name);
      CHECK(it != infered_physical_blob_descs->end());
      infered_physical_blob_descs->emplace(node->name(), it->second);
    } else {
      auto op = CHECK_JUST(
          ConstructOp(node->conf(), XrtDeviceToOfDevice(node->device())));
      CHECK_JUST(op->FillOpParallelDesc(*context.parallel_desc()));

      const NdSbpSignature& nd_sbp_signature =
          context.nd_sbp_signatures()->at(node->name());
      CHECK_JUST(op->FillNdSbpSignature(nd_sbp_signature));

      auto GetBlobDescFn = [&](const std::string& bn) -> BlobDesc* {
        const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
        std::string blob_name = GenLogicalBlobName(lbi);
        auto it = infered_physical_blob_descs->find(blob_name);
        if (it == infered_physical_blob_descs->end()) {
          it = infered_physical_blob_descs
                   ->emplace(blob_name, BlobDesc(DataType::kInvalidDataType))
                   .first;
        }
        return &(it->second);
      };
      auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
        const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
        std::string blob_name = GenLogicalBlobName(lbi);
        auto it = context.logical_blob_descs()->find(blob_name);
        CHECK(it != context.logical_blob_descs()->end());
        return const_cast<BlobDesc*>(&(it->second));
      };
      CHECK_JUST(op->FillLogicalInBlobDesc(GetLogicalBlobDesc4BnInOp));
      CHECK_JUST(op->FillLogicalOutBlobDesc(GetLogicalBlobDesc4BnInOp));

      // finally infer output blob desc
      CHECK_JUST(
          op->InferOutBlobDescsIf(GetBlobDescFn, context.parallel_context()));
    }
    // update blob desc on the output edges
    for (XrtEdge* edge : node->out_edges()) {
      std::string name = edge->argument().name();
      auto it = infered_physical_blob_descs->find(name);
      CHECK(it != infered_physical_blob_descs->end());
      const auto& metadata = edge->argument().meta_data();
      Argument argument(name, it->second.shape(), it->second.data_type(),
                        metadata);
      edge->SetArgument(argument);
    }
  });
}

}  // namespace shape_inference

void RunShapeInferencePass(const XrtGraph* graph,
                           ShapeInferenceContext& context) {
  shape_inference::InferPhysicalShape(graph, context);
}

}  // namespace xrt
}  // namespace oneflow
