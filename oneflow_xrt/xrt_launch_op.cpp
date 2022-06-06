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
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow_xrt/api/api_internal.h"
#include "oneflow_xrt/common/typedef.h"

using google::protobuf::TextFormat;

namespace oneflow {

Maybe<void> XrtLaunchOpInferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const auto& string_proto = ctx->user_op_conf().attr<std::string>("proto");
  xrt::XrtLaunchProto proto;
  if (!TextFormat::ParseFromString(string_proto, &proto)) {
    return Error::RuntimeError() << "failed to parse proto for xrt launch op "
                                 << ctx->user_op_conf().op_name();
  }
  const std::string& op_name = ctx->user_op_conf().op_name();
  auto it = proto.nd_sbp_signatures().find(op_name);
  if (it == proto.nd_sbp_signatures().end()) {
    return Error::RuntimeError()
           << "can not infer nd sbp for xrt launch op " << op_name;
  }
  const auto& nd_sbp_signature = it->second.bn_in_op2nd_sbp();
  for (const auto& input : ctx->inputs()) {
    std::string name = absl::StrCat(input.first, "_", input.second);
    const auto& it = nd_sbp_signature.find(name);
    if (it == nd_sbp_signature.end()) {
      return Error::RuntimeError() << "missing input (" << name
                                   << ") nd sbp for xrt launch op " << op_name;
    }
    *(ctx->NdSbp4ArgNameAndIndex(/*name*/ input.first,
                                 /*index*/ input.second)) = it->second;
  }
  for (const auto& output : ctx->outputs()) {
    std::string name = absl::StrCat(output.first, "_", output.second);
    const auto& it = nd_sbp_signature.find(name);
    if (it == nd_sbp_signature.end()) {
      return Error::RuntimeError() << "missing output (" << name
                                   << ") nd sbp for xrt launch op " << op_name;
    }
    *(ctx->NdSbp4ArgNameAndIndex(/*name*/ output.first,
                                 /*index*/ output.second)) = it->second;
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOpInferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& string_proto = ctx->Attr<std::string>("proto");
  xrt::XrtLaunchProto proto;
  if (!TextFormat::ParseFromString(string_proto, &proto)) {
    return Error::RuntimeError()
           << "failed to parse proto for xrt launch op " << ctx->op_name();
  }
  const auto& logical_blob_descs = proto.logical_blob_descs();
  for (const auto& output : ctx->outputs()) {
    std::string name = absl::StrCat(output.first, "_", output.second);
    const auto& it = logical_blob_descs.find(name);
    if (it == logical_blob_descs.end()) {
      return Error::RuntimeError()
             << "failed to infer output (" << name
             << ") logical tensor desc for xrt launch op " << ctx->op_name();
    }
    const auto& blob_desc = it->second;
    auto* output_tensor_desc =
        ctx->OutputTensorDesc(/*name*/ output.first, /*index*/ output.second);
    *(output_tensor_desc->mut_shape()) = Shape(blob_desc.shape());
    *(output_tensor_desc->mut_data_type()) = blob_desc.data_type();
    *(output_tensor_desc->mut_is_dynamic()) = blob_desc.is_dynamic();
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOpInferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const auto& string_proto = ctx->Attr<std::string>("proto");
  xrt::XrtLaunchProto proto;
  if (!TextFormat::ParseFromString(string_proto, &proto)) {
    return Error::RuntimeError()
           << "failed to parse proto for xrt launch op " << ctx->op_name();
  }
  std::map<std::string, BlobDesc> entry_blob_descs;
  for (const auto& input : ctx->inputs()) {
    std::string name = absl::StrCat(input.first, "_", input.second);
    const auto& tensor_desc =
        ctx->InputTensorDesc(/*name*/ input.first, /*index*/ input.second);
    entry_blob_descs.emplace(
        name, BlobDesc(tensor_desc.shape(), tensor_desc.data_type(),
                       tensor_desc.is_dynamic()));
  }
  xrt::ShapeInferenceContext context(
      &entry_blob_descs, &proto.logical_blob_descs(), &ctx->parallel_ctx(),
      &ctx->parallel_desc(), &proto.nd_sbp_signatures());
  auto graph = xrt::BuildGraph(proto.function());
  xrt::RunShapeInferencePass(graph.get(), context);

  const auto* infered_blob_descs = context.infered_physical_blob_descs();
  for (const auto& output : ctx->outputs()) {
    std::string name = absl::StrCat(output.first, "_", output.second);
    const auto& it = infered_blob_descs->find(name);
    if (it == infered_blob_descs->end()) {
      return Error::RuntimeError()
             << "failed to infer output (" << name
             << ") physical tensor desc for xrt launch op " << ctx->op_name();
    }
    auto* output_tensor_desc =
        ctx->OutputTensorDesc(/*name*/ output.first, /*index*/ output.second);
    *(output_tensor_desc->mut_shape()) = it->second.shape();
    *(output_tensor_desc->mut_data_type()) = it->second.data_type();
    *(output_tensor_desc->mut_is_dynamic()) = it->second.is_dynamic();
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOpInferDataType(user_op::InferContext* ctx) {
  // data type has been infered while infer TensorDesc
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOpModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn,
    const user_op::UserOpConfWrapper& conf) {
  const auto& string_proto = conf.attr<std::string>("proto");
  xrt::XrtLaunchProto proto;
  if (!TextFormat::ParseFromString(string_proto, &proto)) {
    return Error::RuntimeError()
           << "failed to parse proto for xrt launch op " << conf.op_name();
  }
  for (const auto& entry : proto.liveout_entries()) {
    std::string arg_name = entry;
    int index = 0;
    size_t pos = entry.rfind("_");
    if (pos != std::string::npos) {
      arg_name = entry.substr(0, pos);
      index = std::atoi(entry.substr(pos + 1).data());
    }
    user_op::InputArgModifier* arg_modifier =
        GetInputArgModifierFn(/*name*/ arg_name, /*index*/ index);
    arg_modifier->set_is_mutable(true);
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOpGetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

REGISTER_USER_OP(xrt::_XrtLaunchOpType)
    .Input(xrt::_XrtEntryName)
    .Output(xrt::_XrtReturnName)
    .Attr<std::string>("proto")
    .NoGrad()
    .SetNdSbpInferFn(&XrtLaunchOpInferNdSbp)
    .SetGetSbpFn(&XrtLaunchOpGetSbp)
    .SetLogicalTensorDescInferFn(&XrtLaunchOpInferLogicalTensorDesc)
    .SetPhysicalTensorDescInferFn(&XrtLaunchOpInferPhysicalTensorDesc)
    .SetDataTypeInferFn(&XrtLaunchOpInferDataType)
    .SetInputArgModifyFn(&XrtLaunchOpModifyInputArg);

}  // namespace oneflow
