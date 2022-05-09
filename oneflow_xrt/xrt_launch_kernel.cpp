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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow_xrt/api/api.h"
#include "oneflow_xrt/common/device.h"
#include "oneflow_xrt/compiler/compilation_cache.h"
#include "oneflow_xrt/compiler/executable.h"
#include "oneflow_xrt/compiler/graph_compiler.h"

using google::protobuf::TextFormat;

namespace oneflow {

class XrtLaunchKernelState : public user_op::OpKernelState {
 public:
  XrtLaunchKernelState(const xrt::XrtLaunchProto& proto,
                       const ParallelDesc& parallel_desc)
      : proto_(proto),
        parallel_desc_(parallel_desc),
        compilation_cache_(new xrt::CompilationCache) {
    for (const auto& entry : proto.liveout_entries()) {
      liveout_entries_.insert(entry);
    }
  }
  const xrt::XrtLaunchProto& proto() const { return proto_; }
  const std::set<std::string>& liveout_entries() { return liveout_entries_; }

  const ParallelDesc& parallel_desc() const { return parallel_desc_; }

  xrt::CompilationCache* compilation_cache() {
    return compilation_cache_.get();
  }

 private:
  xrt::XrtLaunchProto proto_;
  std::set<std::string> liveout_entries_;
  ParallelDesc parallel_desc_;
  std::shared_ptr<xrt::CompilationCache> compilation_cache_;
};

class XrtLaunchKernel : public user_op::OpKernel {
 public:
  XrtLaunchKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override;

 private:
  void Compute(user_op::KernelComputeContext* ctx,
               user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override;

  std::shared_ptr<xrt::Executable> BuildExecutable(
      user_op::KernelComputeContext* ctx, XrtLaunchKernelState* launch_state,
      const std::vector<xrt::Parameter>& entry_params,
      const std::vector<xrt::Parameter>& return_params,
      const std::vector<xrt::InputOutputAlias>& aliases,
      const xrt::XrtEngine& engine, const xrt::XrtDevice& device,
      const int device_ordinal) const;

  void MakeInputOutputAlias(const std::set<std::string>& liveout_entries,
                            const std::vector<xrt::Parameter>& entry_params,
                            std::vector<xrt::Parameter>* return_params,
                            std::vector<xrt::InputOutputAlias>* aliases) const;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

std::shared_ptr<user_op::OpKernelState> XrtLaunchKernel::CreateOpKernelState(
    user_op::KernelInitContext* ctx) const {
  const auto& string_proto = ctx->Attr<std::string>("proto");
  xrt::XrtLaunchProto proto;
  if (!TextFormat::ParseFromString(string_proto, &proto)) {
    LOG(FATAL) << "failed to parse proto for xrt launch op " << ctx->op_name();
  }
  return std::make_shared<XrtLaunchKernelState>(proto, ctx->parallel_desc());
}

std::shared_ptr<xrt::Executable> XrtLaunchKernel::BuildExecutable(
    user_op::KernelComputeContext* ctx, XrtLaunchKernelState* launch_state,
    const std::vector<xrt::Parameter>& entry_params,
    const std::vector<xrt::Parameter>& return_params,
    const std::vector<xrt::InputOutputAlias>& aliases,
    const xrt::XrtEngine& engine, const xrt::XrtDevice& device,
    const int device_ordinal) const {
  VLOG(2) << "build an executable for launch op " << ctx->op_name();
  auto graph = xrt::BuildGraph(launch_state->proto().function());

  std::map<std::string, BlobDesc> entry_blob_descs;
  for (const auto& input : ctx->inputs()) {
    std::string name = absl::StrCat(input.first, "_", input.second);
    const auto& tensor_desc = ctx->TensorDesc4ArgNameAndIndex(
        /*name*/ input.first, /*index*/ input.second);
    entry_blob_descs.emplace(
        name, BlobDesc(tensor_desc->shape(), tensor_desc->data_type(),
                       tensor_desc->is_dynamic()));
  }
  // the infered shape will be filled to the graph
  xrt::ShapeInferenceContext context(
      &entry_blob_descs, &launch_state->proto().logical_blob_descs(),
      &ctx->parallel_ctx(), &launch_state->parallel_desc(),
      &launch_state->proto().nd_sbp_signatures());
  xrt::RunShapeInferencePass(graph.get(), context);

  xrt::GraphCompiler compiler(ctx->op_name(), engine, device, device_ordinal);
  return compiler.Compile(graph.get(), entry_params, return_params, aliases);
}

void XrtLaunchKernel::MakeInputOutputAlias(
    const std::set<std::string>& liveout_entries,
    const std::vector<xrt::Parameter>& entry_params,
    std::vector<xrt::Parameter>* return_params,
    std::vector<xrt::InputOutputAlias>* aliases) const {
  for (int i = 0; i < entry_params.size(); ++i) {
    const std::string& entry_name = entry_params[i].name();
    if (liveout_entries.count(entry_name) > 0) {
      aliases->push_back(
          {{static_cast<int>(return_params->size())} /*output_index*/,
           i /*param_number=*/,
           {} /*param_index=*/});
      return_params->push_back(entry_params[i]);
    }
  }
}

void XrtLaunchKernel::Compute(user_op::KernelComputeContext* ctx,
                              user_op::OpKernelState* state,
                              const user_op::OpKernelCache*) const {
  auto* launch_state = dynamic_cast<XrtLaunchKernelState*>(state);
  CHECK_NOTNULL(launch_state);

  // prepare input and output parameters
  std::vector<xrt::Parameter> entry_params, return_params;
  for (const auto& input : ctx->inputs()) {
    std::string name = absl::StrCat(input.first, "_", input.second);
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex(
        /*name*/ input.first, /*index*/ input.second);
    entry_params.emplace_back(xrt::BuildParameter(name, input_tensor));
  }
  for (const auto& output : ctx->outputs()) {
    std::string name = absl::StrCat(output.first, "_", output.second);
    const user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex(
        /*name*/ output.first, /*index*/ output.second);
    return_params.emplace_back(xrt::BuildParameter(name, output_tensor));
  }

  std::vector<xrt::InputOutputAlias> aliases;
  MakeInputOutputAlias(launch_state->liveout_entries(), entry_params,
                       &return_params, &aliases);

  const auto& options = launch_state->proto().options();
  xrt::XrtDevice device = options.device();
  int device_ordinal = xrt::GetDeviceId(device);

  xrt::Executable* executable = nullptr;
  xrt::Signature signature =
      xrt::ComputeSignature(ctx->op_name(), device_ordinal, entry_params);
  if (!options.force_compile()) {
    executable = launch_state->compilation_cache()->GetRecord(signature);
  }
  if (!executable) {
    auto result =
        BuildExecutable(ctx, launch_state, entry_params, return_params, aliases,
                        options.engine(), device, device_ordinal);
    if (!result) {
      LOG(FATAL) << "failed to build an executable";
    }
    executable = result.get();
    launch_state->compilation_cache()->Record(signature, result);
  }

  xrt::ExecutableRunOptions run_options;
  run_options.device_ordinal = device_ordinal;
  run_options.return_params = return_params;
  bool block_until_done = true;
  if (device == xrt::XrtDevice::GPU_CUDA) {
#ifdef WITH_CUDA
    run_options.stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    block_until_done = false;
#else
    UNIMPLEMENTED()
        << "CUDA device is not supported since XRT was compiled without CUDA";
#endif  // WITH_CUDA
  }
  bool status = executable->Run(entry_params, run_options, block_until_done);
  CHECK(status) << "failed to run executable";

  const std::vector<xrt::Parameter>& results = executable->Results();
  CHECK_EQ(results.size(), return_params.size());
  for (int i = 0; i < results.size(); ++i) {
    CHECK_EQ(results[i].data(), return_params[i].data());
  }
}

REGISTER_USER_KERNEL(xrt::_XrtLaunchOpType)
    .SetCreateFn<XrtLaunchKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

}  // namespace oneflow
