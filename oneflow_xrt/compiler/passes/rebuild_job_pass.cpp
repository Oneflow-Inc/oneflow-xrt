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
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow_xrt/compiler/passes/options.h"
#include "oneflow_xrt/graph/argument.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/graph/node_util.h"

namespace oneflow {
namespace xrt {

template <typename T>
void DoNoDuplicationAdd(google::protobuf::RepeatedPtrField<T>* repeat_field,
                        const T& val) {
  if (std::find(repeat_field->begin(), repeat_field->end(), val) ==
      repeat_field->end()) {
    repeat_field->Add()->assign(val);
  }
}

int GetRepeatedIndex(const std::string& input) {
  auto name_and_index = GetFieldNameAndIndex4StrVal(input);
  return name_and_index.second;
};

void SetOpInputBlobName(OperatorConf* op_conf, const std::string& input,
                        const std::string& blob_name,
                        const std::string& fixed_blob_name) {
  auto* spec_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  switch (op_conf->op_type_case()) {
    case OperatorConf::kUserConf: {
      std::pair<std::string, int32_t> pair = GetFieldNameAndIndex4StrVal(input);
      auto it = op_conf->user_conf().input().find(pair.first);
      CHECK(it != op_conf->user_conf().input().end());
      CHECK(pair.second >= 0 && pair.second < it->second.s_size());
      CHECK_EQ(it->second.s(pair.second), blob_name);
      (*(op_conf->mutable_user_conf()->mutable_input()))[pair.first].set_s(
          pair.second, fixed_blob_name);
      break;
    }
    default: {
      const auto& old_val =
          ReplaceStrValInPbFdOrPbRpf(spec_conf, input, fixed_blob_name);
      CHECK_EQ(old_val, blob_name);
    }
  }
}

class FoldSubgraphBuilder {
 public:
  FoldSubgraphBuilder(const XrtGraph* graph, Job* job,
                      const ReBuildJobOptions& options);

  virtual ~FoldSubgraphBuilder() {}

  void Build() {
    // 1.Fixup output blob names for launch nodes, and infect the
    //   changes to the input of next nodes
    FixupInOutBlobNames();
    // 2.Add XrtLaunch operator to the job
    BuildXrtLaunchOps();
    // 3.Replace control_in_op_name by the XrtLaunch operator name if
    //   the operator has been folded
    FixupControlInOpNames();
    // 4.Finally remove the folded operators
    RemoveLaunchFoldedOps();
  }

 private:
  void buildFunction(const XrtGraph* sub_graph, const XrtEngine& engine,
                     std::set<std::string>* liveout_entries,
                     FunctionProto* function) const;

  void FixupControlInOpNames();

  void BuildXrtLaunchOps();

  void FixupInOutBlobNames();

  void RemoveLaunchFoldedOps();

 private:
  const XrtGraph* graph_;
  const ReBuildJobOptions options_;

  std::shared_ptr<JobBuilder> builder_;

  std::vector<const XrtNode*> launch_nodes_;

  // the folded nodes except for argument/return nodes for each xrt launch node
  std::vector<std::vector<const XrtNode*>> folded_nodes_;

  std::map<std::string, std::string> fixedup_names_;
  std::map<std::string /*op name*/, XrtLaunchProto> launch_attrs_;
};

FoldSubgraphBuilder::FoldSubgraphBuilder(const XrtGraph* graph, Job* job,
                                         const ReBuildJobOptions& options)
    : graph_(graph), options_(options) {
  for (const XrtNode* node : graph_->Nodes()) {
    if (node->type() == _XrtLaunchOpType) {
      launch_nodes_.emplace_back(node);
    }
  }

  folded_nodes_.resize(launch_nodes_.size());
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    XrtGraph* sub_graph = launch_nodes_[i]->sub_graph();
    CHECK_NOTNULL(sub_graph);
    for (const XrtNode* sub_node : sub_graph->Nodes()) {
      if (!sub_node->IsArgumentNode() && !sub_node->IsReturnNode()) {
        folded_nodes_[i].emplace_back(sub_node);
      }
    }
  }
  builder_ = std::make_shared<JobBuilder>(job);
}

bool IsMutableArgument(const Argument& argument, const std::string& op_type,
                       const XrtEngine& engine) {
  // const auto& mutable_vars = MutableVariables(op_type, field);
  // const std::string& key = argument.meta_data().consume_key;
  // return mutable_vars.count(key) > 0;
  return false;
}

void FoldSubgraphBuilder::buildFunction(const XrtGraph* sub_graph,
                                        const XrtEngine& engine,
                                        std::set<std::string>* liveout_entries,
                                        FunctionProto* function) const {
  for (const XrtNode* node : sub_graph->Nodes()) {
    if (node->IsArgumentNode()) {
      function->add_input(node->name());
      bool is_mutable = false;
      if (is_mutable) {
        liveout_entries->insert(node->name());
      }
    } else if (node->IsReturnNode()) {
      function->add_output(node->name());
    } else {
      *(function->add_node()) =
          *reinterpret_cast<const OperatorConf*>(&node->conf());
    }
  }
}

void AddInOutBlobNames(const XrtNode* node, UserOpConf* launch_conf) {
  std::map<std::string, std::string> input_args;
  for (const XrtEdge* edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      input_args.emplace(arg.meta_data().consume_key, arg.name());
    }
  }
  for (int i = 0; i < input_args.size(); ++i) {
    std::string consume_key = absl::StrCat("in_", i);
    CHECK_GT(input_args.count(consume_key), 0);
    const std::string& val = input_args.at(consume_key);
    (*launch_conf->mutable_input())["input"].mutable_s()->Add()->assign(val);
  }

  std::map<std::string, std::string> output_args;
  for (const XrtEdge* edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      output_args.emplace(arg.meta_data().produce_key, arg.name());
    }
  }
  for (int i = 0; i < output_args.size(); ++i) {
    std::string produce_key = absl::StrCat("out_", i);
    CHECK_GT(output_args.count(produce_key), 0);
    (*launch_conf->mutable_output())["output"].mutable_s()->Add()->assign(
        produce_key);
  }
}

void FoldSubgraphBuilder::BuildXrtLaunchOps() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    const XrtNode* node = launch_nodes_[i];
    {
      // add xrt launch op
      OperatorConf op_conf;
      op_conf.set_name(node->name());
      DeviceType device_type = XrtDeviceToOfDevice(node->device());
      op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type)));

      UserOpConf* launch_conf = op_conf.mutable_user_conf();
      launch_conf->set_op_type_name("xrt_launch");

      // add inputs and outputs
      AddInOutBlobNames(node, launch_conf);

      CHECK_GT(folded_nodes_[i].size(), 0);
      const ParallelConf& parallel_conf = CHECK_JUST(
          builder_->ParallelConf4OpName(folded_nodes_[i][0]->name()));
      builder_->AddOps(parallel_conf, {op_conf});
    }

    // update xrt launch attr
    XrtLaunchProto& proto = launch_attrs_[node->name()];
    std::set<std::string> liveout_entries;
    buildFunction(node->sub_graph(), options_.engine, &liveout_entries,
                  proto.mutable_function());

    for (auto input : proto.function().input()) {
      const auto& it = fixedup_names_.find(input);
      if (it != fixedup_names_.end()) {
        input = it->second;
      }
      if (liveout_entries.count(input) > 0) {
        proto.add_liveout_entries(input);
      }
    }

    auto CopyLogicalBlobDesc4Lbn = [&](const std::string& lbn) -> void {
      const auto& src_map = builder_->job().helper().lbn2logical_blob_desc();
      auto* dst_map = proto.mutable_logical_blob_desc();
      const auto src_it = src_map.find(lbn);
      CHECK(src_it != src_map.end());
      auto dst_it = dst_map->find(lbn);
      if (dst_it != dst_map->end()) {
        CHECK(dst_it->second == src_it->second);
      } else {
        (*dst_map)[lbn] = src_it->second;
      }
    };

    const auto& op_name2arg_signature =
        builder_->job().helper().op_name2arg_signature();
    for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (sub_node->IsArgumentNode() || sub_node->IsReturnNode()) {
        continue;
      }
      const auto op_name2arg_signature_it =
          op_name2arg_signature.find(sub_node->name());
      CHECK(op_name2arg_signature_it != op_name2arg_signature.end());
      for (const auto& pair : op_name2arg_signature_it->second.bn_in_op2lbi()) {
        const LogicalBlobId& lbi = pair.second;
        std::string blob_name = GenLogicalBlobName(lbi);
        CopyLogicalBlobDesc4Lbn(blob_name);
      }
    }

    NdSbpSignature nd_sbp_signature;
    auto* bn_in_op2nd_sbp = nd_sbp_signature.mutable_bn_in_op2nd_sbp();
    for (const XrtEdge* edge : node->in_edges()) {
      const std::string& bn = edge->argument().meta_data().consume_key;
      (*bn_in_op2nd_sbp)[bn] = edge->nd_sbp[1];
    }
    for (const XrtEdge* edge : node->out_edges()) {
      const std::string& bn = edge->argument().meta_data().produce_key;
      (*bn_in_op2nd_sbp)[bn] = edge->nd_sbp[0];
    }
    // append nd sbp signatures
    builder_->AddNdSbpSignature4OpName(node->name(), nd_sbp_signature);

    // save sbp signatures for the folded nodes
    auto* nd_sbp_signatures = proto.mutable_nd_sbp_signatures();
    for (const auto& node_conf : proto.function().node()) {
      const std::string& node_name = node_conf.name();
      (*nd_sbp_signatures)[node_name] =
          builder_->NdSbpSignature4OpName(node_name);
    }
    (*nd_sbp_signatures)[node->name()] = nd_sbp_signature;

    std::string xrt_launch_attr;
    google::protobuf::TextFormat::PrintToString(proto, &xrt_launch_attr);
    auto* op_conf = CHECK_JUST(builder_->MutableOpConf4OpName(node->name()));
    auto* attr = op_conf->mutable_user_conf()->mutable_attr();
    (*attr)["proto"].set_at_string(xrt_launch_attr);
  }
}

void FoldSubgraphBuilder::FixupControlInOpNames() {
  CHECK_EQ(launch_nodes_.size(), folded_nodes_.size());
  // map folded node names to cluster node
  std::map<std::string, const XrtNode*> folded_op_names;
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    for (const XrtNode* node : folded_nodes_[i]) {
      folded_op_names.emplace(node->name(), launch_nodes_[i]);
    }
  }

  auto AddControlInOpName = [&](OperatorConf* conf,
                                const std::string& op_name) -> void {
    std::string ctrl_in_op_name = op_name;
    const auto& it = folded_op_names.find(op_name);
    if (it != folded_op_names.end()) {
      ctrl_in_op_name = it->second->name();
    }
    if (conf->name() != ctrl_in_op_name) {
      DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
    }
  };

  for (const XrtNode* node : graph_->Nodes()) {
    auto* op_conf = CHECK_JUST(builder_->MutableOpConf4OpName(node->name()));
    if (!node->sub_graph()) {
      auto ctrl_in_op_names = op_conf->ctrl_in_op_name();
      op_conf->clear_ctrl_in_op_name();
      for (const auto& op_name : ctrl_in_op_names) {
        AddControlInOpName(op_conf, op_name);
      }
    } else {
      for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->IsArgumentNode() || sub_node->IsReturnNode()) {
          continue;
        }
        const auto& folded_op_conf =
            CHECK_JUST(builder_->OpConf4OpName(sub_node->name()));
        for (const auto& op_name : folded_op_conf.ctrl_in_op_name()) {
          AddControlInOpName(op_conf, op_name);
        }
      }
    }
  }
}

void FoldSubgraphBuilder::FixupInOutBlobNames() {
  for (const XrtNode* node : launch_nodes_) {
    std::string launch_op_name = node->name();
    // fixup input arguments consume key
    std::map<std::string, int> consume_argument_names;
    for (XrtEdge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      const Argument& arg = edge->argument();
      int index = consume_argument_names.size();
      auto it = consume_argument_names.find(arg.name());
      if (it == consume_argument_names.end()) {
        it = consume_argument_names.emplace(arg.name(), index).first;
      }
      index = it->second;

      ArgumentMetaData metadata;
      metadata.consume_key = absl::StrCat("in_", index);
      metadata.produce_key = arg.meta_data().produce_key;
      Argument fixed_arg(arg.name(), arg.shape(), arg.data_type(), metadata);
      edge->SetArgument(fixed_arg);
    }

    // fixup output blob names
    std::map<std::string, int> produce_argument_names;
    for (XrtEdge* edge : node->out_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      const Argument& arg = edge->argument();
      int index = produce_argument_names.size();
      auto it = produce_argument_names.find(arg.name());
      if (it == produce_argument_names.end()) {
        CHECK_EQ(fixedup_names_.count(arg.name()), 0);
        it = produce_argument_names.emplace(arg.name(), index).first;
      }
      index = it->second;

      std::string fixed_blob_name =
          absl::StrCat(launch_op_name, "/out_", index);
      fixedup_names_.emplace(arg.name(), fixed_blob_name);
      // fixup end input blob name
      const XrtNode* end = edge->end();
      if (end->type() != _XrtLaunchOpType) {
        auto* op_conf = CHECK_JUST(builder_->MutableOpConf4OpName(end->name()));
        const std::string& consume_key = arg.meta_data().consume_key;
        SetOpInputBlobName(op_conf, consume_key, arg.name(), fixed_blob_name);
      }
      ArgumentMetaData metadata;
      metadata.consume_key = arg.meta_data().consume_key;
      metadata.produce_key = absl::StrCat("out_", index);
      Argument fixed_arg(fixed_blob_name, arg.shape(), arg.data_type(),
                         metadata);
      edge->SetArgument(fixed_arg);
    }
  }
}

void FoldSubgraphBuilder::RemoveLaunchFoldedOps() {
  std::unordered_set<std::string> removing_names;
  for (const XrtNode* node : launch_nodes_) {
    for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (!sub_node->IsArgumentNode() && !sub_node->IsReturnNode()) {
        removing_names.insert(sub_node->name());
      }
    }
  }
  builder_->RemoveOpByName(removing_names);
}

// rebuild job according to the nodes folded xrt graph. In order to rebuild
// the job, We will add several launch operators in the job, and remove the
// folded operators. In each launch operator, we wll reconstruct the subgraph
// and insert argument nodes if necessary
std::shared_ptr<Job> RunRebuildJobPass(XrtGraph* graph, const Job& origin,
                                       const ReBuildJobOptions& options) {
  auto job = std::make_shared<Job>(origin);
  FoldSubgraphBuilder(graph, job.get(), options).Build();
  return job;
}

}  // namespace xrt
}  // namespace oneflow
