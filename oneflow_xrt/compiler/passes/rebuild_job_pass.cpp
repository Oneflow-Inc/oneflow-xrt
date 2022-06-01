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
#include "flatbuffers/minireflect.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow_xrt/common/typedef.h"
#include "oneflow_xrt/common/protobuf.h"
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

void FixupOpInputBlobName(OperatorConf* op_conf, const std::string& input,
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
  FoldSubgraphBuilder(XrtGraph* graph, Job* job,
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
  void buildFunction(const XrtNode* launch_node, const XrtEngine& engine,
                     std::set<std::string>* liveout_entries,
                     FunctionProtoT* function) const;

  void FixupControlInOpNames();

  void BuildXrtLaunchOps();

  void FixupInOutBlobNames();

  void RemoveLaunchFoldedOps();

  std::string FixedName(const std::string& name);

 private:
  XrtGraph* graph_;
  const ReBuildJobOptions options_;

  std::shared_ptr<JobBuilder> builder_;

  std::vector<const XrtNode*> launch_nodes_;

  // the folded nodes except for argument/return nodes for each xrt launch node
  std::vector<std::vector<const XrtNode*>> folded_nodes_;

  std::map<std::string, std::string> fixedup_names_;
  std::map<std::string /*op name*/, XrtLaunchProtoT> launch_attrs_;
};

FoldSubgraphBuilder::FoldSubgraphBuilder(XrtGraph* graph, Job* job,
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
      if (!sub_node->IsEntryNode() && !sub_node->IsReturnNode()) {
        folded_nodes_[i].emplace_back(sub_node);
      }
    }
  }
  builder_ = std::make_shared<JobBuilder>(job);
}

std::string FoldSubgraphBuilder::FixedName(const std::string& name) {
  const auto& it = fixedup_names_.find(name);
  if (it != fixedup_names_.end()) {
    return it->second;
  }
  return name;
}

void FoldSubgraphBuilder::buildFunction(const XrtNode* launch_node,
                                        const XrtEngine& engine,
                                        std::set<std::string>* liveout_entries,
                                        FunctionProtoT* function) const {
  for (const XrtNode* node : launch_node->sub_graph()->Nodes()) {
    if (node->IsEntryNode()) {
      std::string value;
      bool is_mutable = false;
      for (const XrtEdge* edge : node->out_edges()) {
        const XrtNode* next_node = edge->end();
        is_mutable |=
            IsMutableVariable(edge->argument(), next_node->type(), engine);
        value = edge->argument().name();
      }
      if (is_mutable) {
        liveout_entries->insert(node->name());
      }
      
      function->input.emplace_back(new FunctionArgumentProtoT);
      function->input.back()->name = node->name();
      function->input.back()->value = value;
    } else if (node->IsReturnNode()) {
      const auto* in_edge = node->in_edges().front();
      function->output.emplace_back(new FunctionArgumentProtoT);
      function->output.back()->name = node->name();
      function->output.back()->value = in_edge->argument().name();
    } else {
      function->node.emplace_back(protobuf::PrintToTextString(node->conf()));
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
    std::string consume_key = absl::StrCat(_XrtEntryName, "_", i);
    CHECK_GT(input_args.count(consume_key), 0);
    const std::string& val = input_args.at(consume_key);
    (*launch_conf->mutable_input())[_XrtEntryName].add_s(val);
  }

  std::map<std::string, std::string> output_args;
  for (const XrtEdge* edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      output_args.emplace(arg.meta_data().produce_key, arg.name());
    }
  }
  for (int i = 0; i < output_args.size(); ++i) {
    std::string produce_key = absl::StrCat(_XrtReturnName, "_", i);
    CHECK_GT(output_args.count(produce_key), 0);
    (*launch_conf->mutable_output())[_XrtReturnName].add_s(
        absl::StrCat(node->name(), "/", produce_key));
  }
}

void FoldSubgraphBuilder::BuildXrtLaunchOps() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    const XrtNode* node = launch_nodes_[i];
    {
      // make xrt launch op
      OperatorConf op_conf;
      op_conf.set_name(node->name());
      DeviceType device_type = XrtDeviceToOfDevice(node->device());
      op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type)));

      UserOpConf* launch_conf = op_conf.mutable_user_conf();
      launch_conf->set_op_type_name(_XrtLaunchOpType);

      // add inputs and outputs
      AddInOutBlobNames(node, launch_conf);

      CHECK_GT(folded_nodes_[i].size(), 0);
      const ParallelConf& parallel_conf = CHECK_JUST(
          builder_->ParallelConf4OpName(folded_nodes_[i][0]->name()));
      builder_->AddOps(parallel_conf, {op_conf});

      // update xrt launch op nd sbp signatures
      NdSbpSignature nd_sbp_signature;
      auto* bn_in_op2nd_sbp = nd_sbp_signature.mutable_bn_in_op2nd_sbp();
      for (const XrtEdge* edge : node->in_edges()) {
        const auto& meta_data = edge->argument().meta_data();
        (*bn_in_op2nd_sbp)[meta_data.consume_key] = meta_data.nd_sbp[1];
      }
      for (const XrtEdge* edge : node->out_edges()) {
        const auto& meta_data = edge->argument().meta_data();
        (*bn_in_op2nd_sbp)[meta_data.produce_key] = meta_data.nd_sbp[0];
      }
      builder_->AddNdSbpSignature4OpName(node->name(), nd_sbp_signature);
    }

    std::set<std::string> liveout_entries;
    XrtLaunchProtoT& proto = launch_attrs_[node->name()];

    const auto& engine = node->sub_graph()->engine();
    // add execute options
    proto.options.reset(new ExecuteOptionsProtoT);
    proto.options->engine = engine;
    proto.options->device = node->device();
    proto.options->use_fp16 = options_.use_fp16;
    proto.options->use_int8 = options_.use_int8;
    proto.options->int8_calibration = options_.int8_calibration;
    proto.options->force_compile = options_.force_compile;
    proto.options->strict_types = options_.strict_types;
    proto.options->force_precision_constraints =
        options_.force_precision_constraints;
    proto.options->max_batch_size = options_.max_batch_size;
    proto.options->max_workspace_size = options_.max_workspace_size;

    proto.function.reset(new FunctionProtoT);
    // build function
    buildFunction(node, engine, &liveout_entries, proto.function.get());

    // add liveout entries
    for (const auto& entry : liveout_entries) {
      proto.liveout_entries.emplace_back(entry);
    }

    // save function logical blob descs
    const auto& lbn2logical_blob_desc =
        builder_->job().helper().lbn2logical_blob_desc();
    const auto& op_name2arg_signature =
        builder_->job().helper().op_name2arg_signature();

    std::map<std::string, BlobDescProto> logical_blob_descs;
    // auto* logical_blob_descs = proto.mutable_logical_blob_descs();
    auto CopyLogicalBlobDesc = [&](const std::string& arg_name) {
      const auto src_it = lbn2logical_blob_desc.find(arg_name);
      CHECK(src_it != lbn2logical_blob_desc.end());
      auto dst_it = logical_blob_descs.find(arg_name);
      if (dst_it != logical_blob_descs.end()) {
        CHECK(dst_it->second == src_it->second);
      } else {
        logical_blob_descs[arg_name] = src_it->second;
      }
    };
    for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (sub_node->IsEntryNode() || sub_node->IsReturnNode()) {
        continue;
      }
      const auto& it = op_name2arg_signature.find(sub_node->name());
      CHECK(it != op_name2arg_signature.end());
      for (const auto& p : it->second.bn_in_op2lbi()) {
        std::string arg_name = GenLogicalBlobName(p.second);
        CopyLogicalBlobDesc(arg_name);
      }
    }
    // save the launch op outputs logical blob descs
    for (const auto& output : proto.function->output) {
      const auto& it = logical_blob_descs.find(output->value);
      CHECK(it != logical_blob_descs.end());
      logical_blob_descs[output->name] = it->second;
    }
    for (const auto& it : logical_blob_descs) {
      std::unique_ptr<StringVecT> v(new StringVecT);
      v->data.emplace_back(it.first);
      v->data.emplace_back(protobuf::PrintToTextString(it.second));
      proto.logical_blob_descs.emplace_back(std::move(v));
    }

    // save sbp signatures for the folded nodes
    std::map<std::string, NdSbpSignature> nd_sbp_signatures;
    for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (sub_node->IsEntryNode() || sub_node->IsReturnNode()) {
        continue;
      }
      const std::string& node_name = sub_node->name();
      nd_sbp_signatures[node_name] =
          builder_->NdSbpSignature4OpName(node_name);
    }
    nd_sbp_signatures[node->name()] =
        builder_->NdSbpSignature4OpName(node->name());
    for (const auto& it : nd_sbp_signatures) {
      std::unique_ptr<StringVecT> v(new StringVecT);
      v->data.emplace_back(it.first);
      v->data.emplace_back(protobuf::PrintToTextString(it.second));
      proto.logical_blob_descs.emplace_back(std::move(v));
    }

    // update xrt launch op attribute `proto`
    auto* op_conf = CHECK_JUST(builder_->MutableOpConf4OpName(node->name()));
    auto* attr = op_conf->mutable_user_conf()->mutable_attr();
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(XrtLaunchProto::Pack(fbb, &proto));
    auto s = flatbuffers::FlatBufferToString(fbb.GetBufferPointer(), XrtLaunchProtoTypeTable());
    (*attr)["proto"].set_at_string(s);
  }
}

void FoldSubgraphBuilder::FixupControlInOpNames() {
  CHECK_EQ(launch_nodes_.size(), folded_nodes_.size());
  // mapping the folded node name to cluster node
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
        if (sub_node->IsEntryNode() || sub_node->IsReturnNode()) {
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
    std::map<std::string, std::string> consume_names;
    for (XrtEdge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      const Argument& arg = edge->argument();
      int index = consume_names.size();
      auto it = consume_names.find(arg.name());
      if (it == consume_names.end()) {
        it = consume_names
                 .emplace(arg.name(), absl::StrCat(_XrtEntryName, "_", index))
                 .first;
      }
      edge->argument().meta_data().consume_key = it->second;
    }

    // fixup output blob names
    std::map<std::string, std::string> produce_names;
    for (XrtEdge* edge : node->out_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      const Argument& arg = edge->argument();
      int index = produce_names.size();
      auto it = produce_names.find(arg.name());
      if (it == produce_names.end()) {
        CHECK_EQ(fixedup_names_.count(arg.name()), 0);
        it = produce_names
                 .emplace(arg.name(), absl::StrCat(_XrtReturnName, "_", index))
                 .first;
      }
      std::string fixed_blob_name =
          absl::StrCat(launch_op_name, "/", it->second);
      fixedup_names_.emplace(arg.name(), fixed_blob_name);
      // fixup end input blob name
      const XrtNode* end = edge->end();
      if (end->type() != _XrtLaunchOpType) {
        auto* op_conf = CHECK_JUST(builder_->MutableOpConf4OpName(end->name()));
        const std::string& consume_key = arg.meta_data().consume_key;
        FixupOpInputBlobName(op_conf, consume_key, arg.name(), fixed_blob_name);
      }
      // update output edge argument
      edge->argument().meta_data().produce_key = it->second;
      edge->argument().set_name(fixed_blob_name);
    }

    // fix subgraph entry and return nodes name
    for (XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (sub_node->IsEntryNode()) {
        std::string fixed_name = FixedName(sub_node->name());
        const auto& it = consume_names.find(fixed_name);
        CHECK(it != consume_names.end());
        sub_node->set_name(it->second);
      } else if (sub_node->IsReturnNode()) {
        const auto& it = produce_names.find(sub_node->name());
        CHECK(it != produce_names.end());
        sub_node->set_name(it->second);
      }
    }
  }
}

void FoldSubgraphBuilder::RemoveLaunchFoldedOps() {
  std::unordered_set<std::string> removing_names;
  for (const XrtNode* node : launch_nodes_) {
    for (const XrtNode* sub_node : node->sub_graph()->Nodes()) {
      if (!sub_node->IsEntryNode() && !sub_node->IsReturnNode()) {
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
std::shared_ptr<Job> RunRebuildJobPass(const XrtGraph* graph, const Job& origin,
                                       const ReBuildJobOptions& options) {
  auto job = std::make_shared<Job>(origin);
  auto new_graph = graph->clone();
  FoldSubgraphBuilder(new_graph.get(), job.get(), options).Build();
  return job;
}

}  // namespace xrt
}  // namespace oneflow
