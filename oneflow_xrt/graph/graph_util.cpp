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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

namespace detail {

class GraphBuilder {
 public:
  explicit GraphBuilder(const Job& job);

  explicit GraphBuilder(const FunctionProto& function);

  std::shared_ptr<XrtGraph> Build() {
    BuildGraphEdges();
    UpdateEdgeTimeShapeAndNdSbp();
    return graph_;
  }

 private:
  ArgumentMetaData MakeMetaData(const XrtNode* start, const XrtNode* end,
                                const std::string& arg_name);

  void BuildGraphEdges();
  void UpdateEdgeTimeShapeAndNdSbp();

  struct NodeInfo {
    std::set<std::string> inputs;
    std::map<std::string, std::string> input_output_keys;
    Shape time_shape;
    std::map<std::string, NdSbp> nd_sbp;
  };

 private:
  std::shared_ptr<XrtGraph> graph_;
  std::map<std::string, const XrtNode*> producers_;
  std::map<const XrtNode*, NodeInfo> node_info_;
};

GraphBuilder::GraphBuilder(const Job& job)
    : graph_(std::make_shared<XrtGraph>()) {
  auto op_graph = std::make_shared<OpGraph>(job);
  op_graph->TopoForEachNode([&](const OpNode* op_node) {
    const Operator& op = op_node->op();
    XrtNode* node = graph_->AddNode(op.op_conf());
    auto& node_info = node_info_[node];
    for (const std::string& bn : op.output_bns()) {
      std::string output = GenLogicalBlobName(op.BnInOp2Lbi(bn));
      producers_[output] = node;
      node_info.input_output_keys[output] = bn;
      node_info.nd_sbp[output] = op_node->NdSbp4BnInOp(bn);
    }
    for (const std::string& bn : op.input_bns()) {
      std::string input = GenLogicalBlobName(op.BnInOp2Lbi(bn));
      node_info.input_output_keys[input] = bn;
      node_info.inputs.insert(input);
      node_info.nd_sbp[input] = op_node->NdSbp4BnInOp(bn);
    }
    node_info.time_shape = *CHECK_JUST(op.GetOpTimeShape());
  });
}

GraphBuilder::GraphBuilder(const FunctionProto& function)
    : graph_(std::make_shared<XrtGraph>()) {
  for (const auto& input : function.input()) {
    XrtNode* node = graph_->AddEntryNode(input.name());
    producers_[input.value()] = node;
    node_info_[node].input_output_keys.emplace(input.value(), "value");
  }
  for (const auto& output : function.output()) {
    XrtNode* node = graph_->AddReturnNode(output.name());
    node_info_[node].inputs.insert(output.value());
    node_info_[node].input_output_keys.emplace(output.value(), "value");
  }

  for (const auto& node_conf : function.node()) {
    XrtNode* node = graph_->AddNode(node_conf);
    auto& input_output_keys = node_info_[node].input_output_keys;
    auto op = CHECK_JUST(ConstructOp(node_conf));
    for (const std::string& bn : op->output_bns()) {
      std::string output = GenLogicalBlobName(op->BnInOp2Lbi(bn));
      producers_[output] = node;
      input_output_keys[output] = bn;
    }
    for (const std::string& bn : op->input_bns()) {
      std::string input = GenLogicalBlobName(op->BnInOp2Lbi(bn));
      input_output_keys[input] = bn;
      node_info_[node].inputs.insert(input);
    }
  }
}

ArgumentMetaData GraphBuilder::MakeMetaData(const XrtNode* start,
                                            const XrtNode* end,
                                            const std::string& arg_name) {
  ArgumentMetaData meta_data;
  const auto& produce_keys = node_info_.at(start).input_output_keys;
  const auto& consume_keys = node_info_.at(end).input_output_keys;
  meta_data.produce_key = produce_keys.at(arg_name);
  meta_data.consume_key = consume_keys.at(arg_name);
  return meta_data;
}

void GraphBuilder::BuildGraphEdges() {
  for (const auto& p : node_info_) {
    const XrtNode* node = p.first;
    const std::set<std::string>& inputs = p.second.inputs;
    for (const std::string& input : inputs) {
      const auto& it = producers_.find(input);
      if (it != producers_.end() && it->second != node) {
        Argument argument(input, MakeMetaData(it->second, node, input));
        graph_->Connect(it->second, node, argument);
      }
    }
  }
}

void GraphBuilder::UpdateEdgeTimeShapeAndNdSbp() {
  for (XrtEdge* edge : graph_->Edges()) {
    const auto& src = node_info_.at(edge->start());
    const auto& dst = node_info_.at(edge->end());
    const std::string& name = edge->argument().name();

    auto& meta_data = edge->argument().meta_data();
    meta_data.time_shape[0] = src.time_shape;
    meta_data.time_shape[1] = dst.time_shape;
    if (!src.nd_sbp.empty() && !dst.nd_sbp.empty()) {
      meta_data.nd_sbp[0] = src.nd_sbp.at(name);
      meta_data.nd_sbp[1] = dst.nd_sbp.at(name);
    }
  }
}

}  // namespace detail

std::shared_ptr<XrtGraph> BuildGraph(const FunctionProto& function) {
  return detail::GraphBuilder(function).Build();
}

std::shared_ptr<XrtGraph> BuildGraph(const Job& job) {
  return detail::GraphBuilder(job).Build();
}

}  // namespace xrt
}  // namespace oneflow
