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
#include "oneflow_xrt/graph/node.h"

#include "oneflow_xrt/graph/algorithm.h"

namespace oneflow {
namespace xrt {

XrtNode::XrtNode(const OperatorConf& conf)
    : name_(conf.name()),
      conf_(conf),
      unique_id_(-1),
      sub_graph_(nullptr),
      trainable_(false) {
  if (conf.has_user_conf()) {
    const auto& user_conf = conf.user_conf();
    type_ = user_conf.op_type_name();
    attrs_ = MakeAttrMapFromUserOpConf(user_conf);
  } else if (conf.has_variable_conf()) {
    type_ = "Variable";
    trainable_ = conf.variable_conf().has_trainable() &&
                 conf.variable_conf().trainable();
  } else {
    type_ = _XrtUnsupportedOpType;
  }
  device_ = OfDeviceToXrtDevice(conf.device_tag());
}

void XrtNode::AddInEdge(const XrtEdge* edge) {
  in_edges_.emplace_back(const_cast<XrtEdge*>(edge));
}

void XrtNode::AddOutEdge(const XrtEdge* edge) {
  out_edges_.emplace_back(const_cast<XrtEdge*>(edge));
}

void XrtNode::EraseInEdge(const XrtEdge* edge) {
  in_edges_.remove_if([&](const XrtEdge* e) -> bool {
    return e->unique_id() == edge->unique_id();
  });
}

void XrtNode::EraseOutEdge(const XrtEdge* edge) {
  out_edges_.remove_if([&](const XrtEdge* e) -> bool {
    return e->unique_id() == edge->unique_id();
  });
}

bool XrtNode::IsSourceNode() const { return in_edges_.size() == 0; }

bool XrtNode::IsFinishNode() const { return out_edges_.size() == 0; }

bool XrtNode::IsEntryNode() const { return type_ == _XrtEntryOpType; }

bool XrtNode::IsReturnNode() const { return type_ == _XrtReturnOpType; }

bool XrtNode::IsNoOpNode() const { return type_ == _XrtNoOpType; }

bool XrtNode::IsReachable(const XrtNode& dst_node) const {
  return algorithm::IsReachable(this, &dst_node);
}

std::unique_ptr<XrtNode> XrtNode::clone() const {
  std::unique_ptr<XrtNode> node(new XrtNode(name_));
  node->type_ = type_;
  node->conf_ = conf_;
  if (conf_.has_user_conf()) {
    node->attrs_ = MakeAttrMapFromUserOpConf(conf_.user_conf());
  }
  node->device_ = device_;
  node->cluster_id_ = cluster_id_;
  return node;
}

}  // namespace xrt
}  // namespace oneflow
