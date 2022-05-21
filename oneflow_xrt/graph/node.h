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
#ifndef ONEFLOW_XRT_GRAPH_NODE_H_
#define ONEFLOW_XRT_GRAPH_NODE_H_

#include "oneflow_xrt/common/device.h"
#include "oneflow_xrt/common/typedef.h"
#include "oneflow_xrt/graph/algorithm.h"
#include "oneflow_xrt/graph/argument.h"

namespace oneflow {
namespace xrt {

class XrtNode;
class XrtGraph;

class XrtEdge {
 public:
  XrtNode* start() const { return start_; }
  XrtNode* end() const { return end_; }
  const Argument& argument() const { return arg_; }
  Argument& argument() { return arg_; }

  void SetStart(const XrtNode* start) { start_ = const_cast<XrtNode*>(start); }
  void SetEnd(const XrtNode* end) { end_ = const_cast<XrtNode*>(end); }
  void SetArgument(const Argument& arg) { arg_ = arg; }

  int64_t unique_id() const { return unique_id_; }

  bool IsControlEdge() const { return !arg_.initialized(); }

  friend class XrtGraph;
  virtual ~XrtEdge() = default;

 protected:
  XrtEdge() = default;
  XrtEdge(const XrtNode* start, const XrtNode* end)
      : start_(const_cast<XrtNode*>(start)), end_(const_cast<XrtNode*>(end)) {}

 protected:
  XrtNode* start_ = nullptr;
  XrtNode* end_ = nullptr;
  Argument arg_;
  int64_t unique_id_ = -1;
};

class XrtNode {
 public:
  const std::list<XrtEdge*>& in_edges() const { return in_edges_; }
  const std::list<XrtEdge*>& out_edges() const { return out_edges_; }
  std::list<XrtEdge*>& in_edges() { return in_edges_; }
  std::list<XrtEdge*>& out_edges() { return out_edges_; }

  void AddInEdge(const XrtEdge* edge);
  void AddOutEdge(const XrtEdge* edge);
  void EraseInEdge(const XrtEdge* edge);
  void EraseOutEdge(const XrtEdge* edge);
  void ClearInEdges() { in_edges_.clear(); };
  void ClearOutEdges() { out_edges_.clear(); };

  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }

  const std::string& type() const { return type_; }
  void set_type(const std::string& type) { type_ = type; }

  const OperatorConf& conf() const { return conf_; }
  const AttrMap& attrs() const { return attrs_; }

  int64_t unique_id() const { return unique_id_; }

  const XrtDevice& device() const { return device_; }
  void set_device(const XrtDevice& device) { device_ = device; }

  bool trainable() const { return trainable_; }
  void set_trainable(bool trainable) { trainable_ = trainable; }
  int64_t cluster_id() const { return cluster_id_; }
  void set_cluster_id(int64_t cluster_id) { cluster_id_ = cluster_id; }

  XrtGraph* sub_graph() const { return sub_graph_; }

  std::unique_ptr<XrtNode> clone() const;

  bool IsSourceNode() const;
  bool IsFinishNode() const;
  bool IsEntryNode() const;
  bool IsReturnNode() const;
  bool IsNoOpNode() const;

  bool IsReachable(const XrtNode& dst_node) const;

  friend class XrtGraph;
  virtual ~XrtNode() = default;

 protected:
  XrtNode(const std::string& name)
      : name_(name),
        type_(_XrtUnsupportedOpType),
        unique_id_(-1),
        sub_graph_(nullptr),
        device_(XrtDevice::CPU_X86),
        trainable_(false) {}
  explicit XrtNode(const OperatorConf& conf);

 protected:
  std::list<XrtEdge*> in_edges_;
  std::list<XrtEdge*> out_edges_;
  // node name
  std::string name_;
  // node type, such as "conv2d"
  std::string type_;

  OperatorConf conf_;
  // additional attributes
  AttrMap attrs_;

  // each node has a unique id related to it's index in the graph
  int64_t unique_id_ = -1;

  // the folded subgraph
  // note that the subgraph should be built and managed by the graph, other than
  // the node
  XrtGraph* sub_graph_ = nullptr;

  XrtDevice device_;
  bool trainable_ = false;
  // the folded node will has a cluster id after clustering subgraph
  int64_t cluster_id_ = -1;
};

namespace algorithm {
template <>
struct NodeTypeTrait<XrtNode> {
  typedef XrtEdge* pEdgeType;
};

template <>
struct NodeTypeTrait<const XrtNode> {
  typedef const XrtEdge* pEdgeType;
};
}  // namespace algorithm

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_NODE_H_
