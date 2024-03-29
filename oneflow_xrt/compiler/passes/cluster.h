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
#ifndef ONEFLOW_XRT_COMPILER_PASSES_CLUSTER_H_
#define ONEFLOW_XRT_COMPILER_PASSES_CLUSTER_H_

#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/graph/node_util.h"

namespace oneflow {
namespace xrt {

class ClusterNode;

class ClusterEdge {
 public:
  ClusterEdge() = default;
  ClusterEdge(ClusterNode* start, ClusterNode* end)
      : start_(start), end_(end) {}
  virtual ~ClusterEdge() {}

  void SetStart(ClusterNode* start) { start_ = start; }
  void SetEnd(ClusterNode* end) { end_ = end; }

  ClusterNode* start() const { return start_; }
  ClusterNode* end() const { return end_; }

  bool is_control_edge() const { return is_control_edge_; }
  void set_is_control_edge(bool is_control_edge) {
    is_control_edge_ = is_control_edge;
  }

  bool is_fusion_disabled() const { return is_fusion_disabled_; }
  void set_is_fusion_disabled(bool is_fusion_disabled) {
    is_fusion_disabled_ = is_fusion_disabled;
  }

  bool IsIdentity() const;

  const NdSbp& start_nd_sbp() const { return nd_sbp_[0]; }
  const NdSbp& end_nd_sbp() const { return nd_sbp_[1]; }

  void set_start_nd_sbp(const NdSbp& nd_sbp) { nd_sbp_[0] = nd_sbp; }
  void set_end_nd_sbp(const NdSbp& nd_sbp) { nd_sbp_[1] = nd_sbp; }

  const Shape& start_time_shape() const { return time_shape_[0]; }
  const Shape& end_time_shape() const { return time_shape_[1]; }

  void set_start_time_shape(const Shape& shape) { time_shape_[0] = shape; }
  void set_end_time_shape(const Shape& shape) { time_shape_[1] = shape; }

 protected:
  ClusterNode* start_;
  ClusterNode* end_;
  NdSbp nd_sbp_[2];
  Shape time_shape_[2];

  bool is_control_edge_ = false;
  bool is_fusion_disabled_ = false;
};

class ClusterNode {
 public:
  ClusterNode() : ClusterNode(nullptr, -1) {}
  explicit ClusterNode(int64_t id) : ClusterNode(nullptr, id) {}
  explicit ClusterNode(const XrtNode* node, int64_t id)
      : xrt_node_(node), cluster_id_(id) {
    folded_nodes_.insert(this);
  }
  virtual ~ClusterNode() {}

  std::set<ClusterEdge*>& in_edges() { return in_edges_; }
  std::set<ClusterEdge*>& out_edges() { return out_edges_; }
  const std::set<ClusterEdge*>& in_edges() const { return in_edges_; }
  const std::set<ClusterEdge*>& out_edges() const { return out_edges_; }

  void AddInEdge(const ClusterEdge* edge);
  void AddOutEdge(const ClusterEdge* edge);
  void EraseInEdge(const ClusterEdge* edge);
  void EraseOutEdge(const ClusterEdge* edge);
  void ClearInEdges() { in_edges_.clear(); }
  void ClearOutEdges() { out_edges_.clear(); }

  void FoldNodes(const std::set<ClusterNode*>& nodes) {
    folded_nodes_.insert(nodes.begin(), nodes.end());
  }

  void Merge(ClusterNode& other);
  bool TryMerge(ClusterNode& other, bool strict_sbp_policy);
  bool IsReachable(const ClusterNode& target) const;
  bool IsSatisfySbpPolicy() const;
  bool IsSourceNode() const { return in_edges_.empty(); }

  virtual bool IsCompiled(const XrtEngine& engine) const {
    return IsCanbeCompiledNode(xrt_node_, engine, device());
  }

  virtual bool IsModelUpdate() const { return IsModelUpdateNode(xrt_node_); }

  bool operator==(const ClusterNode& other) const {
    return cluster_id_ == other.cluster_id_;
  }

  const XrtNode* xrt_node() const { return xrt_node_; }
  int64_t cluster_id() const { return cluster_id_; }
  void set_cluster_id(int64_t id) { cluster_id_ = id; }

  virtual std::string type() const { return xrt_node_->type(); }
  virtual std::string name() const { return xrt_node_->name(); }
  virtual XrtDevice device() const { return xrt_node_->device(); }

  size_t size() const { return folded_nodes_.size(); }
  const std::set<ClusterNode*>& folded_nodes() const { return folded_nodes_; }
  std::set<ClusterNode*>& folded_nodes() { return folded_nodes_; }

 private:
  const XrtNode* xrt_node_;
  int64_t cluster_id_ = -1;

  std::set<ClusterNode*> folded_nodes_;
  std::set<ClusterEdge*> in_edges_;
  std::set<ClusterEdge*> out_edges_;
};

namespace algorithm {
template <>
struct NodeTypeTrait<const ClusterNode> {
  typedef const ClusterEdge* pEdgeType;
};
}  // namespace algorithm

typedef std::shared_ptr<ClusterNode> ClusterNodePtr;
typedef std::shared_ptr<ClusterEdge> ClusterEdgePtr;

ClusterNodePtr BuildClusterNode(const XrtNode* node, int64_t id);

ClusterEdgePtr BuildClusterEdge(const ClusterNode* start,
                                const ClusterNode* end);

void SetupClusterEdge(ClusterEdge* cluster_edge, const XrtEdge* xrt_edge);

bool IsNodeDirectChildren(const ClusterNode* parent,
                          const ClusterNode* children);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_PASSES_CLUSTER_H_
