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
#include "oneflow_xrt/compiler/passes/cluster.h"

namespace oneflow {
namespace xrt {

bool ClusterEdge::IsIdentity() const {
  return is_control_edge() || (start()->device() == end()->device() &&
                               start_nd_sbp() == end_nd_sbp() &&
                               start_time_shape() == end_time_shape());
}

void ClusterNode::AddInEdge(const ClusterEdge* edge) {
  in_edges_.insert(const_cast<ClusterEdge*>(edge));
}

void ClusterNode::AddOutEdge(const ClusterEdge* edge) {
  out_edges_.insert(const_cast<ClusterEdge*>(edge));
}

void ClusterNode::EraseInEdge(const ClusterEdge* edge) {
  in_edges_.erase(const_cast<ClusterEdge*>(edge));
}

void ClusterNode::EraseOutEdge(const ClusterEdge* edge) {
  out_edges_.erase(const_cast<ClusterEdge*>(edge));
}

bool ClusterNode::IsSatisfySbpPolicy() const {
  std::map<int64_t, std::set<bool>> connection_kinds;
  for (ClusterEdge* edge : in_edges_) {
    int64_t parent_id = edge->start()->cluster_id();
    connection_kinds[parent_id].insert(edge->IsIdentity());
    if (connection_kinds[parent_id].size() > 1) {
      return false;
    }
  }
  for (ClusterEdge* edge : out_edges_) {
    int64_t children_id = edge->end()->cluster_id();
    connection_kinds[children_id].insert(edge->IsIdentity());
    if (connection_kinds[children_id].size() > 1) {
      return false;
    }
  }
  return true;
}

bool ClusterNode::IsReachable(const ClusterNode& target) const {
  return algorithm::IsReachable(this, &target);
}

class ClusterMergeNode : public ClusterNode {
 public:
  struct EdgeSnapshot {
    ClusterNode* from;
    ClusterNode* to;
    ClusterEdge* edge;
  };

  ClusterMergeNode(ClusterNode* lhs, ClusterNode* rhs)
      : ClusterNode(-100), lhs_(lhs), rhs_(rhs) {
    BuildInputEdges();
    BuildOutputEdges();
  }

  virtual ~ClusterMergeNode() { Fallback(); }

  void Fallback() {
    for (const EdgeSnapshot& snapshot : snapshot_edges_) {
      snapshot.edge->SetStart(snapshot.from);
      snapshot.edge->SetEnd(snapshot.to);
    }
    snapshot_edges_.clear();
  }

  void Complete() {
    Fallback();
    lhs_->Merge(*rhs_);
  }

 private:
  void BuildInputEdges() {
    for (ClusterEdge* edge : lhs_->in_edges()) {
      if (edge->start() != rhs_) {
        SnapshotEdge(edge);
        edge->SetEnd(this);
        AddInEdge(edge);
      }
    }
    for (ClusterEdge* edge : rhs_->in_edges()) {
      if (edge->start() != lhs_) {
        SnapshotEdge(edge);
        edge->SetEnd(this);
        AddInEdge(edge);
      }
    }
  }

  void BuildOutputEdges() {
    for (ClusterEdge* edge : lhs_->out_edges()) {
      if (edge->end() != rhs_) {
        SnapshotEdge(edge);
        edge->SetStart(this);
        AddOutEdge(edge);
      }
    }
    for (ClusterEdge* edge : rhs_->out_edges()) {
      if (edge->end() != lhs_) {
        SnapshotEdge(edge);
        edge->SetStart(this);
        AddOutEdge(edge);
      }
    }
  }

  void SnapshotEdge(ClusterEdge* edge) {
    EdgeSnapshot snapshot;
    snapshot.from = edge->start();
    snapshot.to = edge->end();
    snapshot.edge = edge;
    snapshot_edges_.emplace_back(snapshot);
  }

  ClusterNode* lhs_;
  ClusterNode* rhs_;
  std::vector<EdgeSnapshot> snapshot_edges_;
};

void ClusterNode::Merge(ClusterNode& other) {
  for (ClusterEdge* edge : other.in_edges()) {
    if (edge->start() != this) {
      edge->SetEnd(this);
      AddInEdge(edge);
    } else {
      EraseOutEdge(edge);
    }
  }
  for (ClusterEdge* edge : other.out_edges()) {
    if (edge->end() != this) {
      edge->SetStart(this);
      AddOutEdge(edge);
    } else {
      EraseInEdge(edge);
    }
  }

  FoldNodes(other.folded_nodes());
}

bool ClusterNode::TryMerge(ClusterNode& other) {
  ClusterMergeNode node(this, &other);
  if (!node.IsSatisfySbpPolicy() || node.IsReachable(node)) {
    // explicit fallback
    node.Fallback();
    return false;
  }
  node.Complete();
  return true;
}

ClusterNodePtr BuildClusterNode(const XrtNode* node, int64_t id) {
  return std::make_shared<ClusterNode>(node, id);
}

ClusterEdgePtr BuildClusterEdge(const ClusterNode* start,
                                const ClusterNode* end) {
  return std::make_shared<ClusterEdge>(const_cast<ClusterNode*>(start),
                                       const_cast<ClusterNode*>(end));
}

void SetupClusterEdge(ClusterEdge* cluster_edge, const XrtEdge* xrt_edge) {
  cluster_edge->set_is_control_edge(xrt_edge->IsControlEdge());
  cluster_edge->set_start_nd_sbp(xrt_edge->nd_sbp[0]);
  cluster_edge->set_end_nd_sbp(xrt_edge->nd_sbp[1]);
  cluster_edge->set_start_time_shape(xrt_edge->time_shape[0]);
  cluster_edge->set_end_time_shape(xrt_edge->time_shape[1]);
}

bool IsNodeDirectChildren(const ClusterNode* parent,
                          const ClusterNode* children) {
  for (const ClusterEdge* edge : parent->out_edges()) {
    if (edge->end() == children) {
      return true;
    }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
