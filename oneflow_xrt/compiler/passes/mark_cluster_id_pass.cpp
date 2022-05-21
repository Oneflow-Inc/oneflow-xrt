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
#include "oneflow_xrt/compiler/passes/options.h"
#include "oneflow_xrt/graph/graph.h"

namespace oneflow {
namespace xrt {

class MarkClusterIdPass {
 public:
  MarkClusterIdPass() = default;

  void Run(XrtGraph* graph, const ClusteringOptions& options);

  const std::set<ClusterNode*>& Nodes() const { return root_nodes_; }

 private:
  void BuildClusterNodesAndEdges(XrtGraph* graph);
  void ClusteringSubgraphs(const ClusteringOptions& options);

  void RemoveInvalidClusterNodes(const ClusteringOptions& options);

  // rerank cluster id start by 0
  void RerankClusterIds();
  void UpdateNodeClusterIdInGraph(XrtGraph* graph);

  bool TryToFuseWithParent(ClusterNode* children, ClusterNode* parent,
                           const ClusteringOptions& options);

 private:
  // root cluster nodes
  std::set<ClusterNode*> root_nodes_;

  // all allocated nodes and edges that will always alive while
  // running the pass `MarkClusterIdPass`
  std::vector<ClusterNodePtr> allocated_nodes_;
  std::vector<ClusterEdgePtr> allocated_edges_;
};

namespace algorithm {
template <>
struct GraphTypeTrait<MarkClusterIdPass> {
  typedef ClusterNode* pNodeType;
  typedef ClusterEdge* pEdgeType;
};
}  // namespace algorithm

void MarkClusterIdPass::BuildClusterNodesAndEdges(XrtGraph* graph) {
  std::map<int64_t, ClusterNode*> cluster_nodes;
  algorithm::TopologyVisit(*graph, [&](XrtNode* node) {
    int64_t cluster_id = allocated_nodes_.size();
    auto cluster_node = BuildClusterNode(node, cluster_id);
    root_nodes_.insert(cluster_node.get());
    cluster_nodes[node->unique_id()] = cluster_node.get();
    allocated_nodes_.emplace_back(std::move(cluster_node));
  });

  for (ClusterNode* start : root_nodes_) {
    for (const XrtEdge* edge : start->xrt_node()->out_edges()) {
      int64_t unique_id = edge->end()->unique_id();
      ClusterNode* end = cluster_nodes.at(unique_id);

      auto cluster_edge = BuildClusterEdge(start, end);
      SetupClusterEdge(cluster_edge.get(), edge);

      start->AddOutEdge(cluster_edge.get());
      end->AddInEdge(cluster_edge.get());
      allocated_edges_.emplace_back(std::move(cluster_edge));
    }
  }
}

void MarkClusterIdPass::ClusteringSubgraphs(const ClusteringOptions& options) {
  for (int i = 0; i < options.max_iteration; ++i) {
    bool has_changed = false;
    std::vector<ClusterNode*> ordered_nodes;
    algorithm::TopologyVisit(*this, [&](ClusterNode* node) {
      if (!node->IsCompiled(options.engine) ||
          node->IsModelUpdate() /* skip model update op */) {
        return;
      }
      ordered_nodes.emplace_back(node);
    });

    for (int i = ordered_nodes.size() - 1; i >= 0; --i) {
      ClusterNode* node = ordered_nodes[i];
      std::set<ClusterNode*> candidate_parents;
      for (ClusterEdge* edge : node->in_edges()) {
        candidate_parents.insert(edge->start());
      }
      for (ClusterNode* parent : candidate_parents) {
        if (parent->IsCompiled(options.engine) &&
            (parent->size() + node->size()) <= options.maximum_nodes &&
            TryToFuseWithParent(node, parent, options)) {
          has_changed = true;
          root_nodes_.erase(node);
          break;
        }
      }
    }
    if (!has_changed) {
      break;
    }
  }
}

bool MarkClusterIdPass::TryToFuseWithParent(ClusterNode* children,
                                            ClusterNode* parent,
                                            const ClusteringOptions& options) {
  if (!options.ignore_pipeline) {
    for (const ClusterEdge* edge : parent->out_edges()) {
      if (edge->end() !=
              children && /* !children->IsReachable(*(edge->end())) */
          !IsNodeDirectChildren(children, edge->end())) {
        return false;
      }
    }
  }

  bool can_be_fusion = true;
  for (const ClusterEdge* edge : children->in_edges()) {
    if (edge->start() == parent) {
      can_be_fusion =
          can_be_fusion && !edge->is_fusion_disabled() && edge->IsIdentity();
    }
  }
  if (can_be_fusion) {
    return parent->TryMerge(*children);
  }
  return false;
}

void MarkClusterIdPass::RerankClusterIds() {
  int64_t rank = 0;
  for (ClusterNode* node : root_nodes_) {
    node->set_cluster_id(rank++);
  }
}

void MarkClusterIdPass::UpdateNodeClusterIdInGraph(XrtGraph* graph) {
  for (const ClusterNode* node : root_nodes_) {
    for (const ClusterNode* folded_node : node->folded_nodes()) {
      const_cast<XrtNode*>(folded_node->xrt_node())
          ->set_cluster_id(node->cluster_id());
    }
  }
}

void MarkClusterIdPass::RemoveInvalidClusterNodes(
    const ClusteringOptions& options) {
  const int min_nodes = options.minimum_nodes;
  const int max_nodes = options.maximum_nodes;
  std::vector<ClusterNode*> removing_clusters;
  for (ClusterNode* node : root_nodes_) {
    if (!node->IsCompiled(options.engine) || node->size() < min_nodes ||
        node->size() > max_nodes) {
      removing_clusters.emplace_back(node);
    }
  }
  for (ClusterNode* node : removing_clusters) {
    root_nodes_.erase(node);
  }
}

void MarkClusterIdPass::Run(XrtGraph* graph, const ClusteringOptions& options) {
  BuildClusterNodesAndEdges(graph);

  // clustering nodes iteratively
  ClusteringSubgraphs(options);

  RemoveInvalidClusterNodes(options);
  RerankClusterIds();

  UpdateNodeClusterIdInGraph(graph);
}

std::shared_ptr<XrtGraph> RunMarkClusterIdPass(
    const XrtGraph* graph, const ClusteringOptions& options) {
  auto new_graph = graph->clone();
  MarkClusterIdPass().Run(new_graph.get(), options);
  return new_graph;
}

}  // namespace xrt
}  // namespace oneflow
