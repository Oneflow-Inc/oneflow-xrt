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
#ifndef ONEFLOW_XRT_GRAPH_GRAPH_H_
#define ONEFLOW_XRT_GRAPH_GRAPH_H_

#include <vector>

#include "oneflow_xrt/graph/algorithm.h"
#include "oneflow_xrt/graph/argument.h"
#include "oneflow_xrt/graph/node.h"

namespace oneflow {
namespace xrt {

class XrtGraph : public std::enable_shared_from_this<XrtGraph> {
 public:
  XrtGraph() = default;
  virtual ~XrtGraph() = default;

  XrtNode* Node(int64_t node_id);
  const XrtNode* Node(int64_t node_id) const;

  XrtNode* AddNode(const std::string& name);
  XrtNode* AddNode(const OperatorConf& conf);
  XrtNode* AddNode(std::unique_ptr<XrtNode>&& node);

  XrtNode* AddEntryNode(const std::string& name);
  XrtNode* AddReturnNode(const std::string& name);

  XrtEdge* AddEdge();
  XrtEdge* AddEdge(const XrtNode* start, const XrtNode* end);

  XrtEdge* Connect(const XrtNode* start, const XrtNode* end);
  XrtEdge* Connect(const XrtNode* start, const XrtNode* end,
                   const Argument& arg);
  void Disconnect(const XrtEdge* edge);

  // create a subgraph for node which unique id is `node_id`
  XrtGraph* AddSubgraphForNode(int64_t node_id);

  const std::vector<XrtNode*>& Nodes() const { return nodes_; }
  std::vector<XrtNode*>& Nodes() { return nodes_; }

  const std::vector<XrtEdge*>& Edges() const { return edges_; }
  std::vector<XrtEdge*>& Edges() { return edges_; }

  std::string ToDot() const;

  std::vector<Argument> Arguments() const;

 protected:
  std::vector<XrtNode*> nodes_;
  // all allocated nodes in the graph. The node unique id is related to it's
  // index in the vector. The node in `nodes_` can be nullptr since we will
  // always keep it in `nodes_` even if it has been removed from the graph
  std::vector<std::unique_ptr<XrtNode>> allocated_nodes_;

  std::vector<XrtEdge*> edges_;
  // all allocated edges in the graph. The edge unique id is related to it's
  // index in the vector. And the xrt edge in `edges_` can also be nullptr
  std::vector<std::unique_ptr<XrtEdge>> allocated_edges_;

  // all allocated subgraphs. The key of the map means node unique id, and the
  // value is the subgraph which belongs to the node
  std::map<int64_t, std::unique_ptr<XrtGraph>> subgraphs_;
};

namespace algorithm {
template <>
struct GraphTypeTrait<XrtGraph> {
  typedef XrtNode* pNodeType;
  typedef XrtEdge* pEdgeType;
};

template <>
struct GraphTypeTrait<const XrtGraph> {
  typedef const XrtNode* pNodeType;
  typedef const XrtEdge* pEdgeType;
};
}  // namespace algorithm

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_GRAPH_H_
