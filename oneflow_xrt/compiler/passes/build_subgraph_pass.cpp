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
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "oneflow_xrt/common/typedef.h"
#include "oneflow_xrt/compiler/passes/options.h"
#include "oneflow_xrt/graph/graph.h"
#include "oneflow_xrt/graph/node.h"

namespace oneflow {
namespace xrt {

class BuildSubGraphPass {
 public:
  BuildSubGraphPass() = default;

  void Run(XrtGraph* graph, const ClusteringOptions& options);

 private:
  void RebuildSubgraphInputs(XrtNode* node, XrtNode* n, XrtGraph* sub_graph,
                             std::map<int64_t, XrtNode*>* sub_graph_nodes);

  void RebuildSubgraphOutputs(XrtNode* node, XrtNode* n, XrtGraph* sub_graph,
                              std::map<int64_t, XrtNode*>* sub_graph_nodes);

  void CreateLaunchNodes(XrtGraph* graph,
                         std::map<int64_t, XrtNode*>* launch_nodes);

  void DivideEntryAndReturnNodes(XrtGraph* sub_graph);
  void DumpSubgraphs(const XrtGraph* graph, const std::string& path);
};

void BuildSubGraphPass::Run(XrtGraph* graph, const ClusteringOptions& options) {
  // create xrt launch nodes
  std::map<int64_t, XrtNode*> launch_nodes;
  CreateLaunchNodes(graph, &launch_nodes);

  // redirect all outer edges of the launch nodes
  std::map<int64_t, std::set<XrtNode*>> folded_nodes;
  for (XrtNode* node : graph->Nodes()) {
    int64_t cluster_id = node->cluster_id();
    if (cluster_id != -1 && node->type() != _XrtLaunchOpType) {
      XrtNode* launch_node = launch_nodes[cluster_id];
      // redirect input edges
      for (XrtEdge* edge : node->in_edges()) {
        XrtNode* start = edge->start();
        if (start->cluster_id() != cluster_id) {
          edge->SetEnd(launch_node);
          launch_node->AddInEdge(edge);
        }
      }
      // redirect output edges
      for (XrtEdge* edge : node->out_edges()) {
        XrtNode* end = edge->end();
        if (end->cluster_id() != cluster_id) {
          edge->SetStart(launch_node);
          launch_node->AddOutEdge(edge);
        }
      }
      folded_nodes[cluster_id].insert(node);
    }
  }

  // build subgraph for xrt launch nodes and repair error connections
  // caused by redirect. Add argument nodes and create connections
  // between them and the folded nodes
  for (auto& kv : folded_nodes) {
    int64_t cluster_id = kv.first;
    XrtNode* launch_node = launch_nodes[cluster_id];
    XrtGraph* sub_graph = graph->AddSubgraphForNode(launch_node->unique_id());
    sub_graph->set_engine(options.engine);

    std::map<int64_t, XrtNode*> sub_graph_nodes;
    for (XrtNode* n : kv.second) {
      XrtNode* node = sub_graph->AddNode(n->clone());
      sub_graph_nodes[n->unique_id()] = node;

      // rebuild inputs if the end node of input edges has been changed,
      // otherwise repair input for the node of the subgraph
      RebuildSubgraphInputs(node, n, sub_graph, &sub_graph_nodes);

      // rebuild outputs same as rebuilding the inputs
      RebuildSubgraphOutputs(node, n, sub_graph, &sub_graph_nodes);
    }
    // divide argument nodes if they have multiple inputs or outputs with
    // different argument (or `LogicalBlobId`), and then fill their names
    DivideEntryAndReturnNodes(sub_graph);
  }

  for (const XrtNode* node : graph->Nodes()) {
    CHECK(!node->IsReachable(*node));
  }

  if (!options.dump_subgraph_dir.empty()) {
    DumpSubgraphs(graph, options.dump_subgraph_dir);
  }
}

void BuildSubGraphPass::CreateLaunchNodes(
    XrtGraph* graph, std::map<int64_t, XrtNode*>* launch_nodes) {
  std::map<int64_t, XrtDevice> cluster_ids;
  for (XrtNode* node : graph->Nodes()) {
    int64_t cluster_id = node->cluster_id();
    if (cluster_id != -1) {
      cluster_ids.emplace(cluster_id, node->device());
    }
  }

  for (const auto& pair : cluster_ids) {
    int64_t cluster_id = pair.first;
    XrtNode* launch_node =
        graph->AddNode(absl::StrCat(_XrtLaunchPrefix, cluster_id));
    launch_node->set_cluster_id(cluster_id);
    launch_node->set_device(pair.second);
    launch_node->set_type(_XrtLaunchOpType);
    launch_nodes->emplace(cluster_id, launch_node);
  }
}

void BuildSubGraphPass::DivideEntryAndReturnNodes(XrtGraph* sub_graph) {
  // find all argument and return nodes
  std::vector<XrtNode*> nodes;
  for (XrtNode* node : sub_graph->Nodes()) {
    if (node->IsEntryNode() || node->IsReturnNode()) {
      nodes.emplace_back(node);
    }
  }
  // start to divide nodes
  for (XrtNode* node : nodes) {
    std::list<XrtEdge*> in_edges = node->in_edges();
    std::list<XrtEdge*> out_edges = node->out_edges();
    // argument node should has either inputs or outputs
    CHECK(in_edges.size() == 0 || out_edges.size() == 0);

    // clear node input and output edges, then rebuild them
    node->ClearInEdges();
    node->ClearOutEdges();

    std::unordered_map<Argument, XrtNode*> divided_nodes;
    for (XrtEdge* edge : in_edges) {
      const Argument& arg = edge->argument();
      if (node->in_edges().size() == 0) {
        node->set_name(arg.name());
        divided_nodes.emplace(arg, node);
      }
      const auto& it = divided_nodes.find(arg);
      if (it == divided_nodes.end()) {
        XrtNode* argument = sub_graph->AddReturnNode(arg.name());
        argument->set_device(node->device());
        argument->AddInEdge(edge);
        edge->SetEnd(argument);
        divided_nodes.emplace(arg, argument);
      } else {
        it->second->AddInEdge(edge);
        edge->SetEnd(it->second /* consumer */);
      }
    }

    for (XrtEdge* edge : out_edges) {
      const Argument& arg = edge->argument();
      if (node->out_edges().size() == 0) {
        node->set_name(arg.name());
        divided_nodes.emplace(arg, node);
      }
      const auto& it = divided_nodes.find(arg);
      if (it == divided_nodes.end()) {
        XrtNode* argument = sub_graph->AddEntryNode(arg.name());
        argument->set_device(node->device());
        argument->AddOutEdge(edge);
        edge->SetStart(argument);
        divided_nodes.emplace(arg, argument);
      } else {
        it->second->AddOutEdge(edge);
        edge->SetStart(it->second /* producer */);
      }
    }
  }
}

void BuildSubGraphPass::RebuildSubgraphInputs(
    XrtNode* node, XrtNode* n, XrtGraph* sub_graph,
    std::map<int64_t, XrtNode*>* sub_graph_nodes) {
  for (XrtEdge* e : n->in_edges()) {
    int64_t start_id = e->start()->unique_id();
    // check if the edge had been redirected
    if (e->end()->unique_id() != n->unique_id()) {
      XrtNode* argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddEntryNode("^unused");
        argument->set_device(e->start()->device());
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(argument, node, e->argument());
    } else {
      if (sub_graph_nodes->count(start_id) != 0) {
        XrtNode* start = (*sub_graph_nodes)[start_id];
        sub_graph->Connect(start, node, e->argument());
      }
    }
  }
}

void BuildSubGraphPass::RebuildSubgraphOutputs(
    XrtNode* node, XrtNode* n, XrtGraph* sub_graph,
    std::map<int64_t, XrtNode*>* sub_graph_nodes) {
  for (XrtEdge* e : n->out_edges()) {
    // check if the edge had been redirected
    if (e->start()->unique_id() != n->unique_id()) {
      // start_id is the launch node id
      int64_t start_id = e->start()->unique_id();
      XrtNode* argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddReturnNode("^unused");
        argument->set_device(e->start()->device());
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(node, argument, e->argument());
    } else {
      int64_t end_id = e->end()->unique_id();
      if (sub_graph_nodes->count(end_id) != 0) {
        XrtNode* end = (*sub_graph_nodes)[end_id];
        sub_graph->Connect(node, end, e->argument());
      }
    }
  }
}

void BuildSubGraphPass::DumpSubgraphs(const XrtGraph* graph,
                                      const std::string& path) {
  for (const XrtNode* node : graph->Nodes()) {
    if (node->type() == _XrtLaunchOpType) {
      std::string file = absl::StrCat(path, "/cluster_", node->cluster_id());
      std::ofstream ost(file.c_str());
      CHECK(ost.good())
          << "can not dump subgraph, please check if the directory (" << path
          << ") exists";
      ost << node->sub_graph()->ToDot();
    }
  }
}

std::shared_ptr<XrtGraph> RunBuildSubGraphPass(
    const XrtGraph* graph, const ClusteringOptions& options) {
  auto new_graph = graph->clone();
  BuildSubGraphPass().Run(new_graph.get(), options);
  return new_graph;
}

}  // namespace xrt
}  // namespace oneflow
