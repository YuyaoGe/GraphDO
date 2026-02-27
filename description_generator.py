"""
High-level graph description generator.
Orchestrates data loading, ordering, and encoding to produce natural language descriptions.
Reads graph data from ./graph/{task}/ (local pre-sampled dataset).
"""

import json
import os
from typing import List, Dict, Any, Optional
import networkx as nx
from datetime import datetime

from data_loader import GraphDataLoader
from graph_ordering import GraphOrdering, PersonalizationStrategy
from text_encoder import GraphTextEncoder


class GraphDescriptionGenerator:
    """
    High-level orchestrator for generating graph descriptions with different orderings.
    """

    def __init__(self, graph_root: str = "./graph"):
        self.data_loader = GraphDataLoader(graph_root)
        self.available_orders = ["random", "bfs", "dfs", "pagerank", "ppr"]

    def generate_single_graph_descriptions(self, graph_info: Dict[str, Any],
                                           orders: List[str] = None) -> Dict[str, Any]:
        """Generate descriptions for a single graph with different orderings."""
        if orders is None:
            orders = self.available_orders

        graph = graph_info["networkx_graph"]
        is_directed = graph_info["is_directed"]

        ordering = GraphOrdering(graph)
        encoder = GraphTextEncoder("directed" if is_directed else "undirected")

        descriptions = {}
        for order in orders:
            try:
                ordered_edges = self._generate_ordered_edges(ordering, order, graph_info)
                description = self._encode_description(encoder, ordered_edges, graph_info)
                descriptions[order] = {
                    "description": description,
                    "ordered_edges": ordered_edges,
                    "order_method": order,
                }
            except Exception as e:
                print(f"Error generating {order} description for {graph_info['filename']}: {e}")
                descriptions[order] = {"error": str(e), "order_method": order}

        return descriptions

    def _generate_ordered_edges(self, ordering: GraphOrdering, order: str,
                                graph_info: Dict[str, Any]) -> List[tuple]:
        """Generate ordered edges using the specified ordering method."""
        if order == "random":
            return ordering.random_order()
        elif order == "bfs":
            return ordering.bfs_order(list(ordering.graph.nodes())[0])
        elif order == "dfs":
            return ordering.dfs_order(list(ordering.graph.nodes())[0])
        elif order == "pagerank":
            return ordering.pagerank_order()
        elif order == "ppr":
            task_params = self._extract_task_params(graph_info)
            personalization = PersonalizationStrategy.create_personalization(
                ordering.graph, graph_info["task"], task_params
            )
            return ordering.personalized_pagerank_order(personalization)
        elif order == "shortest_path_order":
            task_params = self._extract_task_params(graph_info)
            if "source" not in task_params or "target" not in task_params:
                raise ValueError("shortest_path_order requires a shortest_path task with source/target query")
            return ordering.shortest_path_order(task_params["source"], task_params["target"])
        elif order == "longest_path_order":
            task_params = self._extract_task_params(graph_info)
            if "source" not in task_params or "target" not in task_params:
                raise ValueError("longest_path_order requires a shortest_path task with source/target query")
            return ordering.longest_path_order(task_params["source"], task_params["target"])
        else:
            raise ValueError(f"Unknown ordering method: {order}")

    def _encode_description(self, encoder: GraphTextEncoder, ordered_edges: List[tuple],
                            graph_info: Dict[str, Any]) -> str:
        """Encode ordered edges into a natural language description."""
        if graph_info["is_weighted"]:
            graph = graph_info["networkx_graph"]
            weighted_edges = []
            for u, v in ordered_edges:
                w = graph[u][v].get("weight", 1) if graph.has_edge(u, v) else 1
                weighted_edges.append((u, v, w))
            return encoder.encode_weighted_edges(weighted_edges)
        else:
            return encoder.encode_unweighted_edges(ordered_edges)

    def _extract_task_params(self, graph_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific parameters for personalization/ordering."""
        task_type = graph_info["task"]
        graph_data = graph_info["graph_data"]
        graph = graph_info["networkx_graph"]
        params = {}

        if task_type == "connectivity":
            if "questions" in graph_data and graph_data["questions"]:
                params["query_nodes"] = graph_data["questions"][0]

        elif task_type == "shortest_path":
            if "query" in graph_data:
                source, target = graph_data["query"]
                params["source"] = source
                params["target"] = target
                try:
                    if nx.has_path(graph, source, target):
                        path = nx.shortest_path(graph, source, target,
                                                weight="weight" if graph_info["is_weighted"] else None)
                        params["path_nodes"] = path
                except Exception:
                    pass

        elif task_type == "cycle":
            try:
                cycles = nx.cycle_basis(graph) if isinstance(graph, nx.Graph) else list(nx.simple_cycles(graph))
                if cycles:
                    params["cycle_nodes"] = cycles[0]
            except Exception:
                pass

        return params

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_task_descriptions(self, task: str, max_graphs: int = None,
                                   orders: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate descriptions for all graphs in a task.

        Args:
            task: Task name (connectivity, cycle, shortest_path, hamilton, topology)
            max_graphs: Maximum number of graphs to process (None = all 100)
            orders: Ordering methods to use (None = all 5 defaults)

        Returns:
            List of graph description dicts
        """
        graphs = self.data_loader.get_all_graphs_for_task(task, max_graphs)
        results = []
        for i, graph_info in enumerate(graphs):
            print(f"  [{i+1}/{len(graphs)}] {graph_info['filename']}")
            descriptions = self.generate_single_graph_descriptions(graph_info, orders)
            results.append({
                "graph_id": graph_info["graph_id"],
                "filename": graph_info["filename"],
                "task": task,
                "num_nodes": graph_info["networkx_graph"].number_of_nodes(),
                "num_edges": graph_info["networkx_graph"].number_of_edges(),
                "is_weighted": graph_info["is_weighted"],
                "is_directed": graph_info["is_directed"],
                "graph_data": graph_info["graph_data"],
                "descriptions": descriptions,
            })
        return results

    def save_descriptions_to_json(self, descriptions: List[Dict[str, Any]], output_file: str):
        """Save graph descriptions to a JSON file."""
        output_data = {
            "descriptions": descriptions,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_graphs": len(descriptions),
                "generated_by": "GraphDO Description Generator",
            },
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved {len(descriptions)} graphs -> {output_file}")

    def generate_all_task_descriptions(self, tasks: List[str] = None,
                                       output_dir: str = "descriptions",
                                       max_graphs_per_task: int = None):
        """Generate and save descriptions for all (or specified) tasks."""
        os.makedirs(output_dir, exist_ok=True)
        if tasks is None:
            tasks = list(self.data_loader.TASK_CONFIGS.keys())

        for task in tasks:
            print(f"\nTask: {task}")
            try:
                descriptions = self.generate_task_descriptions(task, max_graphs=max_graphs_per_task)
                output_file = os.path.join(output_dir, f"{task}_descriptions.json")
                self.save_descriptions_to_json(descriptions, output_file)
            except Exception as e:
                print(f"  ERROR: {e}")

        print(f"\nAll descriptions saved to {output_dir}/")
